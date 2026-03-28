# RL2 Rollout 模块研读笔记

> 文档版本：2026-03-28  
> 目标文件：`RL2/workers/rollout.py`

---

## 一、整体架构概览

Rollout 是 RL 训练中的核心环节，负责：

1. 从数据加载器获取 prompt
2. 让 Actor 模型生成 response（通过 SGLang 推理引擎）
3. 与环境交互获取 reward
4. 收集 trajectory 数据供训练使用

### 1.1 核心类与数据结构

| 类名 | 文件位置 | 职责 |
|------|----------|------|
| `Rollout` | `rollout.py` | 任务编排、推理服务器管理 |
| `SampleGroup` | `rl.py` | 一个 prompt 的多个 response |
| `Sample` | `rl.py` | 单个 trajectory，包含 state/action/reward 序列 |

---

## 二、Rollout 类详解

### 2.1 初始化流程（`__init__`）

```python
def __init__(self, config: DictConfig):
```

**执行步骤**：

1. **准备设备网格** `_prepare_device_mesh()`
   - world_size = dp_size × tp_size
   - dp：数据并行度（模型副本数）
   - tp：张量并行度（模型切分到的 GPU 数）
   - 示例：8 GPU, tp_size=2 → dp_size=4, tp_size=2

2. **准备环境变量** `_prepare_environment_variables()`
   - 设置 `CUDA_VISIBLE_DEVICES`
   - 为每个 tp rank 分配不同 GPU
   - 通过 `all_gather_object` 同步分配结果

3. **Rank 0 专属初始化**
   - 加载 tokenizer
   - 创建 train/test 数据加载器
   - 加载环境定义（从 `env_path`）
   - 启动 Router 进程

4. **启动 SGLang 服务器** `_launch_server_process()`
   - 每个 tp_local_rank 启动一个服务器实例
   - 服务器负责加载模型权重、处理生成请求、管理 KV cache

### 2.2 推理主流程（`__call__`）

```python
async def __call__(
    self,
    train: bool,
    step: int
) -> Optional[Tuple[Optional[Dict[str, torch.Tensor]], Optional[torch.Tensor]]]:
```

**核心逻辑**：

```
┌─────────────────────────────────────────────────────────┐
│                    Rank 0: 任务调度                      │
├─────────────────────────────────────────────────────────┤
│  1. 从 dataloader 获取 SampleGroup                       │
│  2. 为每个 SampleGroup 创建 asyncio 任务                  │
│  3. 并发生成 response + 环境交互                         │
│  4. 收集结果并打包成 tensor_dict                         │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              其他 Rank：同步等待                          │
│  dist.barrier(group=get_gloo_group())                  │
└─────────────────────────────────────────────────────────┘
```

**关键代码片段**：

```python
# 调度任务
def _schedule_tasks(sample_groups):
    for sample_group in sample_groups:
        pendings.add(
            asyncio.create_task(
                sample_group.generate(self.router_url, self.env.generate)
            )
        )

# 等待任务完成
done, pendings = await asyncio.wait(
    pendings, return_when=asyncio.FIRST_COMPLETED
)

# 处理完成的任务
for task in done:
    sample_group = task.result()
    all_tensor_dicts_delta, metrics_delta = (
        sample_group.to_all_tensor_dicts_and_metrics()
    )
```

**返回数据格式**：

```python
tensor_dict = {
    "states": torch.Tensor,      # 输入 token IDs
    "actions": torch.Tensor,     # 生成的 action token IDs
    "action_mask": torch.Tensor, # 有效 token 掩码
    "llm_logps": torch.Tensor,  # log probabilities
    "rewards": torch.Tensor     # 环境返回的 reward
}

cu_seqs = torch.Tensor  # cumulative sequence lengths，用于序列打包
```

### 2.3 模型权重更新（`update`）

```python
def update(self, named_tensor_generator: Generator[Tuple[str, torch.Tensor], None, None]):
```

**执行步骤**：

1. 释放 KV cache 占用（避免 OOM）
2. 按 dtype 和 bucket 分批传输权重
3. 恢复 KV cache 占用

```python
# 按 bucket 分批传输
for name, tensor in named_tensor_generator:
    param_size = tensor.numel() * tensor.element_size()
    
    if bucket_size > 0 and bucket_size + param_size > (self.config.bucket_size << 20):
        _update_tensor_bucket(dtype_to_named_tensors)
        dtype_to_named_tensors = defaultdict(list)
        bucket_size = 0
    
    # DTensor 需要 redispatch 到 Replicate 布局
    if isinstance(tensor, DTensor):
        tensor = tensor.redistribute(
            placements=[Replicate()] * tensor.device_mesh.ndim,
            async_op=True
        ).to_local()
```

---

## 三、Sample 与 SampleGroup 详解

### 3.1 Sample 数据结构

位置：`RL2/datasets/rl.py`

```python
@dataclass
class Sample:
    # 初始化相关
    sample: Dict[str, Any]           # 原始数据
    
    # 环境交互相关
    state_text: str                  # 当前输入状态
    action_text: str                  # 生成的 action
    
    # 训练相关
    state_dict: Dict[str, List[int | float]]   # 单步状态
    state_dicts: List[Dict[str, List[int | float]]]  # 多步状态
    
    # 日志相关
    turn: int                         # 当前 turn 数
    metrics: Dict[str, List]         # 指标
    
    # Partial rollout 相关
    status: Status                   # RUNNING / ABORTED / DONE
```

### 3.2 核心生成循环

位置：`RL2/datasets/rl.py` 的 `base_generate` 函数

```python
while True:
    # 1. LLM 生成
    response = await async_request(
        router_url,
        "generate",
        json={
            "input_ids": sample.state_dict["states"],
            "sampling_params": {**sampling_params, ...},
            "return_logprob": True
        }
    )
    add_llm_response(sample, response)
    
    # 2. 环境交互
    response = await env_step_fn(sample)
    add_env_response(tokenizer, sample, response)
    
    # 3. 判断是否结束
    if sample.status == Sample.Status.DONE:
        return
```

### 3.3 环境交互函数

位置：`RL2/datasets/rl.py` 的 `add_env_response`

```python
def add_env_response(tokenizer, sample, response):
    # 更新最后一步的 reward
    sample.state_dict["rewards"][-1] = response["reward"]
    
    if response["done"]:
        # episode 结束
        sample.status = Sample.Status.DONE
        sample.state_dicts.append(sample.state_dict)
        return
    
    # 处理多步情况
    if response["next_state"].startswith(sample.state_text + sample.action_text):
        # 状态连续，追加到当前 state_dict
        ...
    else:
        # 状态不连续，创建新的 state_dict（多序列）
        sample.state_dicts.append(sample.state_dict)
        sample.state_dict = initialize_state_dict(tokenizer, response["next_state"])
```

---

## 四、环境定义示例

位置：`envs/orz.py`

```python
async def env_step(sample: Sample) -> Dict[str, Any]:
    """ORZ 环境：验证数学答案"""
    reward = float(
        verify(
            parse(sample.sample["answer"]),
            parse(sample.action_text)
        )
    )
    return {
        "next_state": None,
        "done": True,
        "reward": reward
    }

generate = partial(base_generate, env_step_fn=env_step)
```

---

## 五、Train-Inference Mismatch 注入点分析

根据代码分析，可以在以下几个位置引入扰动：

### 注入点 1：LLM 生成参数扰动

| 项目 | 说明 |
|------|------|
| **位置** | `RL2/datasets/rl.py:181-193` |
| **函数** | `base_generate` 中的 `sampling_params` |
| **可扰动项** | `temperature`, `max_new_tokens`, `top_p`, `top_k` |
| **影响** | 改变生成的 action，从而影响后续环境交互 |

```python
sampling_params = OmegaConf.to_container(config.sampling_params)
response = await async_request(
    router_url,
    "generate",
    json={
        "input_ids": sample.state_dict["states"],
        "sampling_params": {
            **sampling_params,
            "max_new_tokens": ...,  # 可扰动
            # temperature, top_p, top_k 可扰动
        },
    }
)
```

### 注入点 2：环境 Step 扰动

| 项目 | 说明 |
|------|------|
| **位置** | `envs/*.py` 下的各个环境文件 |
| **函数** | `env_step` |
| **可扰动项** | `reward` 值、`next_state` |
| **影响** | 直接影响策略梯度计算 |

```python
async def env_step(sample: Sample) -> Dict[str, Any]:
    # 原始 reward
    reward = float(verify(...))
    
    # 可在此处注入扰动
    # reward = reward + noise(...)
    
    return {
        "next_state": None,
        "done": True,
        "reward": reward
    }
```

### 注入点 3：Response 后处理扰动

| 项目 | 说明 |
|------|------|
| **位置** | `RL2/datasets/rl.py:67-103` |
| **函数** | `add_llm_response` |
| **可扰动项** | `action_text`、`logps` |
| **影响** | 影响训练数据的质量 |

```python
def add_llm_response(sample: Sample, response: Dict[str, Any]):
    # 原始 action_text
    sample.action_text = sample.previous_action_text + response["text"]
    
    # 可在此处注入扰动
    # sample.action_text = perturb_action(sample.action_text)
    
    # 处理 logps
    if "output_token_logprobs" in meta_info:
        logp, action, _ = map(list, zip(*meta_info["output_token_logprobs"]))
        sample.state_dict["states"].extend(action)
        sample.state_dict["actions"].extend(action)
        sample.state_dict["logps"].extend(logp)
```

### 注入点 4：Rollout 层整体扰动

| 项目 | 说明 |
|------|------|
| **位置** | `RL2/workers/rollout.py:178-290` |
| **函数** | `Rollout.__call__` |
| **可扰动项** | `sample_group` 的结果 |
| **影响** | 全局性的数据扰动 |

```python
# 在收集 sample_group 结果后、转换为 tensor_dict 之前注入扰动
sample_group = task.result()

# 可在此处注入扰动
# sample_group = perturb_sample_group(sample_group)

all_tensor_dicts_delta, metrics_delta = (
    sample_group.to_all_tensor_dicts_and_metrics()
)
```

---

## 六、关键配置项

位置：`RL2/trainer/config/ppo.yaml`

| 配置项 | 说明 | 示例值 |
|--------|------|--------|
| `rollout.train.sampling_params.temperature` | 生成温度 | 1.0 |
| `rollout.train.sampling_params.max_new_tokens` | 最大生成长度 | null |
| `rollout.train.responses_per_prompt` | 每 prompt 的 response 数 | 1 |
| `rollout.env_path` | 环境定义文件路径 | null |
| `rollout.train.dynamic_filtering` | 动态过滤（reward 全同则跳过） | true |
| `rollout.train.partial_rollout` | 部分 rollou | false |

---

## 七、分布式协作机制

### 7.1 设备网格

```
Device Mesh (dp_size=4, tp_size=2)

       GPU 0   GPU 1
       ─────────────
DP 0   Worker  Worker
DP 1   Worker  Worker
DP 2   Worker  Worker
DP 3   Worker  Worker

- 每列是一个 TP Group（共同处理一个模型）
- 每行是一个 DP Group（处理不同数据）
```

### 7.2 同步机制

- **Gloo Group**：用于避免影响 SGLang 服务器的同步操作
- **Barrier**：确保所有 rank 在关键节点同步
- **Broadcast/Gather**：用于在 rank 0 和其他 rank 之间传递数据

---

## 八、参考资料

- 原仓库：[ChenmienTan/RL2](https://github.com/ChenmienTan/RL2)
- SGLang：[sgl-project/sglang](https://github.com/sgl-project/sglang)
- RL2 数据集模块：`RL2/datasets/rl.py`
- PPO 配置示例：`RL2/trainer/config/ppo.yaml`
