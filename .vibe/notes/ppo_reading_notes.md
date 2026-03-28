# RL2 PPO 模块研读笔记

> 文档版本：2026-03-28  
> 目标文件：`RL2/trainer/ppo.py`

---

## 一、整体架构概览

PPO (Proximal Policy Optimization) 是 RL2 框架中的核心训练算法。整个训练流程涉及以下组件：

| 组件 | 文件位置 | 职责 |
|------|----------|------|
| `PPOTrainer` | `trainer/ppo.py` | 训练主循环编排 |
| `Trainer` | `trainer/base.py` | 基础类，处理检查点、日志等 |
| `FSDPActor` | `workers/fsdp/actor.py` | Actor 模型训练 |
| `FSDPCritic` | `workers/fsdp/critic.py` | Critic 模型训练（仅 GAE 需要） |
| `Rollout` | `workers/rollout.py` | 交互数据生成 |

### 1.1 训练流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                        PPO Training Loop                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  for step in range(total_steps):                                │
│                                                                  │
│    ┌─────────────────────────────────────────────────────────┐  │
│    │  1. Rollout (async)                                     │  │
│    │     tensor_dict, cu_seqs = await rollout(True, step)    │  │
│    └─────────────────────────────────────────────────────────┘  │
│                          │                                      │
│                          ▼                                      │
│    ┌─────────────────────────────────────────────────────────┐  │
│    │  2. Compute Reference Logps (if kl.coef > 0)           │  │
│    │     tensor_dict = ref_actor.compute_logps(tensor_dict)  │  │
│    └─────────────────────────────────────────────────────────┘  │
│                          │                                      │
│                          ▼                                      │
│    ┌─────────────────────────────────────────────────────────┐  │
│    │  3. Compute Values (if estimator == "gae")              │  │
│    │     tensor_dict = critic.compute_values(tensor_dict)    │  │
│    └─────────────────────────────────────────────────────────┘  │
│                          │                                      │
│                          ▼                                      │
│    ┌─────────────────────────────────────────────────────────┐  │
│    │  4. Compute Actor Logps (if kl.coef > 0 or            │  │
│    │                               update_per_rollout > 1)    │  │
│    │     tensor_dict = actor.compute_logps(tensor_dict)      │  │
│    └─────────────────────────────────────────────────────────┘  │
│                          │                                      │
│                          ▼                                      │
│    ┌─────────────────────────────────────────────────────────┐  │
│    │  5. Compute Advantages (rank 0 only)                    │  │
│    │     compute_advantages(config, tensor_dict, cu_seqs)   │  │
│    └─────────────────────────────────────────────────────────┘  │
│                          │                                      │
│                          ▼                                      │
│    ┌─────────────────────────────────────────────────────────┐  │
│    │  6. PPO Update                                          │  │
│    │     actor.ppo_update(tensor_dict, step)                 │  │
│    │     critic.ppo_update(tensor_dict, step)  # if GAE     │  │
│    └─────────────────────────────────────────────────────────┘  │
│                          │                                      │
│                          ▼                                      │
│    ┌─────────────────────────────────────────────────────────┐  │
│    │  7. Update Rollout Weights                              │  │
│    │     actor.update_rollout(rollout, step)                 │  │
│    └─────────────────────────────────────────────────────────┘  │
│                          │                                      │
│                          ▼                                      │
│    ┌─────────────────────────────────────────────────────────┐  │
│    │  8. Test Eval (if step % test_freq == 0)               │  │
│    │     await rollout(False, step)                          │  │
│    └─────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、核心类详解

### 2.1 PPOTrainer

位置：`RL2/trainer/ppo.py`

```python
class PPOTrainer(Trainer):

    def __init__(self, config: DictConfig):
        super().__init__(config)

        if not config.trainer.eval_only:
            # 初始化 Actor
            self.actor = initialize_actor(config.actor, True)
            self.actor.prepare_scheduler(self.config.trainer.total_steps)
            
            # 如果使用 KL 散度约束，初始化 Reference Actor
            if config.actor.kl.coef > 0:
                self.ref_actor = initialize_actor(config.ref_actor, False)
            
            # 如果使用 GAE 优势估计，初始化 Critic
            if config.adv.estimator == "gae":
                self.critic = initialize_critic(config.critic)
                self.critic.prepare_scheduler(self.config.trainer.total_steps)

        # 初始化 Rollout worker
        self.rollout = initialize_rollout(self.config.rollout)
```

### 2.2 训练主循环

```python
@shutdown_processes_when_exit
@with_session
async def train(self):
```

**关键步骤**：

1. **加载检查点**（可选）
   ```python
   initial = self.load_ckpt(
       (self.actor, self.critic) if self.config.adv.estimator == "gae"
       else (self.actor,)
   )
   ```

2. **Rollout 阶段**
   ```python
   tensor_dict, cu_seqs = await self.rollout(True, step)
   ```
   - 异步执行，生成训练数据
   - 返回 `tensor_dict`：包含 states、actions、action_mask、llm_logps、rewards
   - 返回 `cu_seqs`：累计序列长度，用于打包

3. **计算参考 Logps**（如果启用 KL 约束）
   ```python
   if self.config.actor.kl.coef > 0:
       tensor_dict = self.ref_actor.compute_logps(tensor_dict, step)
   ```
   - Reference Actor 是原始策略的副本
   - 用于计算 KL 散度约束

4. **计算 Value**（如果使用 GAE）
   ```python
   if self.config.adv.estimator == "gae":
       tensor_dict = self.critic.compute_values(tensor_dict, step)
   ```
   - Critic 估计每个状态的值函数 V(s)
   - 用于 GAE (Generalized Advantage Estimation)

5. **重新计算 Actor Logps**
   ```python
   if self.config.actor.kl.coef > 0 or self.config.actor.update_per_rollout > 1:
       tensor_dict = self.actor.compute_logps(tensor_dict, step)
   ```
   - 当 kl.coef > 0 时：需要计算 old_logps 用于 PPO 损失
   - 当 update_per_rollout > 1 时：多次更新需要重新计算

6. **计算 Advantages**（仅 rank 0）
   ```python
   if dist.get_rank() == 0:
       compute_advantages(self.config, tensor_dict, cu_seqs, step)
   ```
   - 根据配置使用 REINFORCE 或 GAE
   - 更新 tensor_dict 中的 advantages

7. **PPO 更新**
   ```python
   self.actor.ppo_update(tensor_dict, step)
   if self.config.adv.estimator == "gae":
       self.critic.ppo_update(tensor_dict, step)
   ```

8. **保存检查点**
   ```python
   self.save_ckpt(workers, step)
   ```

9. **更新 Rollout 权重**
   ```python
   self.actor.update_rollout(self.rollout, step)
   ```

10. **测试评估**（可选）
    ```python
    if step % self.config.trainer.test_freq == 0:
        await self.rollout(False, step)
    ```

---

## 三、优势估计方法

### 3.1 REINFORCE

位置：`RL2/utils/algorithms.py` 的 `_compute_reinforce_adv`

```python
def _compute_reinforce_adv(
    tensor_dict,
    responses_per_prompt,
    global_norm,
    norm_var
):
    # 每个 prompt 的 reward 求和
    rewards = tensor_dict["rewards"].sum(-1).view(-1, responses_per_prompt)

    # 计算 baseline（均值）和标准差
    if responses_per_prompt == 1 or global_norm:
        baseline = rewards.mean()          # 全局 baseline
        std = rewards.std()
    else:
        baseline = rewards.mean(-1, keepdim=True)  # per-prompt baseline
        std = rewards.std(-1, keepdim=True)

    # 优势 = reward - baseline
    advantages = rewards - baseline
    if norm_var:
        advantages /= std

    advantages = advantages.view(-1, 1) * tensor_dict["action_mask"]
    return {"advantages": advantages}
```

**特点**：
- 简单高效，适合稀疏 reward
- 默认算法（`config.adv.estimator == "reinforce"`）

### 3.2 GAE (Generalized Advantage Estimation)

位置：`RL2/utils/algorithms.py` 的 `_compute_gae`

```python
def _compute_gae(
    tensor_dict, gamma, lamda
):
    # TD error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
    next_values = F.pad(tensor_dict["old_values"][:, 1:], (0, 1), value=0)
    deltas = tensor_dict["rewards"] + gamma * next_values - tensor_dict["old_values"]

    # GAE: A_t = δ_t + γ * λ * A_{t+1}
    gae, reversed_gaes = 0, []
    for t in reversed(range(deltas.shape[-1])):
        gae = deltas[:, t] + gamma * lamda * gae
        reversed_gaes.append(gae)
    gaes = torch.stack(reversed_gaes[::-1], -1)
    returns = gaes + tensor_dict["old_values"]

    # 归一化
    action_gaes = gaes[torch.where(tensor_dict["action_mask"])]
    advantages = (gaes - action_gaes.mean()) * tensor_dict["action_mask"] / (
        action_gaes.std() + torch.finfo(gaes.dtype).eps
    )

    return {"advantages": advantages, "returns": returns}
```

**特点**：
- 需要 Critic 模型估计 V(s)
- 通过 γ 和 λ 平衡偏差和方差
- 默认配置：`gamma=1.0, lamda=1.0`

---

## 四、PPO 损失函数

### 4.1 Actor PPO Loss

位置：`RL2/utils/algorithms.py` 的 `actor_ppo_loss`

```python
def actor_ppo_loss(config, minibatch):
    # ratio = π_new(a|s) / π_old(a|s)
    ratio = torch.exp(
        minibatch["logps"] - minibatch.get("old_logps", minibatch["logps"].detach())
    )
    
    # Clipped surrogate objective
    clipped_ratio = torch.clamp(ratio, 1 - config.clip, 1 + config.clip)
    objective = minibatch["advantages"] * ratio
    clipped_objective = minibatch["advantages"] * clipped_ratio
    losses = - torch.min(objective, clipped_objective)
    
    # 裁剪比例
    clip_ratios = objective > clipped_objective
```

**关键点**：
- `config.clip = 0.2`：裁剪范围 [0.8, 1.2]
- 最小化裁剪和未裁剪目标的负值
- clip_ratio 反映了多少比例被裁剪

### 4.2 KL 散度约束

```python
# KL 类型 1: reward
if config.actor.kl.type == "reward":
    tensor_dict["rewards"] -= config.actor.kl.coef * old_ref_approx_kl

# KL 类型 2: advantage
if config.actor.kl.type == "advantage":
    tensor_dict["advantages"] -= config.actor.kl.coef * old_ref_approx_kl
```

**KL Estimator**：
- `k1`：log_ratio（原始）
- `k2`：log_ratio² / 2
- `k3`：log_ratio + exp(-log_ratio) - 1

### 4.3 Critic PPO Loss

```python
def critic_ppo_loss(config, minibatch):
    # 裁剪 Value
    clipped_values = torch.clamp(
        minibatch["values"],
        minibatch["old_values"] - config.clip,
        minibatch["old_values"] + config.clip
    )
    
    # Value loss
    mse = (minibatch["values"] - minibatch["returns"]).pow(2)
    clipped_mse = (clipped_values - minibatch["returns"]).pow(2)
    losses = torch.max(mse, clipped_mse)
```

---

## 五、Train-Inference Mismatch 注入点分析

### 注入点 1：Rollout 生成参数

| 项目 | 说明 |
|------|------|
| **位置** | `ppo.py:61` → `rollout.py` → `rl.py:181-193` |
| **可扰动项** | `sampling_params`（temperature, top_p 等） |
| **影响** | 改变生成的 action 分布 |

### 注入点 2：Reward 信号扰动

| 项目 | 说明 |
|------|------|
| **位置** | `compute_advantages` 中的 `tensor_dict["rewards"]` |
| **可扰动项** | `rewards` 值 |
| **影响** | 直接影响 advantage 计算 |

```python
# 在 compute_advantages 中可以扰动 rewards
if config.actor.kl.type == "reward":
    tensor_dict["rewards"] -= config.actor.kl.coef * old_ref_approx_kl
# 在此之前注入扰动：
# tensor_dict["rewards"] = perturb_rewards(tensor_dict["rewards"])
```

### 注入点 3：Advantage 计算扰动

| 项目 | 说明 |
|------|------|
| **位置** | `compute_advantages` 函数 |
| **可扰动项** | `advantages` 值 |
| **影响** | 影响策略梯度更新方向和幅度 |

### 注入点 4：Actor/Critic 更新后权重同步

| 项目 | 说明 |
|------|------|
| **位置** | `ppo.py:83` → `actor.update_rollout` |
| **可扰动项** | 同步到推理服务器的权重 |
| **影响** | 推理时模型与训练模型不一致 |

---

## 六、关键配置项

位置：`RL2/trainer/config/ppo.yaml`

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `actor.clip` | PPO 裁剪范围 | 0.2 |
| `actor.kl.coef` | KL 散度系数 | 0.0 |
| `actor.kl.type` | KL 应用方式 | null |
| `actor.tis_coef` | Teacher-Student 系数 | 0.0 |
| `actor.entropy.coef` | 熵正则化系数 | 0.0 |
| `adv.estimator` | 优势估计器 | "reinforce" |
| `adv.gamma` | 折扣因子 | 1.0 |
| `adv.lamda` | GAE λ | 1.0 |
| `rollout.train.responses_per_prompt` | 每 prompt 的响应数 | 1 |

---

## 七、数据流与张量变换

### 7.1 Rollout 返回数据

```python
tensor_dict = {
    "states": torch.Tensor([...]),      # 输入 token IDs [batch_size, seq_len]
    "actions": torch.Tensor([...]),     # action token IDs
    "action_mask": torch.Tensor([...]), # 有效位置掩码
    "llm_logps": torch.Tensor([...]),  # LLM log probabilities
    "rewards": torch.Tensor([...])     # 环境 reward
}

cu_seqs = torch.Tensor([0, seq1_len, seq1_len+seq2_len, ...])
# 累计序列长度，用于 pack_tensor_dicts
```

### 7.2 优势计算后

```python
# compute_advantages 后新增
tensor_dict["advantages"] = torch.Tensor([...])  # 归一化优势
# 如果使用 GAE
tensor_dict["returns"] = torch.Tensor([...])      # 回报
```

---

## 八、分布式协作

### 8.1 进程组初始化

```python
initialize_global_process_group(True)
```

### 8.2 同步点

| 同步点 | 方式 | 说明 |
|--------|------|------|
| Rollout 完成 | `dist.barrier` | 确保所有 rank 完成生成 |
| 优势计算 | rank 0 only | 仅 rank 0 计算并广播 |
| 模型更新 | 隐式同步 | FSDP 自动同步 |

### 8.3 数据分发

```python
# Actor/Critic 内部
minibatches = self._scatter_data(tensor_dict)
# 按 device_mesh["dp"] 分发到各 rank
```

---

## 九、相关文件索引

| 文件 | 说明 |
|------|------|
| `RL2/trainer/ppo.py` | PPO 训练器主入口 |
| `RL2/trainer/base.py` | Trainer 基类 |
| `RL2/workers/fsdp/actor.py` | Actor Worker 实现 |
| `RL2/workers/fsdp/critic.py` | Critic Worker 实现 |
| `RL2/workers/rollout.py` | Rollout Worker 实现 |
| `RL2/utils/algorithms.py` | PPO/GAE/REINFORCE 算法实现 |
| `RL2/trainer/config/ppo.yaml` | PPO 默认配置 |

---

## 十、参考资料

- PPO 原始论文：[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- RL2 框架：[ChenmienTan/RL2](https://github.com/ChenmienTan/RL2)
- KL 散度近似：[Joschuan's Blog](http://joscha.net/blog/kl-approx.html)
