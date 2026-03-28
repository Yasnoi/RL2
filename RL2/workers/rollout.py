"""
Rollout Worker for RL2 Framework

研读笔记 by AI Assistant - 2026-03-28
============================================

## 整体架构理解

Rollout是RL训练中的核心环节，负责：
1. 从数据加载器获取prompt
2. 让Actor模型生成response（通过SGLang推理引擎）
3. 与环境交互获取reward
4. 收集trajectory数据供训练使用

## 关键类和数据结构

### Rollout类 (主入口)
- 初始化阶段：
  * `_prepare_device_mesh()`: 设置分布式设备网格 (dp x tp)
  * `_prepare_environment_variables()`: 设置CUDA_VISIBLE_DEVICES
  * 启动SGLang推理服务器和Router
  * 加载环境和数据加载器

- 推理阶段 (`__call__`):
  * rank 0 负责任务调度：创建asyncio任务调度LLM生成
  * 收集sample_group结果，打包成tensor_dict返回
  * 其他rank等待完成后同步

- 模型更新阶段 (`update`):
  * 将训练后的模型权重更新到推理服务器
  * 通过HTTP请求传输序列化后的权重

### SampleGroup (RL2/datasets/rl.py)
- 一个SampleGroup对应一个prompt的多个response（通过config.responses_per_prompt控制）
- 内部包含多个Sample对象
- generate(): 调度组内所有Sample的生成任务

### Sample (RL2/datasets/rl.py)
- state_text: 当前的输入状态（prompt + 已生成的action）
- action_text: 生成的action（response）
- state_dict: 包含 states, actions, action_mask, logps, rewards
- status: RUNNING / ABORTED / DONE

## 核心生成循环 (base_generate in rl.py)

```python
while True:
    # 1. LLM生成
    response = await async_request(router_url, "generate", ...)
    add_llm_response(sample, response)  # 更新state_dict中的logps和actions
    
    # 2. 环境交互
    response = await env_step_fn(sample)  # 调用env.step(sample)
    add_env_response(tokenizer, sample, response)  # 更新reward和next_state
    
    # 3. 判断是否结束
    if sample.status == Sample.Status.DONE:
        return
```

## Train-Inference Mismatch 可能的注入点

根据代码分析，可以在以下几个位置引入扰动：

### 注入点1: LLM生成参数扰动 (base_generate)
- 位置：rl.py:181-193 的sampling_params
- 可扰动项：temperature, max_new_tokens, top_p, top_k
- 影响：改变生成的action，从而影响后续环境交互

### 注入点2: 环境step扰动 (env_step)
- 位置：envs/ 下的各个环境文件
- 可扰动项：reward值、next_state
- 影响：直接影响策略梯度计算

### 注入点3: Response后处理扰动 (add_llm_response)
- 位置：rl.py:67-103
- 可扰动项：action_text, logps
- 影响：影响训练数据的质量

### 注入点4: Rollout层整体扰动 (rollout.__call__)
- 位置：rollout.py:178-290
- 可扰动项：sample_group的结果
- 影响：全局性的数据扰动

## 配置相关

关键配置项 (ppo.yaml):
- rollout.train.sampling_params.temperature: 生成温度
- rollout.train.sampling_params.max_new_tokens: 最大生成长度
- rollout.train.responses_per_prompt: 每个prompt生成多少个response
- rollout.env_path: 环境定义文件路径
"""

from typing import Optional, Union, Dict, List, Tuple, Generator, Sequence
from omegaconf import OmegaConf, DictConfig
import os
import asyncio
import importlib
import functools
import multiprocessing
from collections import defaultdict
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate
from transformers import AutoTokenizer
from sglang.srt.server_args import ServerArgs
from sglang.srt.entrypoints.http_server_engine import launch_server_process
from sglang.srt.utils import  MultiprocessingSerializer
from sglang_router.launch_router import RouterArgs, launch_router
from RL2.datasets import (
    get_dataloaders,
    pack_tensor_dicts,
    RLDataset,
    SampleGroup
)
from RL2.utils.communication import (
    get_host,
    get_available_port,
    get_gloo_group,
    broadcast_object,
    gather_and_concat_list,
    sync_request,
    async_request
)
from RL2.utils.logging import (
    progress_bar,
    time_logger,
    gather_and_log
)

try:
    from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
except ImportError:
    from sglang.srt.patch_torch import monkey_patch_torch_reductions

try:
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket
except ImportError:
    from sglang.srt.model_executor.model_runner import FlattenedTensorBucket

PROCESSES = []

def shutdown_processes_when_exit(func):
    """
    装饰器：确保在退出时关闭所有启动的子进程（SGLang服务器和Router）
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        finally:
            for process in PROCESSES:
                process.terminate()
                process.join(timeout=3)
                if process.is_alive():
                    process.kill()
    return wrapper


class Rollout:
    """
    Rollout Worker - 负责任务编排和环境交互的核心类
    
    工作流程：
    1. rank 0 初始化阶段：
       - 加载tokenizer和数据加载器
       - 启动Router进程（用于请求分发）
       - 为每个tp_local_rank启动SGLang推理服务器
       
    2. rank 0 推理阶段 (__call__)：
       - 从dataloader获取SampleGroup
       - 通过Router调度LLM生成和环境交互
       - 收集结果并打包成tensor_dict返回
    
    3. 模型权重更新阶段 (update)：
       - 将训练后的actor权重同步到SGLang服务器
       - 通过HTTP接口传输序列化后的权重
    """

    def __init__(self, config: DictConfig):
        """
        初始化Rollout worker
        
        Args:
            config: OmegaConf配置对象，包含：
                - server_args: SGLang服务器参数
                - train/test: 数据集和rollout配置
                - env_path: 环境定义文件路径
        """
        self.config = config
        
        # 步骤1: 初始化设备网格
        # world_size = dp_size * tp_size
        # dp用于数据并行，tp用于张量并行
        self._prepare_device_mesh()
        
        # 步骤2: 设置环境变量（CUDA_VISIBLE_DEVICES）
        self._prepare_environment_variables()

        # 步骤3: rank 0 专属初始化
        if dist.get_rank() == 0:
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.server_args.model_path, trust_remote_code=True
            )
            
            # 创建数据加载器 (train和test)
            # RLDataset: 每个item是一个SampleGroup
            self.train_dataloader, self.test_dataloader = get_dataloaders(
                RLDataset, config, self.tokenizer, 1
            )

            # 加载环境定义（从env_path加载用户自定义的env_step函数）
            self._prepare_environment()
            
            # 用于partial rollout的buffer
            self.sample_buffer: List[SampleGroup] = []

            # 启动Router进程（用于请求负载均衡）
            self._launch_router_process()

        # 同步屏障，确保rank 0完成初始化后再继续
        dist.barrier(group=get_gloo_group())
        
        # 步骤4: 每个tp_local_rank启动一个SGLang推理服务器
        # tp_size个服务器共同处理一个模型的推理
        if self.device_mesh["tp"].get_local_rank() == 0:
            self._launch_server_process()

    def _prepare_device_mesh(self):
        """
        初始化分布式设备网格
        
        设备网格是一个2D网格：(dp_size, tp_size)
        - dp_size: 数据并行度（有多少份模型副本）
        - tp_size: 张量并行度（模型被切分到多少GPU）
        
        例如：8 GPU, tp_size=2 → dp_size=4, tp_size=2
        """
        world_size = dist.get_world_size()
        tp_size = self.config.server_args.tp_size
        assert world_size % tp_size == 0, \
            f"World_size {world_size} must be divisible by tp_size {tp_size}."

        self.device_mesh = dist.device_mesh.init_device_mesh(
            "cpu",
            mesh_dim_names=("dp", "tp"),
            mesh_shape=(world_size // tp_size, tp_size)
        )

    def _prepare_environment_variables(self):
        """
        设置CUDA_VISIBLE_DEVICES环境变量
        
        为每个tp rank分配不同的GPU：
        - tp_size=2, local_rank=0 → GPU 0
        - tp_size=2, local_rank=1 → GPU 1
        
        通过all_gather_object确保所有rank都知道分配结果
        """
        if "TORCHELASTIC_USE_AGENT_STORE" in os.environ.keys():
            del os.environ["TORCHELASTIC_USE_AGENT_STORE"]
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible_devices:
            cuda_visible_devices = cuda_visible_devices.split(",")
            cuda_visible_device = cuda_visible_devices[int(os.environ["LOCAL_RANK"])]
        else:
            cuda_visible_device = os.environ["LOCAL_RANK"]
        cuda_visible_devices = self.device_mesh["tp"].size() * [None]
        dist.all_gather_object(
            cuda_visible_devices,
            cuda_visible_device,
            self.device_mesh["tp"].get_group(),
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(cuda_visible_devices)
        monkey_patch_torch_reductions()

    def _launch_server_process(self):
        """
        启动SGLang推理服务器进程
        
        每个tp_local_rank启动一个服务器实例
        服务器负责：
        - 加载模型权重
        - 处理生成请求
        - 管理KV cache
        """
        # TODO: support cross-node server
        server_args = OmegaConf.to_container(self.config.server_args)
        server_args = ServerArgs(
            enable_memory_saver=True,
            host=get_host(),
            port=get_available_port(),
            log_level="error",
            **server_args
        )
        server_process = launch_server_process(server_args)
        PROCESSES.append(server_process)
        
        # 获取服务器URL供后续请求使用
        self.worker_url = server_args.url()

        # 将worker URL注册到Router，供负载均衡使用
        router_url = broadcast_object(
            self.router_url if dist.get_rank() == 0 else None,
            process_group=self.device_mesh["dp"].get_group(),
            group_src=0
        )
        sync_request(router_url, f"add_worker?url={self.worker_url}")
        
        # 收集所有dp_rank的worker URLs
        self.worker_urls = gather_and_concat_list(
            [self.worker_url],
            self.device_mesh["dp"].get_group()
        )

    def _launch_router_process(self):
        """
        启动SGLang Router进程
        
        Router负责：
        - 接收请求
        - 负载均衡分发到多个worker
        - 健康检查
        """
        router_args = RouterArgs(
            host=get_host(),
            port=get_available_port(),
            log_level="error"
        )
        self.router_url = f"http://{router_args.host}:{router_args.port}"

        router_process = multiprocessing.Process(
            target=launch_router, args=(router_args,)
        )
        router_process.start()
        PROCESSES.append(router_process)
        sync_request(self.router_url, "health", "GET", 10)

    def _prepare_environment(self):
        """
        加载环境定义模块
        
        从env_path加载用户定义的环境文件
        环境文件应提供generate函数（包含env_step）
        参考：envs/orz.py, envs/countdown.py
        """
        spec = importlib.util.spec_from_file_location(
            "custom_module", self.config.env_path
        )
        self.env = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.env)

    @time_logger("rollout")
    async def __call__(
        self,
        train: bool,
        step: int
    ) -> Optional[Tuple[Optional[Dict[str, torch.Tensor]], Optional[torch.Tensor]]]:
        """
        执行Rollout的主函数
        
        流程：
        1. rank 0 负责任务调度
        2. 创建asyncio任务并发生成
        3. 收集结果并打包
        4. 其他rank同步等待
        
        Args:
            train: 是否为训练模式（决定使用哪个dataloader）
            step: 当前训练步数（用于日志保存）
            
        Returns:
            tensor_dict: 包含states, actions, action_mask, llm_logps, rewards
            cu_seqs: cumulative sequence lengths，用于序列打包
        """

        def _schedule_tasks(
            sample_groups: Sequence[SampleGroup]
        ):
            """
            为每个SampleGroup创建异步生成任务
            
            每个SampleGroup内部会并发生成多个response
            所有任务被添加到pendings集合
            """
            for sample_group in sample_groups:
                pendings.add(
                    asyncio.create_task(
                        sample_group.generate(
                            self.router_url, self.env.generate
                        )
                    )
                )

        # ====== rank 0: 任务调度 ======
        if dist.get_rank() == 0:

            config = self.config.train if train else self.config.test
            dataloader = self.train_dataloader if train else self.test_dataloader
            
            # 确定本轮rollout的prompt数量
            groups_to_complete = config.prompts_per_rollout or len(dataloader)
            
            tbar = progress_bar(
                total=groups_to_complete, desc="Rollout"
            )

            pendings, first_iter = set(), True
            filtered_groups, completed_groups = 0, 0
            all_tensor_dicts: List[List[Dict[str, torch.Tensor]]] = []
            metrics: Dict[str, List[Union[float, int, bool]]] = defaultdict(list)

            # Partial rollout: 使用之前未完成的sample_buffer
            if train and config.partial_rollout:
                _schedule_tasks(self.sample_buffer)

            # ====== 核心循环：调度和收集 ======
            while completed_groups < groups_to_complete:

                # 第一次迭代或partial rollout时，从dataloader获取新样本
                if first_iter or (train and config.partial_rollout):
                    sample_groups = dataloader(
                        groups_to_complete - len(pendings)
                    )
                    _schedule_tasks(sample_groups)

                # 等待任一任务完成
                done, pendings = await asyncio.wait(
                    pendings, return_when=asyncio.FIRST_COMPLETED
                )

                # 处理完成的任务
                for task in done:
                    if completed_groups < groups_to_complete:
                        tbar.update()
                    completed_groups += 1
                    sample_group = task.result()
                    
                    # 首次迭代打印示例
                    if first_iter:
                        sample_group.print()
                        first_iter = False
                    
                    # 保存到磁盘
                    await asyncio.to_thread(sample_group.save, step)
                    
                    # 转换为tensor格式
                    all_tensor_dicts_delta, metrics_delta = (
                        sample_group.to_all_tensor_dicts_and_metrics()
                    )
                    for k, v in metrics_delta.items():
                        metrics[k].extend(v)
                    
                    # Dynamic filtering: 如果所有reward相同则跳过
                    if (
                        train and config.dynamic_filtering and
                        len(metrics_delta["rewards"]) > 1 and
                        torch.tensor(metrics_delta["rewards"]).std() == 0
                    ):
                        filtered_groups += 1
                        continue
                    all_tensor_dicts.extend(all_tensor_dicts_delta)
                    
            # ====== Partial rollout后处理 ======
            if train and config.partial_rollout:
                await async_request(self.worker_urls, "pause_generation")
                done, _ = await asyncio.wait(pendings)
                self.sample_buffer = [task.result() for task in done]
                await async_request(self.worker_urls, "continue_generation")

            # 记录metrics
            metrics["dynamic_filtering_ratio"].append(
                filtered_groups / completed_groups
            )
            suffix = "train" if train else "test"
            metrics = {f"{k}/{suffix}": v for k, v in metrics.items()}
            gather_and_log(metrics, step)

        # ====== 同步屏障 ======
        # 使用GLOO group避免影响SGLang服务器
        await asyncio.to_thread(dist.barrier, group=get_gloo_group())

        if not train:
            return

        # ====== rank 0: 释放KV cache内存 ======
        if self.device_mesh["tp"].get_local_rank() == 0:
            await async_request(
                self.worker_url,
                "release_memory_occupation",
                json={"tags": ["weights", "kv_cache"]}
            )

        if dist.get_rank() != 0:
            return None, None

        # ====== 打包返回值 ======
        tensor_dicts: List[Dict[str, torch.Tensor]] = [
            td for tds in all_tensor_dicts for td in tds
        ]
        tensor_dict: Dict[str, torch.Tensor] = pack_tensor_dicts(tensor_dicts)
        
        # 计算每个序列的长度，用于序列打包
        seqs = torch.LongTensor([
            len(tensor_dicts) for tensor_dicts in all_tensor_dicts
        ])
        cu_seqs = torch.cumsum(
            torch.cat((torch.LongTensor([0]), seqs)), dim=0
        )
        
        return tensor_dict, cu_seqs
    
    @torch.no_grad()
    def update(
        self, named_tensor_generator: Generator[Tuple[str, torch.Tensor], None, None]
    ):
        """
        将训练后的模型权重更新到推理服务器
        
        流程：
        1. 释放KV cache占用
        2. 按dtype和bucket分批传输权重
        3. 恢复KV cache占用
        
        Args:
            named_tensor_generator: (name, tensor)元组的生成器
        """

        torch.cuda.empty_cache()
        dist.barrier(group=get_gloo_group())
        # 必须先释放KV cache，否则可能OOM
        
        if self.device_mesh["tp"].get_local_rank() == 0:
            sync_request(
                self.worker_url,
                "resume_memory_occupation",
                json={"tags": ["weights"]}
            )

        def _update_tensor_bucket(
            dtype_to_named_tensors: Dict[torch.dtype, List[Tuple[str, torch.Tensor]]]
        ):
            """
            传输一批权重到推理服务器
            
            按dtype分组，序列化为字符串后传输
            服务器收到后会直接复制到GPU
            """
            torch.cuda.synchronize()
            serialized_tensors = []
            for named_tensors in dtype_to_named_tensors.values():

                flattened_tensor_bucket = FlattenedTensorBucket(named_tensors)
                flattened_tensor_data = {
                    "flattened_tensor": flattened_tensor_bucket.get_flattened_tensor(),
                    "metadata": flattened_tensor_bucket.get_metadata()
                }
                serialized_tensors.append(
                    MultiprocessingSerializer.serialize(
                        flattened_tensor_data, output_str=True
                    )
                )

            # 在tp group内gather所有序列化的权重
            gathered_serialized_tensors = [
                None for _ in range(self.device_mesh["tp"].size())
            ] if self.device_mesh["tp"].get_local_rank() == 0 else None
            dist.gather_object(
                serialized_tensors,
                gathered_serialized_tensors,
                group_dst=0,
                group=self.device_mesh["tp"].get_group(),
            )
            
            # rank 0 负责发送到推理服务器
            if self.device_mesh["tp"].get_local_rank() == 0:
                for serialized_named_tensors in zip(*gathered_serialized_tensors):
                    sync_request(
                        self.worker_url,
                        "update_weights_from_tensor",
                        json={
                            "serialized_named_tensors": serialized_named_tensors,
                            "load_format": "flattened_bucket",
                            "flush_cache": False
                        }
                    )
        
        # 按bucket分批传输权重
        dtype_to_named_tensors = defaultdict(list)
        bucket_size = 0
        for name, tensor in named_tensor_generator:
            param_size = tensor.numel() * tensor.element_size()

            # 达到bucket大小限制时先传输这一批
            if bucket_size > 0 and bucket_size + param_size > (self.config.bucket_size << 20):

                _update_tensor_bucket(dtype_to_named_tensors)
                dtype_to_named_tensors = defaultdict(list)
                bucket_size = 0

            # 移动到目标GPU并转换为local tensor
            tensor = tensor.to(
                torch.cuda.current_device(), non_blocking=True
            ).detach()
            if isinstance(tensor, DTensor):
                # DTensor需要redistribute到Replicate布局
                tensor = tensor.redistribute(
                    placements=[Replicate()] * tensor.device_mesh.ndim,
                    async_op=True
                ).to_local()
            
            dtype_to_named_tensors[tensor.dtype].append((name, tensor))
            bucket_size += param_size

        # 传输最后一批
        _update_tensor_bucket(dtype_to_named_tensors)

        # 恢复KV cache占用
        if self.device_mesh["tp"].get_local_rank() == 0:
            sync_request(
                self.worker_url,
                "resume_memory_occupation",
                json={"tags": ["kv_cache"]}
            )
        dist.barrier(group=get_gloo_group())
