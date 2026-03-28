# Code Conventions for RL2 Modification

## 项目结构
```
RL2/               # 核心框架
├── datasets/      # 数据集处理模块
├── trainer/       # 训练器实现 (SFT/DPO/RM/PPO)
├── workers/       # Worker进程 (actor/critic/rollout)
│   └── rollout.py # ← 主要改动位置
├── utils/         # 工具函数
envs/              # 环境定义
examples/          # 示例脚本
```

## 代码风格
- 使用类型注解 (typing)
- async/await 用于异步操作
- PyTorch 风格：tensor操作
- 配置使用YAML，通过OmegaConf加载

## Rollout相关关键文件
- `RL2/workers/rollout.py`: Rollout worker实现
- `RL2/workers/fsdp/actor.py`: Actor模型前向传播
- `RL2/trainer/ppo.py`: PPO训练循环

## 配置加载顺序
1. 默认配置 (YAML文件)
2. 命令行覆盖 (--key value)
3. 使用 `OmegaConf.merge()` 合并

## 测试建议
- 使用小数据集验证代码正确性
- 优先在CPU或单GPU环境测试
- 检查点保存便于复现
