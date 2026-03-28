# Rollout Modification Prompt Template

## 目标
在RL2框架的Rollout阶段引入可控扰动，模拟训练-推理不一致（train-inference mismatch）。

## 核心研究问题
- 在单步on-policy强化学习中，Rollout阶段的扰动如何影响策略优化？
- 如何设计扰动使得既能模拟真实场景的inference mismatch，又保持实验可控性？

## Rollout机制理解
参考 `RL2/workers/rollout.py`：
- Rollout负责与环境交互生成trajectories
- 返回 state, action, reward, next_state, done 等信息
- Actor模型在推理时可能与训练时存在差异（如量化、精度变化等）

## 改动方向
1. 在 `RL2/workers/rollout.py` 的 `rollout_step` 或类似函数中引入扰动
2. 可选方案：
   - 动作空间的噪声注入
   - 状态表示的扰动
   - Reward信号的轻微偏差
   - Temperature/top-k/top-p的调整模拟

## 代码修改检查清单
- [ ] 理解现有Rollout代码结构
- [ ] 确定注入扰动的具体位置
- [ ] 实现扰动开关（可通过config控制）
- [ ] 添加相应的配置项
- [ ] 确保扰动不影响训练流程的核心逻辑
- [ ] 编写单元测试或验证脚本

## 配置示例
```yaml
actor:
  rollout:
    mismatch_enabled: true
    mismatch_type: "action_noise"  # action_noise, state_perturbation, reward_bias
    mismatch_strength: 0.1
```
