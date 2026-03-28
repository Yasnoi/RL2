# Research Background - Train-Inference Mismatch in RL

## 研究主题
强化学习中的训练-推理不一致问题（Train-Inference Mismatch）

## 核心问题
在LLM强化学习中，训练阶段和推理阶段可能存在多种不一致：
1. **模型精度不一致**：训练用FP32/BF16，推理用INT8/INT4量化
2. **环境交互差异**：训练时模拟环境 vs 真实推理环境
3. **rollout策略差异**：训练时on-policy vs 推理时的exploration策略
4. **计算资源差异**：训练用高性能GPU集群 vs 推理用有限资源

## 研究框架
- 框架：RL2 (Ray Less Reinforcement Learning)
- 模型：Qwen3系列
- 算法：PPO with on-policy (单步)
- 目标：验证Rollout阶段的扰动对策略优化的影响

## 关键文献
- RL2: Ray Less Reinforcement Learning (ChenmienTan et al.)
- Dr. GRPO: 默认使用的优势估计方法
- DeepSeek GRPO: 对比使用的变体

## 实验设计
- Baseline: 标准PPO rollout
- Treatment: Rollout阶段引入可控扰动
- 对比指标: reward, score, policy loss, kl divergence
