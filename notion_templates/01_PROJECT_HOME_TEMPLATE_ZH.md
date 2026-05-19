# CBCT 解耦生成项目主页

## 当前 Baseline

- 模型：
- 条件注入：
- 输出形式：
- 当前最佳 checkpoint：
- 当前最佳指标：
- 当前主要问题：

## 本周重点

- [ ] 
- [ ] 
- [ ] 

## 快速入口

- Ideas 数据库
- Experiments 数据库
- Runs 数据库
- Decisions 数据库

## 固定观察样例

- Case 1：
- Case 2：
- Case 3：

建议这里固定 3 到 5 个病例，每次都看同一组样例，方便观察：

- 左右亮度是否一致
- 是否存在模糊
- 是否存在过暗/过亮
- `full - left - right` 的残差是否异常

## 当前结论摘要

- 
- 
- 

## 当前实验优先级

1. 先把纯 branch baseline 训练到稳定收敛
2. 检查 `left_diff_loss` 和 `right_diff_loss`
3. 确认亮度不一致是否来自 baseline 本身
4. 再决定是否做 consistency finetune
