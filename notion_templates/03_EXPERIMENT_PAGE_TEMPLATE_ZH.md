# {Exp ID} {实验标题}

## 1. 目标问题

这次实验想验证什么：

## 2. 假设

如果改动有效，我预计会看到：

## 3. 对照设计

- Baseline：
- 本次唯一主改动：
- 保持不变：
  - 数据划分
  - seed
  - sampler
  - eval 方式
  - 训练预算

## 4. 最小实验预算

- 数据量：
- 训练步数：
- GPU 预算：
- 是否从 checkpoint warm-start：

## 5. 记录指标

- `val_mae`
- `val_psnr`
- `val_ssim`
- `physics_mae`
- `left_diff_loss`
- `right_diff_loss`
- 是否存在亮度异常

## 6. 成功判据

- 
- 
- 

## 7. 停止判据

- 指标明显差于 baseline
- 视觉质量明显退化
- 出现亮度/结构异常
- 工程复杂度超过收益

## 8. 关联运行

- 

## 9. 结论

一句话结论：
