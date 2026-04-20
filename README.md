# CBCT Decoupling Diffusion 使用说明

这是一份新的项目说明文档，不会覆盖现有的 `README.md`。

当前项目已经支持以下几类任务：

- `full -> left` 单边生成
- `full -> [left, right]` 双通道联合生成
- `full -> side-conditioned single output`，通过 `side_emb` 控制输出 `left/right`

同时支持：

- DDPM / DDIM 推理
- EMA 权重
- `best_ema.pt`、`best_ema_mae.pt`、`best_ema_ssim.pt`
- `step` 学习率调度
- `cosine_with_warmup` 学习率调度
- 基于 `MAE + SSIM` 的 early stopping

## 1. 目录结构

推荐的数据目录形式如下：

```text
D:/nnunet/2D/
  Bone_0001/
    std_full.npy
    std_left.npy
    std_right.npy
    aug_full.npy
    aug_left.npy
    aug_right.npy
  Bone_0002/
  ...
```

其中：

- `*_full.npy` 是条件输入
- `*_left.npy` 是左侧目标
- `*_right.npy` 是右侧目标
- `std` / `aug` 是两个变体

## 2. 配置文件

项目中已经提供了三份可直接使用的配置：

- [configs/full_to_left_single.yaml](/D:/vscode_workplace/codeplace/palette/visual/CBCT_decouping_diffusion/configs/full_to_left_single.yaml)
- [configs/full_to_dual.yaml](/D:/vscode_workplace/codeplace/palette/visual/CBCT_decouping_diffusion/configs/full_to_dual.yaml)
- [configs/full_to_sidecond.yaml](/D:/vscode_workplace/codeplace/palette/visual/CBCT_decouping_diffusion/configs/full_to_sidecond.yaml)

它们分别对应：

- `full_to_left_single.yaml`
  训练 `full -> left`
- `full_to_dual.yaml`
  训练 `full -> [left, right]`
- `full_to_sidecond.yaml`
  训练 side-conditioned 模型，单模型可生成 `left` 或 `right`

如果在服务器上训练，请先把下面这些路径改成服务器路径：

- `dataset.train.case_root`
- `dataset.val.case_root`
- `output.root`

## 3. 训练

### 3.1 full -> left

```powershell
python -m CBCT_decouping_diffusion.train_research --config CBCT_decouping_diffusion/configs/full_to_left_single.yaml
```

### 3.2 full -> [left, right]

```powershell
python -m CBCT_decouping_diffusion.train_research --config CBCT_decouping_diffusion/configs/full_to_dual.yaml
```

### 3.3 full -> side-conditioned

```powershell
python -m CBCT_decouping_diffusion.train_research --config CBCT_decouping_diffusion/configs/full_to_sidecond.yaml
```

### 3.4 断点续训

```powershell
python -m CBCT_decouping_diffusion.train_research `
  --config CBCT_decouping_diffusion/configs/full_to_left_single.yaml `
  --resume /path/to/checkpoints/last.pt
```

## 4. 训练模式说明

### 4.1 single

配置关键项：

```yaml
target_mode: "single"
target_side: "left"
target_template: "{variant}_{side}.npy"
```

含义：

- 每个样本只取一个目标
- 如果 `target_side: left`，训练的是 `full -> left`
- 如果改成 `right`，训练的是 `full -> right`

### 4.2 dual

配置关键项：

```yaml
target_mode: "dual"
target_sides: ["left", "right"]
target_template: "{variant}_{side}.npy"
```

含义：

- 一个样本同时加载 `left` 和 `right`
- 输出固定为双通道
- `ch0 = left`
- `ch1 = right`

### 4.3 side_cond

配置关键项：

```yaml
target_mode: "side_cond"
target_sides: ["left", "right"]
side_labels:
  left: 0
  right: 1
```

含义：

- 训练时把 `left/right` 作为 side label
- 模型内部使用 `side_emb + time_emb`
- 仍然是单输出通道，但推理时可以通过 side 控制生成左侧或右侧

## 5. 学习率与 Early Stop

项目支持两类学习率调度：

- `step`
- `cosine_with_warmup`

例如：

```yaml
lr_schedule:
  enabled: true
  type: "cosine_with_warmup"
  warmup_steps: 10000
  warmup_start_lr: 0.0
  min_lr: 0.000001
```

当前 early stopping 使用综合分数：

```text
score = ssim_weight * val_ssim - mae_weight * val_mae
```

例如：

```yaml
early_stop:
  enabled: true
  patience_validations: 20
  min_delta: 0.0002
  score:
    mae_weight: 1.0
    ssim_weight: 1.0
```

## 6. 保存的权重

训练过程中会保存：

- `checkpoints/best_ema.pt`
  按 `train.best_metric` 保存的主 best
- `checkpoints/best_ema_mae.pt`
  `val_mae` 最优
- `checkpoints/best_ema_ssim.pt`
  `val_ssim` 最优
- `checkpoints/last.pt`
  最近一次保存的 checkpoint
- `checkpoints/step_xxxxxx.pt`
  周期保存

## 7. 推理

### 7.1 单边模型推理

```powershell
python infer_research.py `
  --config configs/full_to_left_single.yaml `
  --checkpoint output/checkpoints/best_ema_mae.pt `
  --split val `
  --case-root D:/nnunet/2D `
  --case-name Bone_0001 `
  --variant std `
  --output-dir output/infer/Bone_0001_std
```

### 7.2 双通道模型推理

```powershell
python infer_research.py `
  --config configs/full_to_dual.yaml `
  --checkpoint output/checkpoints/best_ema_ssim.pt `
  --split val `
  --case-root D:/nnunet/2D `
  --case-name Bone_0001 `
  --variant std `
  --output-dir output/infer_dual/Bone_0001_std
```

输出结果中：

- `*_ch0.png` 对应 left
- `*_ch1.png` 对应 right

### 7.3 side-conditioned 模型推理

生成 `left`：

```powershell
python infer_research.py `
  --config configs/full_to_sidecond.yaml `
  --checkpoint output/checkpoints/best_ema_ssim.pt `
  --split val `
  --case-root D:/nnunet/2D `
  --case-name Bone_0001 `
  --variant std `
  --side left `
  --output-dir output/infer_side_left/Bone_0001_std
```

生成 `right`：

```powershell
python infer_research.py `
  --config configs/full_to_sidecond.yaml `
  --checkpoint output/checkpoints/best_ema_ssim.pt `
  --split val `
  --case-root D:/nnunet/2D `
  --case-name Bone_0001 `
  --variant std `
  --side right `
  --output-dir output/infer_side_right/Bone_0001_std
```

## 8. DDIM 采样过程可视化

如果想导出采样过程：

```powershell
python infer_research.py `
  --config configs/full_to_left_single.yaml `
  --checkpoint output/checkpoints/best_ema_ssim.pt `
  --split val `
  --case-root D:/nnunet/2D `
  --case-name Bone_0001 `
  --variant aug `
  --output-dir output/trace/Bone_0001_aug `
  --save-trace `
  --trace-channel 0
```

会生成：

- `*_sampling.gif`
  DDIM 当前采样状态演化
- `*_predx0.gif`
  每一步预测的 `x0_hat`
- `*_sampling_grid.png`
- `*_predx0_grid.png`
- `*_trace/`
  每一步单独 PNG

## 9. 评估

评估流程分两步：

### 9.1 先批量推理

```powershell
python infer_research.py `
  --config configs/full_to_left_single.yaml `
  --checkpoint output/checkpoints/best_ema_mae.pt `
  --split val `
  --case-root D:/nnunet/2D `
  --output-dir output/eval_preds/val_left
```

### 9.2 再计算指标

```powershell
python evaluate_research.py `
  --config configs/full_to_left_single.yaml `
  --split val `
  --case-root D:/nnunet/2D `
  --pred-dir output/eval_preds/val_left
```

如需 FID：

```powershell
python evaluate_research.py `
  --config configs/full_to_left_single.yaml `
  --split val `
  --case-root D:/nnunet/2D `
  --pred-dir output/eval_preds/val_left `
  --with-fid
```

当前支持：

- `MAE`
- `MSE`
- `PSNR`
- `SSIM`
- `FID`

## 10. 评估时的注意事项

### 10.1 不要复用旧的 pred 目录

每换一个 checkpoint，建议换一个新的 `pred-dir`，避免旧预测和新预测混在一起。

### 10.2 dual 模型的评估

`full_to_dual.yaml` 下，评估会把预测的双通道结果和 `[left, right]` 真值一起比较。

### 10.3 side_cond 模型的评估

`full_to_sidecond.yaml` 下，dataset 会展开成：

- `Bone_xxxx__std__left`
- `Bone_xxxx__std__right`

所以推理和评估时，预测文件名也必须和这个样本命名一致。

## 11. 运行方式

如果按包方式运行，建议在项目上一级目录执行：

```powershell
python -m CBCT_decouping_diffusion.train_research --config CBCT_decouping_diffusion/configs/full_to_left_single.yaml
```

如果直接在当前目录运行，也已经兼容：

```powershell
python train_research.py --config configs/full_to_left_single.yaml
```

推理和评估脚本也支持同样两种方式。

## 12. 建议的使用顺序

如果你想先做稳妥实验，建议顺序是：

1. 先跑 `full_to_left_single.yaml`
2. 再跑 `full_to_dual.yaml`
3. 最后试 `full_to_sidecond.yaml`

原因是：

- `single` 最稳定，最容易分析
- `dual` 能直接看联合输出是否值得
- `side_cond` 更灵活，但训练稳定性和可解释性更依赖实验结果
