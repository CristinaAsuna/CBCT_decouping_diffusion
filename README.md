# palette_decoupling

一个面向医学 `.npy` 数据的简化版 Palette baseline。目标是保留 Palette 的核心条件扩散思路，同时去掉原工程里不利于快速拓展的实验框架。

## 核心思路

- 数据直接读取 `.npy`
- `condition` 与 `noisy target` 在 channel 维度拼接
- 支持一个 `condition` 对多个 target 通道
- 对你的任务，可以把 `full` 作为 condition，把 `left/right` 作为两个 target 通道一起生成

```text
input to denoiser = concat(condition, noisy_target)
condition = full
target = [left, right]
```

## 推荐数据组织

```text
train/
  full/
    0001.npy
  left/
    0001.npy
  right/
    0001.npy
```

每个 split 下按文件同名自动配对。如果后续改成“真实侧位片 -> left/right”，只需要把 `condition_dir` 指到真实侧位片投影后的 `.npy` 目录。

## 也支持你现在这种病例文件夹结构

例如：

```text
D:/nnunet/2D_test_v2/
  Bone_0001/
    std_full.npy
    std_left.npy
    std_right.npy
    aug_full.npy
    aug_left.npy
    aug_right.npy
    meta.json
  Bone_0002/
  ...
```

这时可以使用：

- `palette_decoupling/configs/cbct_casefolders_10cases.yaml`

这个数据集模式会把每个病例里的投影文件直接组成样本。你可以选择：

- 只用 `std_full -> std_left/std_right`
- 只用 `aug_full -> aug_left/aug_right`
- 同时把 `std` 和 `aug` 都作为样本加入训练

当前配置默认使用：

```yaml
variants: ["std", "aug"]
condition_template: "{variant}_full.npy"
target_templates:
  - "{variant}_left.npy"
  - "{variant}_right.npy"
```

相当于把每个病例扩成两条严格配对的样本：

- `Bone_0001__std`: `std_full -> [std_left, std_right]`
- `Bone_0001__aug`: `aug_full -> [aug_left, aug_right]`

## 训练

先修改配置文件 `palette_decoupling/configs/cbct_lr_concat.yaml` 或 `palette_decoupling/configs/cbct_casefolders_10cases.yaml`，然后运行：

```bash
python -m palette_decoupling.train --config palette_decoupling/configs/cbct_lr_concat.yaml
```

运行后如果是正常训练，你会看到：

- 启动时打印 `device/train_samples/val_samples`
- 每个 epoch 的 `train epoch x` 进度条
- 验证时的 `validate` 进度条
- 每个 epoch 结束后打印一行 `{'epoch': ..., 'train_loss': ..., 'val_loss': ...}`

如果最后显示 `KeyboardInterrupt by user`，那表示是手动中断，不是模型本身报错。

## 推理

```bash
python -m palette_decoupling.infer ^
  --config palette_decoupling/configs/cbct_lr_concat.yaml ^
  --checkpoint D:/your_output/checkpoints/best.pt ^
  --split test ^
  --output-dir D:/your_output/infer_test
```

## 评估

```bash
python -m palette_decoupling.evaluate ^
  --config palette_decoupling/configs/cbct_lr_concat.yaml ^
  --pred-dir D:/your_output/infer_test ^
  --split test ^
  --with-fid
```

目前支持 `MAE/MSE/PSNR/SSIM/FID`。FID 会按输出 channel 分别算一遍，再取平均。

## 后续适合继续加的东西

1. EMA
2. mixed precision
3. resume training
4. 专门面向真实侧位片的推理数据集
5. 更医学化的结构一致性指标
