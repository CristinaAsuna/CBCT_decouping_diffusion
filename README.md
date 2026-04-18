# palette_decoupling

一个面向医学 `.npy` 数据的简化版 Palette 条件扩散基线工程。目标是保留 Palette 的核心思路，同时把原始项目里不利于快速改医学任务的实验框架解耦出来，方便直接做 `full -> left/right` 这类投影生成任务。

## 核心思路

- 数据直接读取 `.npy`
- 条件扩散的输入是 `concat(condition, noisy_target)`
- 支持一个 `condition` 对一个或多个 target 通道
- 支持病例文件夹结构，自动保证同一个病例内 `std` 和 `aug` 各自严格配对

例如：

```text
input to denoiser = concat(condition, noisy_target)
condition = full
target = left
```

或者：

```text
input to denoiser = concat(condition, noisy_target)
condition = full
target = [left, right]
```

## 支持的数据组织

### 1. 平铺目录

```text
train/
  full/
    0001.npy
  left/
    0001.npy
  right/
    0001.npy
```

按同名自动配对。

### 2. 病例文件夹

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

对于病例文件夹模式，当前工程支持：

- `std_full -> std_left`
- `aug_full -> aug_left`
- `std_full -> [std_left, std_right]`
- `aug_full -> [aug_left, aug_right]`

不会把 `std_full` 错配到 `aug_left/right`。

## 目前推荐的研究版 baseline

当前更推荐使用研究版入口：

- `train_research.py`
- `infer_research.py`
- `evaluate_research.py`

它们支持：

- 按病例级 `train/val/test` 划分
- 固定强度范围归一化，例如 `[0, 6] -> [-1, 1]`
- `step` 驱动训练，而不是按 epoch 决策
- EMA 权重
- DDIM 风格快速采样
- checkpoint 保存与 resume
- 可选 FID

## 推荐配置

对于目前的 `1175 case, full -> left` baseline，推荐配置：

- `palette_decoupling/configs/cbct_1175_full_to_left_step_baseline.yaml`

这份配置默认：

- 使用病例级 `90% train / 10% val`
- 每个病例同时包含 `std` 和 `aug`
- 输入分辨率 `256 x 256`
- 强度范围按 `[0, 6]` 裁剪并映射到 `[-1, 1]`
- 使用 EMA
- 每 `1000` step 验证一次
- 默认用 DDIM `50 steps` 做验证采样
- 默认关闭训练中的 FID，避免离线集群下载 Inception 权重时报错

## 训练

### 常规训练

```bash
python -m palette_decoupling.train_research --config palette_decoupling/configs/cbct_1175_full_to_left_step_baseline.yaml
```

### 断点续训

```bash
python -m palette_decoupling.train_research \
  --config palette_decoupling/configs/cbct_1175_full_to_left_step_baseline.yaml \
  --resume /path/to/checkpoints/last.pt
```

当前 checkpoint 会保存：

- `model`
- `ema_model`
- `optimizer`
- `step`
- `best_metric`

恢复训练时会自动接着之前的 step 往下跑。

## 推理

研究版推理默认优先使用训练配置里的 DDIM 采样参数。

```bash
python -m palette_decoupling.infer_research \
  --config palette_decoupling/configs/cbct_1175_full_to_left_step_baseline.yaml \
  --checkpoint /path/to/checkpoints/best_ema.pt \
  --split val \
  --output-dir /path/to/infer_val_ema
```

可选参数：

- `--weights ema`：优先加载 `ema_model`
- `--weights model`：加载普通模型权重

## 评估

```bash
python -m palette_decoupling.evaluate_research \
  --config palette_decoupling/configs/cbct_1175_full_to_left_step_baseline.yaml \
  --pred-dir /path/to/infer_val_ema \
  --split val \
  --with-fid
```

当前支持：

- `MAE`
- `MSE`
- `PSNR`
- `SSIM`
- `FID`

注意：在离线集群环境里，`FID` 依赖 `cleanfid` 下载 Inception 权重。如果节点无法联网，建议训练时先关闭周期性 FID，等训练结束后再在有权重或可联网环境里单独评估。

## 输出目录说明

训练过程中常见输出：

- `checkpoints/best_ema.pt`
  当前最佳 EMA 权重
- `checkpoints/last.pt`
  最近一次保存的 checkpoint
- `checkpoints/step_xxxxxx.pt`
  周期性保存的断点
- `samples/step_xxxxxx_<sample>_pred.npy`
  该 step 下某个验证样本的原始预测数组
- `samples/step_xxxxxx_<sample>_ch0.png`
  该预测结果第 0 个通道的可视化图

如果当前是 `full -> left` 单目标任务，那么：

- `.npy` 通常形状接近 `1 x 256 x 256`
- `ch0.png` 就是这个唯一输出通道的可视化

## 日志与 SwanLab

### 训练进度条和 SwanLab step 为什么不完全一样

终端里的 `train steps` 是每次参数更新都加 1。

SwanLab 默认不是每一步都记录，而是按配置里的：

- `log_every_steps`
- `validate_every_steps`

来抽样记录训练和验证指标。

因此：

- 终端的 step 是真实全局 step
- SwanLab 曲线是稀疏采样后的可视化

最稳妥的做法是直接看 SwanLab 里记录点的 `step` 字段，而不是靠“第几个点”去换算。

### SwanLab 在集群 local 模式下的注意事项

如果集群节点无法联网，建议：

```bash
export SWANLAB_MODE="local"
```

如果需要本地 dashboard 依赖，可手动安装：

```bash
uv pip install "swanlab[dashboard]" --prerelease=allow
```

### SwanLab 与 resume

当前代码支持从环境变量读取：

- `SWANLAB_RUN_ID`
- `SWANLAB_RESUME`
- `SWANLAB_MODE`
- `SWANLAB_LOGDIR`

例如：

```bash
export SWANLAB_MODE="local"
export SWANLAB_LOGDIR="/path/to/swanlog"
export SWANLAB_RUN_ID="ib2mwwfgear8ctqvdujux"
export SWANLAB_RESUME="allow"
```

但是要注意：

- 训练 checkpoint 的 resume 是可靠的
- SwanLab 在 `local` 模式下不一定能像 `cloud` 模式那样真正续写到同一个本地 run 目录
- 在 `local` 模式下，更推荐把 SwanLab 当作本地分段日志，再通过 `swanlab sync` 做后续同步

也就是说：

- 模型断点续训：靠 `--resume last.pt`
- SwanLab 本地记录：可能会新建一个 `run-*` 目录

## 集群运行建议

如果以包方式运行，请在包目录的上一级执行：

```bash
python -m CBCT_decouping_diffusion.train_research --config CBCT_decouping_diffusion/configs/xxx.yaml
```

不要直接：

```bash
python train_research.py
```

因为当前代码里使用了相对导入，直接脚本方式运行会报：

```text
ImportError: attempted relative import with no known parent package
```

## 当前最适合继续扩展的方向

1. `full -> right` 镜像 baseline
2. 双目标 `full -> [left, right]`
3. 自动 early stopping
4. 更医学任务相关的结构一致性指标
5. 更严格的测试集独立评估与可视化报告
