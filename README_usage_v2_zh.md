# CBCT Decoupling Diffusion 使用说明（新版）

这份文档不覆盖原始 `README.md`，专门面向当前代码版本的实际使用流程。

重点说明：

- 如何训练 `full -> left`
- 如何训练 `full -> right`
- 如何训练 `full -> [left, right]`
- 如何训练 side-conditioned 模型
- 如何训练 branch decoder 模型
- 如何做推理
- 如何做 `val` 评估
- 如何安全断点续传
- 断点续传后学习率、优化器状态、EMA 是否连续

---

## 1. 项目当前支持的主要模型

当前项目主要支持以下几类任务：

1. 单边生成
   - `full -> left`
   - `full -> right`
2. 双输出生成
   - `full -> [left, right]`
3. side embedding 注入
   - 单模型，根据 `side label` 生成 `left` 或 `right`
4. branch decoder
   - 共享前半解码器，左右两支分支分别输出

对应配置文件如下：

- `configs/full_to_left_single.yaml`
  - 单边生成，输出 `left`
- `configs/full_to_right_single.yaml`
  - 单边生成，输出 `right`
- `configs/full_to_dual.yaml`
  - 直接双通道输出 `[left, right]`
- `configs/full_to_sidecond.yaml`
  - side-conditioned 模型
- `configs/full_to_brach.yaml`
  - branch decoder 模型
  - 注意：文件名当前是 `brach`，不是 `branch`

---

## 2. 推荐的数据组织方式

推荐的 case-folder 结构如下：

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
- `std` / `aug` 是不同变体

---

## 3. 环境依赖

`requirements.txt` 中列出的核心依赖包括：

- `torch`
- `torchvision`
- `numpy`
- `Pillow`
- `PyYAML`
- `tqdm`
- `swanlab`
- `clean-fid`

如果你只做训练、推理、基础评估，最关键的是先保证：

- `torch`
- `numpy`
- `Pillow`
- `PyYAML`

如果需要 FID，再额外确保：

- `clean-fid`

---

## 4. 启动方式

当前代码支持两种启动方式：

1. 包方式启动

```powershell
python -m CBCT_decouping_diffusion.train_research --config CBCT_decouping_diffusion/configs/full_to_left_single.yaml
```

2. 直接脚本启动

```powershell
python train_research.py --config configs/full_to_left_single.yaml
```

推理和评估脚本同样支持这两种方式。

如果你在项目根目录 `CBCT_decouping_diffusion` 下工作，通常直接脚本方式最方便。

---

## 5. 训练

### 5.1 `full -> left`

```powershell
python train_research.py --config configs/full_to_left_single.yaml
```

### 5.2 `full -> right`

```powershell
python train_research.py --config configs/full_to_right_single.yaml
```

### 5.3 `full -> [left, right]`

```powershell
python train_research.py --config configs/full_to_dual.yaml
```

### 5.4 side-conditioned

```powershell
python train_research.py --config configs/full_to_sidecond.yaml
```

### 5.5 branch decoder

```powershell
python train_research.py --config configs/full_to_brach.yaml
```

---

## 6. DDP 训练

当前训练脚本已经支持 DDP，推荐通过 `torchrun` 启动。

### 6.1 单机双卡示例

```powershell
torchrun --standalone --nproc_per_node=2 train_research.py --config configs/full_to_left_single.yaml
```

### 6.2 单机四卡示例

```powershell
torchrun --standalone --nproc_per_node=4 train_research.py --config configs/full_to_dual.yaml
```

### 6.3 说明

- `torchrun` 会自动注入 `WORLD_SIZE`、`RANK`、`LOCAL_RANK`
- 当前代码内部会根据这些环境变量自动初始化 DDP
- 主进程负责打印日志、验证、保存 checkpoint
- 非主进程不重复做这些工作

建议：

- 单卡训练可直接 `python train_research.py ...`
- 多卡训练尽量统一使用 `torchrun`

---

## 7. 训练输出目录

训练输出根目录由配置文件中的：

```yaml
output:
  root: "/path/to/output"
```

控制。

通常会生成以下内容：

### 7.1 checkpoints

位于：

```text
output_root/checkpoints/
```

包含：

- `best_ema.pt`
  - 按 `train.best_metric` 保存的主 best
- `best_ema_mae.pt`
  - `val_mae` 最优
- `best_ema_ssim.pt`
  - `val_ssim` 最优
- `last.pt`
  - 最近一次保存的训练状态
- `step_xxxxxx.pt`
  - 周期性保存的训练状态

### 7.2 samples

位于：

```text
output_root/samples/
```

通常会保存验证预览图与预览 `.npy`。

---

## 8. 推理

推理脚本：

```powershell
python infer_research.py --config ... --checkpoint ... --output-dir ...
```

推荐推理时优先使用：

- `best_ema.pt`
- `best_ema_mae.pt`
- `best_ema_ssim.pt`

通常不建议推理时优先用 `last.pt`，除非你就是想看最近一步的效果。

### 8.1 单边模型推理

```powershell
python infer_research.py `
  --config configs/full_to_left_single.yaml `
  --checkpoint output/checkpoints/best_ema.pt `
  --split val `
  --case-root D:/nnunet/2D `
  --case-name Bone_0001 `
  --variant std `
  --output-dir output/infer_left/Bone_0001_std
```

### 8.2 `full -> right` 推理

```powershell
python infer_research.py `
  --config configs/full_to_right_single.yaml `
  --checkpoint output/checkpoints/best_ema.pt `
  --split val `
  --case-root D:/nnunet/2D `
  --case-name Bone_0001 `
  --variant std `
  --output-dir output/infer_right/Bone_0001_std
```

### 8.3 双通道模型推理

```powershell
python infer_research.py `
  --config configs/full_to_dual.yaml `
  --checkpoint output/checkpoints/best_ema.pt `
  --split val `
  --case-root D:/nnunet/2D `
  --case-name Bone_0001 `
  --variant std `
  --output-dir output/infer_dual/Bone_0001_std
```

输出中：

- `*_ch0.png` 对应 `left`
- `*_ch1.png` 对应 `right`

### 8.4 side-conditioned 推理

生成 `left`：

```powershell
python infer_research.py `
  --config configs/full_to_sidecond.yaml `
  --checkpoint output/checkpoints/best_ema.pt `
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
  --checkpoint output/checkpoints/best_ema.pt `
  --split val `
  --case-root D:/nnunet/2D `
  --case-name Bone_0001 `
  --variant std `
  --side right `
  --output-dir output/infer_side_right/Bone_0001_std
```

说明：

- side-conditioned 数据集内部每个样本是按 `left/right` 分开组织的
- 当你只想推单个 case 时，建议显式传 `--side left` 或 `--side right`

### 8.5 branch decoder 推理

```powershell
python infer_research.py `
  --config configs/full_to_brach.yaml `
  --checkpoint output/checkpoints/best_ema.pt `
  --split val `
  --case-root D:/nnunet/2D `
  --case-name Bone_0001 `
  --variant std `
  --output-dir output/infer_branch/Bone_0001_std
```

branch decoder 的输出同样是双通道：

- `ch0 = left`
- `ch1 = right`

### 8.6 批量推理整个 split

如果不传 `--case-name`，脚本会遍历整个 split：

```powershell
python infer_research.py `
  --config configs/full_to_dual.yaml `
  --checkpoint output/checkpoints/best_ema.pt `
  --split val `
  --output-dir output/infer_dual_val
```

### 8.7 保存 DDIM 中间轨迹

```powershell
python infer_research.py `
  --config configs/full_to_left_single.yaml `
  --checkpoint output/checkpoints/best_ema.pt `
  --split val `
  --case-root D:/nnunet/2D `
  --case-name Bone_0001 `
  --variant std `
  --save-trace `
  --trace-channel 0 `
  --output-dir output/infer_trace/Bone_0001_std
```

---

## 9. `val` 评估

评估脚本：

```powershell
python evaluate_research.py --config ... --pred-dir ...
```

### 9.1 单边模型评估

```powershell
python evaluate_research.py `
  --config configs/full_to_left_single.yaml `
  --pred-dir output/infer_left_val `
  --split val `
  --case-root D:/nnunet/2D
```

### 9.2 双通道模型评估

```powershell
python evaluate_research.py `
  --config configs/full_to_dual.yaml `
  --pred-dir output/infer_dual_val `
  --split val `
  --case-root D:/nnunet/2D
```

### 9.3 side-conditioned 模型评估

如果你保存的是 side-conditioned 在 `val` split 上的整批输出，也可以直接评估：

```powershell
python evaluate_research.py `
  --config configs/full_to_sidecond.yaml `
  --pred-dir output/infer_side_val `
  --split val `
  --case-root D:/nnunet/2D
```

### 9.4 计算 FID

```powershell
python evaluate_research.py `
  --config configs/full_to_dual.yaml `
  --pred-dir output/infer_dual_val `
  --split val `
  --case-root D:/nnunet/2D `
  --with-fid
```

---

## 10. 如何断点续传

### 10.1 最常用的恢复命令

如果你训练到一半中断，最推荐从：

- `last.pt`
或
- `step_xxxxxx.pt`

继续训练。

示例：

```powershell
python train_research.py `
  --config configs/full_to_left_single.yaml `
  --resume output/checkpoints/last.pt
```

也可以用 DDP 方式恢复：

```powershell
torchrun --standalone --nproc_per_node=2 train_research.py `
  --config configs/full_to_left_single.yaml `
  --resume output/checkpoints/last.pt
```

### 10.2 如果我在 200K 时发现 bug，修复后想继续训练怎么办

假设你已经训练到 `200000` step，后来发现了一个 bug，并且已经修复了代码。

推荐做法：

1. 保留原来的 `output/checkpoints/last.pt`
2. 修复代码
3. 确认模型结构和配置没有发生不兼容变化
4. 使用同一个配置文件继续训练
5. 通过 `--resume output/checkpoints/last.pt` 恢复

示例：

```powershell
python train_research.py `
  --config configs/full_to_left_single.yaml `
  --resume output/checkpoints/last.pt
```

如果你当时正好保存了某个整步 checkpoint，例如：

- `step_200000.pt`

也可以直接从它恢复：

```powershell
python train_research.py `
  --config configs/full_to_left_single.yaml `
  --resume output/checkpoints/step_200000.pt
```

---

## 11. 断点续传时，哪些状态会被恢复

当前训练代码会从 checkpoint 中恢复以下内容：

- 模型参数 `model`
- EMA 参数 `ema_model`
- 优化器状态 `optimizer`
- 当前训练步数 `step`
- `best_metric`
- `best_metrics`
- early stopping 相关状态
- `train_epoch`
- `batches_seen_in_epoch`
- 最近若干步 loss 统计

这意味着：

- 学习率不会从头重新开始
- AdamW 的动量统计不会清空
- EMA 不会重新从头累积
- `best_ema.pt` / `best_ema_mae.pt` / `best_ema_ssim.pt` 的比较逻辑会延续之前状态

---

## 12. 断点续传后学习率会不会变

### 12.1 结论

正常情况下，不会“重置回初始学习率”，而是按恢复后的全局 step 继续走。

### 12.2 原因

训练循环里每一步都会根据当前 `step` 重新计算学习率。

也就是说，恢复到 `200000` step 后：

- 如果你用的是 `step` 调度
  - 会按 `200000` 之后的规则继续
- 如果你用的是 `cosine_with_warmup`
  - 会按 `200000 / max_steps` 所对应的位置继续

不会重新回到 warmup 起点。

### 12.3 举例

如果你的配置是：

```yaml
train:
  max_steps: 400000
  lr: 0.00005
  lr_schedule:
    enabled: true
    type: "cosine_with_warmup"
    warmup_steps: 10000
    warmup_start_lr: 0.0
    min_lr: 0.000001
```

那么训练到 `200000` step 中断后再恢复：

- 学习率会继续使用接近 `200000` 位置的 cosine 值
- 不会重新从 `0.0 -> 0.00005` 再 warmup 一遍

---

## 13. 会不会出现重新训练后大幅振荡

### 13.1 一般情况下

如果满足下面条件，通常不会因为“正常恢复”而出现明显大幅振荡：

- 恢复的是 `last.pt` 或合适的 `step_xxxxxx.pt`
- 配置文件没有乱改
- 模型结构没有改动
- batch size / world size 没有大改
- 优化器状态成功恢复

因为当前代码恢复了：

- 模型参数
- 优化器状态
- EMA 状态
- 全局 step

这和“重新从头训练再把权重手工载入”是不同的，连续性会好很多。

### 13.2 可能出现波动的情况

下面几种情况更容易导致恢复后损失曲线不平滑，甚至明显抖动：

1. 你修复的 bug 改变了前向传播或 loss 定义
2. 你改了模型结构
3. 你改了输入归一化方式
4. 你改了 batch size
5. 你从单卡切到多卡，或 world size 变化明显
6. 你改了数据集划分或采样顺序
7. 你改了学习率配置

这些情况不是“断点续传本身的问题”，而是“训练系统本身发生了变化”。

### 13.3 当前代码还有一个需要知道的点

当前代码虽然保存了 `rng_state`，但恢复时并没有真正启用它，恢复位置如下：

```python
# restore_rng_state(training_state.get("rng_state"))
```

这意味着：

- 恢复后不会做到严格 bitwise 一致
- 后续数据顺序、扩散噪声采样、随机性细节可能和原始未中断轨迹略有不同

但是通常这不会单独导致“大幅振荡”，更常见的是：

- 曲线不会与中断前完全像素级重合
- 但整体训练趋势应当是连续的

---

## 14. 哪些 checkpoint 用于什么场景

### 14.1 继续训练

优先使用：

- `last.pt`
- `step_xxxxxx.pt`

理由：

- 它们最适合表示“某个确定训练时刻的完整状态”
- 尤其是 `last.pt` 最适合“中断后继续”

### 14.2 推理

优先使用：

- `best_ema.pt`
- `best_ema_mae.pt`
- `best_ema_ssim.pt`

理由：

- 这些通常是验证结果最好的权重

### 14.3 能不能用 `best_ema.pt` 继续训练

技术上，如果这个文件包含完整训练状态，也可以继续。

但通常不建议作为“中断续训的首选”，因为：

- `best_ema.pt` 对应的是“最佳验证结果时刻”
- 它不一定是“最近训练时刻”

例如：

- `best_ema.pt` 保存于 `180K`
- `last.pt` 保存于 `200K`

如果你想从 bug 修复后尽量接近中断点继续，就应优先用 `last.pt` 或 `step_200000.pt`。

---

## 15. 恢复训练前的检查清单

恢复前建议确认以下几点：

1. `--config` 与原训练时保持一致
2. 模型结构配置不要改
   - 例如 `inner_channel`
   - `channel_mults`
   - `attn_res`
   - `res_blocks`
   - `branch_decoder`
3. 数据目标模式保持一致
   - `single`
   - `dual`
   - `side_cond`
4. 如果是 side-conditioned，`side_labels` 不要改
5. 如果是 branch decoder，确保 `target_channels=2` 的逻辑没变
6. 优先从 `last.pt` 或 `step_xxxxxx.pt` 恢复

如果你修改了模型结构，当前代码会更严格地检查权重是否匹配，不匹配时应重新训练，而不是强行续训。

---

## 16. 推荐实验顺序

如果你准备重新从头做完整实验，建议按这个顺序推进：

1. `full_to_left_single.yaml`
2. `full_to_right_single.yaml`
3. `full_to_dual.yaml`
4. `full_to_sidecond.yaml`
5. `full_to_brach.yaml`

这样做的好处是：

- 先验证单边任务是否稳定
- 再看双通道是否优于单边
- 再比较 side embedding 和 branch decoder 的收益

---

## 17. 常见建议

### 17.1 单边任务

适合先验证基础可训练性，最容易定位问题。

### 17.2 双通道直接输出

适合验证联合生成是否有效。

### 17.3 side-conditioned

优点：

- 一个模型可以生成左右两边

注意：

- 推理单个 case 时要注意 `--side`

### 17.4 branch decoder

适合你想保留共享表征，但让左右输出分支各自学习的情况。

---

## 18. 一套典型工作流示例

### 18.1 训练

```powershell
python train_research.py --config configs/full_to_left_single.yaml
```

### 18.2 中断后恢复

```powershell
python train_research.py `
  --config configs/full_to_left_single.yaml `
  --resume output/checkpoints/last.pt
```

### 18.3 推理 `val`

```powershell
python infer_research.py `
  --config configs/full_to_left_single.yaml `
  --checkpoint output/checkpoints/best_ema.pt `
  --split val `
  --output-dir output/infer_left_val
```

### 18.4 评估 `val`

```powershell
python evaluate_research.py `
  --config configs/full_to_left_single.yaml `
  --pred-dir output/infer_left_val `
  --split val
```

---

## 19. 总结

对于你现在这套代码，最推荐的使用原则是：

1. 继续训练用 `last.pt` 或 `step_xxxxxx.pt`
2. 推理优先用 `best_ema.pt`
3. 恢复训练时学习率不会从头开始
4. 正常恢复通常不会单独导致大幅振荡
5. 真正容易引入振荡的是模型、损失、归一化、batch size、world size、数据流程的变化

如果后续你还准备继续扩展实验，我建议把每一种方法单独再整理成一页“推荐配置 + 推荐命令 + 推荐 checkpoint 选择”的实验手册，这样后面复现实验会轻松很多。
