# CBCT Decoupling Diffusion 使用说明（新版）


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

5. Physical consistency loss
   - 训练时使用+loss,推理时显示计算residaul=full-left_pre-right_pre
   - left_crr=left_pre+0.5*res
   - 当然,设定tolerate_mae=0.15,if mae>tolearate, do crr

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

## Infer,Val

Infer,去掉limit,则为全量

```powershell
python infer_research.py `
  --config configs/full_to_brach.yaml `
  --checkpoint D:\vscode_workplace\codeplace\palette\CBCT_decouping_diffusion\output\checkpoints\branch_consistent\best_ema.pt `
  --split val `
  --case-root D:\nnunet\2d_projection_physics_consistent `
  --output-dir D:\vscode_workplace\codeplace\palette\CBCT_decouping_diffusion\output\results\branch_consistent_corr_check `
  --weights ema `
  --limit 20 `
  --branch-correction equal `
  --branch-correction-strength 1


```

Val
```powershell
python evaluate_research.py `
  --config configs/full_to_brach.yaml `
  --pred-dir D:\vscode_workplace\codeplace\palette\CBCT_decouping_diffusion\output\results\branch_consistent_corr_check `
  --split val `
  --case-root D:\nnunet\2d_projection_physics_consistent

```

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
