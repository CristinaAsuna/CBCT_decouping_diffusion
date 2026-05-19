# Tiny Ablation

这个目录是低成本验证 idea 的“沙盒”。它不替代 full training，主要回答三个问题：

- 代码是否能跑通，loss 是否正常下降。
- tiny set 是否能快速过拟合，左右分支是否明显失衡。
- 新模块是否值得进入 small-budget 或 full-budget 实验。

## 目录

- `branch_configs/`：只研究 branch 分支的 6 个 tiny 实验。
- `modules/`：tiny 专用实验模块，例如 CNN condition encoder 和 window attention。
- `train_branch_tiny.py`：branch-only tiny 训练入口，不修改主目录训练逻辑。
- `scripts/make_tiny_cases.py`：从 case root 抽取少量 case。
- `scripts/run_branch_tiny_ablation.py`：在一块 GPU 上顺序运行 branch tiny 实验。
- `outputs/`：tiny 实验输出，不建议提交到 Git。
- `generated_branch_configs/`：运行脚本生成的临时配置，不建议提交到 Git。

## Branch-Only 实验

当前新增的 6 个 branch 实验是：

- `tiny_branch_01_baseline.yaml`：branch decoder baseline。
- `tiny_branch_02_consistency_finetune.yaml`：从 baseline checkpoint 继续，用 consistency loss 微调。
- `tiny_branch_03_consistency_scratch.yaml`：从头训练 branch + consistency loss。
- `tiny_branch_04_cnn_plus_concat.yaml`：在 raw full concat 的基础上，再 concat CNN condition feature。
- `tiny_branch_05_cnn_only_no_raw_concat.yaml`：只使用 CNN condition feature，不直接 concat raw full。
- `tiny_branch_06_window_attention.yaml`：用 tiny window attention 替换原始全局 attention。

注意：`02/04/05/06` 默认从 `01 baseline` 的 `last.pt` warm-start。`04/05/06` 使用 `strict_resume: false`，会加载 shape 匹配的旧参数，跳过新增模块、输入卷积或被替换的 attention 参数。

## Windows 本地运行

在仓库根目录运行：

```powershell
.\tiny_ablation\run_branch_tiny.ps1 `
  -CaseRoot D:/nnunet/2D `
  -NumCases 16
```

只查看将要执行的命令：

```powershell
.\tiny_ablation\run_branch_tiny.ps1 `
  -CaseRoot D:/nnunet/2D `
  -NumCases 16 `
  -DryRun
```

如果只是快速 smoke test，可以覆盖步数。对 warm-start 实验，覆盖值会被解释为“从 baseline checkpoint 之后继续跑多少步”：

```powershell
.\tiny_ablation\run_branch_tiny.ps1 `
  -CaseRoot D:/nnunet/2D `
  -NumCases 4 `
  -MaxSteps 20
```

## Linux/集群运行

在仓库根目录运行：

```bash
CASE_ROOT=/public_bme2/bme-cuizhm/maxquan/Datasets/CBCT/2D \
NUM_CASES=16 \
bash tiny_ablation/run_branch_tiny.sh
```

只跑某一个配置：

```bash
python tiny_ablation/scripts/run_branch_tiny_ablation.py \
  --case-root /public_bme2/bme-cuizhm/maxquan/Datasets/CBCT/2D \
  --configs tiny_ablation/branch_configs/tiny_branch_01_baseline.yaml
```

## TensorBoard

每个实验会写到：

```text
tiny_ablation/outputs/<experiment_name>/tensorboard
```

查看所有 branch tiny 实验：

```bash
tensorboard --logdir tiny_ablation/outputs --port 6006
```

## 如何判断 idea 是否值得放大

优先看这些信号：

- `train/diff_loss`、`train/left_diff_loss`、`train/right_diff_loss` 是否稳定下降。
- `val/val_left_diff_loss` 和 `val/val_right_diff_loss` 是否明显失衡。
- `val/val_mae`、`val/val_psnr`、`val/val_ssim` 是否至少不比 baseline 更差。
- TensorBoard images 里 `pred_ch0` / `pred_ch1` 是否出现亮度漂移、全灰、全黑、左右互换。
- consistency 实验的 `consistency_ratio` 是否保持很小；如果 tiny 都明显伤害亮度，就不要贸然 full run。

Tiny 通过不代表 full run 一定成功，但 tiny 不通过通常说明这个方向还不值得消耗完整训练预算。
