# Notion 导入说明

这套模板面向你的 CBCT 解耦生成项目，目标是把实验管理拆成四层：

- `Idea`：我想试什么
- `Experiment`：我准备怎么公平验证
- `Run`：机器实际跑了什么
- `Decision`：最后学到了什么

建议的 Notion 顶层结构：

1. `CBCT 解耦生成项目主页`
2. `Ideas` 数据库
3. `Experiments` 数据库
4. `Runs` 数据库
5. `Decisions` 数据库

## 一、导入顺序

建议按下面顺序导入：

1. 导入 `ideas_db_template.csv`
2. 导入 `experiments_db_template.csv`
3. 导入 `runs_db_template.csv`
4. 导入 `decisions_db_template.csv`
5. 新建一个普通页面，复制 `01_PROJECT_HOME_TEMPLATE_ZH.md`
6. 为四个数据库分别建立页面模板，复制：
   - `02_IDEA_PAGE_TEMPLATE_ZH.md`
   - `03_EXPERIMENT_PAGE_TEMPLATE_ZH.md`
   - `04_RUN_PAGE_TEMPLATE_ZH.md`
   - `05_DECISION_PAGE_TEMPLATE_ZH.md`

## 二、如何导入 CSV

在 Notion 中：

1. 进入目标页面
2. 输入 `/csv`
3. 选择对应的 CSV 文件导入
4. 导入后，把每一列类型按下面建议调整

## 三、字段类型建议

### 1. Ideas

- `Idea ID`：Title
- `标题`：Text
- `模块`：Select
- `动机`：Text
- `假设`：Text
- `预期收益`：Select
- `实现成本`：Select
- `训练成本`：Select
- `是否可warm-start`：Checkbox
- `优先级`：Select
- `状态`：Select
- `关联实验`：Relation -> `Experiments`
- `备注`：Text

`模块` 选项建议：

- `Condition Injection`
- `UNet`
- `Attention`
- `Loss`
- `Sampler`
- `Data`
- `Eval`
- `Other`

`预期收益` / `实现成本` / `训练成本` 选项建议：

- `高`
- `中`
- `低`

`优先级` 选项建议：

- `P0`
- `P1`
- `P2`
- `P3`

`状态` 选项建议：

- `Inbox`
- `Ready`
- `Testing`
- `Promising`
- `Rejected`
- `Parked`

### 2. Experiments

- `Exp ID`：Title
- `实验标题`：Text
- `对应 Idea`：Relation -> `Ideas`
- `Baseline`：Text
- `唯一主改动`：Text
- `固定条件`：Text
- `数据预算`：Text
- `训练预算`：Text
- `起始 checkpoint`：Text
- `主指标`：Text
- `成功判据`：Text
- `停止判据`：Text
- `状态`：Select
- `优先级`：Select
- `关联 Runs`：Relation -> `Runs`
- `结论摘要`：Text

`状态` 选项建议：

- `Planned`
- `Running`
- `Analyzing`
- `Done`
- `Stopped`

### 3. Runs

- `Run ID`：Title
- `Run 名称`：Text
- `对应实验`：Relation -> `Experiments`
- `Git Commit`：Text
- `配置文件`：Text
- `Resume Checkpoint`：Text
- `机器/GPU`：Text
- `数据子集`：Text
- `训练步数`：Number
- `Seed`：Number
- `输出目录`：Text
- `Best val_mae`：Number
- `Best val_psnr`：Number
- `Best val_ssim`：Number
- `Physics MAE`：Number
- `Left diff loss`：Number
- `Right diff loss`：Number
- `是否有亮度异常`：Checkbox
- `视觉判断`：Text
- `状态`：Select
- `一句话结论`：Text

`状态` 选项建议：

- `Queued`
- `Running`
- `Finished`
- `Failed`
- `Aborted`

### 4. Decisions

- `Decision ID`：Title
- `主题`：Text
- `基于哪些实验`：Relation -> `Experiments`
- `结论`：Text
- `证据`：Text
- `下一步动作`：Text
- `置信度`：Select
- `是否已纳入 baseline`：Checkbox

`置信度` 选项建议：

- `高`
- `中`
- `低`

## 四、建议的数据库视图

### Ideas

- 看板视图：按 `状态` 分组
- 表格视图：按 `优先级` 排序

### Experiments

- 表格视图：全部实验
- 看板视图：按 `状态` 分组
- 过滤视图：`状态 != Done`

### Runs

- 表格视图：按 `Best val_mae` 升序
- 表格视图：按 `Best val_ssim` 降序
- 过滤视图：`是否有亮度异常 = true`

### Decisions

- 表格视图：全部结论
- 过滤视图：`是否已纳入 baseline = false`

## 五、推荐命名规范

- Idea：`I-001`
- Experiment：`E-001`
- Run：`R-001`
- Decision：`D-001`

实验标题建议格式：

- `[Condition] concat -> cond encoder | vs branch baseline`
- `[Loss] late consistency finetune | vs branch baseline`
- `[Attention] add decoder attention | vs branch baseline`

## 六、你现在最先可以填的内容

### Ideas

- `I-001` concat 改为 condition encoder
- `I-002` 在 encoder/decoder 增加 attention
- `I-003` 用 window attention 替换 MHA
- `I-004` late consistency finetune
- `I-005` 分析 left/right 亮度不一致

### Experiments

- `E-001` 纯 branch baseline 训练到收敛
- `E-002` 打印并比较 left_diff_loss / right_diff_loss
- `E-003` consistency 仅用于后期微调

## 七、推荐使用原则

1. 一个 `Idea` 可以对应多个 `Experiment`
2. 一个 `Experiment` 可以对应多个 `Run`
3. 只有经过 `Decision` 沉淀的结论，才算真正进展
4. 不要把训练日志原样贴进 `Experiment` 页面
5. `Run` 只写摘要和关键结果，不写所有 step
