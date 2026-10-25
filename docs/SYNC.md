# 实验结果导出与模型归档

## 结果目录

`results/`保存指标、实验协议、来源哈希及固定视角图片；输入数据、完整PLY和checkpoint单独存储。运行标签必须唯一，每次实验保留独立记录。

baseline导出示例：

```bash
python scripts/export_results.py --config configs/train.json --name train-baseline-repro
```

压缩实验使用对应阶段的 `export` 子命令，完整示例见[README](../README.md)。导出器检查模型、指标及来源记录，并生成用于对照与报告的小型结果集。

## 重复计时

```bash
python scripts/publish_timing.py --source results/timing-raw-repro \
  --destination results/timing-repeat-repro
```

发布工具将机器特定元数据规范化，保留指标与逐帧耗时。`public_redaction.json`记录原始文件与发布文件的SHA256映射，汇总文件使用发布后的哈希。原始目录用于任务续跑，发布目录用于分析与分享。

新计时记录包含模型及配置哈希。旧baseline计时仅能验证数量、分辨率、图片尺寸和统计一致性，无法证明模型未被同数量的新参数替换；导出会注明此验证边界。已有实验的代码哈希约束保持不变，继续旧实验应使用其记录的代码版本，新代码用于独立的新实验目录。

导出先写暂存目录，再发布完整结果目录；失败后可以使用同一标签重试。重复计时遇到截断JSON会隔离损坏文件，只补测未成功保存的样本；完整但数值或来源不符的记录仍会报错，不会自动覆盖。

## 模型归档

```bash
python scripts/archive_run.py --config configs/train.json
```

生成 `archives/train_baseline_30k.tar.gz` 和对应的 `.sha256` 文件。归档不含输入数据；数据来源、下载哈希和划分清单应与模型一同保留。归档可通过文件传输或对象存储共享，下载后用 `sha256sum` 或PowerShell的 `Get-FileHash -Algorithm SHA256` 核对完整性。

归档完成可读性检查及SHA256计算后才发布最终文件名。传输时同时保留归档和校验文件；暂存文件不是完成的归档。

复现实验需要代码版本、官方子模块版本、环境配置、数据划分、最终模型和评测记录。仅下载Git中的小型结果可以重建报告图表，不能继续训练或重新渲染完整场景。
