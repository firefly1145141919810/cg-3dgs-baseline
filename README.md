# 3DGS 质量保持压缩

计算机图形学课程项目：在官方 3D Gaussian Splatting baseline 上，研究**训练视角可见贡献评分、固定数量剪枝和短程恢复训练**，比较画质、模型大小与渲染开销。

已完成 train / truck 两场景、70% / 50% / 30% 保留率实验与独立重复计时。推荐展示工作点为保留50%高斯：相对同预算完整恢复模型，PSNR分别下降0.0526 / 0.0085 dB，PLY约减半，渲染加速约1.46 / 1.41倍。贡献度方法不在所有指标上占优，完整对照及限制见实验文档。

## 阅读入口

- [项目演示：结果展示与现场运行](docs/DEMO.md)
- [实验目的、设计、配置、结果与结论](docs/COMPRESSION_EXPERIMENT.md)
- [最终报告PDF](output/pdf/3dgs_final_report.pdf) · [LaTeX源码与图表重建](report/README.md)
- [零基础原理讲解](docs/BASELINE_GUIDE.md) · [论文阅读索引](references/READING_NOTES.md) · [中文批注论文](output/pdf/3dgs_annotated_zh.pdf)
- [来源与AI辅助说明](NOTICE.md) · [验证范围](docs/STATUS.md)

## 运行环境

实测配置：Linux（glibc 2.28）、RTX 4090 D 24GB、驱动550.120、Python 3.10.22、PyTorch 2.2.2+cu121、CUDA Toolkit 12.1、GCC 8.5。这是结果的测量环境，不是所有使用者必须具备的同款硬件。

训练和渲染需要受工具链支持的NVIDIA CUDA GPU、`nvcc`和C++编译器。驱动、Toolkit与PyTorch需匹配；`nvidia-smi`显示的CUDA版本不能代替已安装Toolkit版本。安装脚本提供cu121 / cu118入口，不保证适用于所有新架构。

官方版本由 `baseline.lock.json` 固定，位于 `vendor/gaussian-splatting` 子模块；不启用新版曝光补偿、深度监督或抗锯齿功能。

## 1. 获取代码与安装

除特别说明外，以下命令均从仓库根目录执行。

```bash
git clone https://github.com/firefly1145141919810/cg-3dgs-baseline.git
cd cg-3dgs-baseline
python scripts/bootstrap.py
conda create -n cg3dgs python=3.10 -y
conda activate cg3dgs
# 已选择CUDA Toolkit 12.1时使用cu121；11.8对应cu118
bash scripts/setup_cloud.sh cu121
python scripts/doctor.py
```

安装需要联网；离线计算节点应提前准备依赖、数据与评测权重。详细安装与排错见[环境指南](docs/CLOUD.md)。

## 2. 数据与baseline

阅读并接受数据条款后下载作者预处理的数据；下载器支持进度显示与断点续传。

```bash
python scripts/download_data.py --accept-dataset-terms
export TORCH_HOME="$PWD/data/torch-cache"
python scripts/run.py all --config configs/smoke.json
python scripts/run.py all --config configs/train.json
python scripts/benchmark.py --config configs/train.json
python scripts/export_results.py --config configs/train.json --name train-baseline-repro
```

`all`依次训练、渲染、评测；冒烟1k步，正式baseline 30k步。`configs/train.json`和`configs/truck.json`指定数据/输出路径、分辨率、迭代数和背景。`train`在这里是火车场景名，不等于训练集。

数据默认位于 `data/tandt/{train,truck}`，模型位于 `outputs/`。只克隆Git中的小型结果不能继续剪枝或训练，还需要数据与完整模型。

## 3. train：50%剪枝与恢复

从已完成train baseline出发执行以下主流程。第一阶段显式指定 `train_phase1_v2`，与后续阶段默认读取的目录保持一致。每次新实验使用独立输出目录。

```bash
# 保存重载验证与不透明度剪枝
python scripts/compression_phase1.py prepare --experiment outputs/compression/train_phase1_v2
python scripts/compression_phase1.py evaluate --experiment outputs/compression/train_phase1_v2
# 全训练视角贡献评分及无恢复评测
python scripts/contribution_phase3.py score --views 0
python scripts/contribution_phase3.py prepare
python scripts/contribution_phase3.py evaluate
# 完整/不透明度/贡献度，各自不恢复或恢复3000步，共六组
python scripts/recovery_phase4.py prepare --steps 3000 --seed 0
python scripts/recovery_phase4.py train
python scripts/recovery_phase4.py evaluate
python scripts/recovery_phase4.py export --name train-scoring-recovery3k-repro
```

独立阶段结果可用 `compression_phase1.py export --experiment outputs/compression/train_phase1_v2 --name train-opacity-repro` 和 `contribution_phase3.py export --name train-contribution-repro` 导出。

如需复现早期“不透明度＋恢复”四组实验，在第一阶段完成后运行下列命令；最终六组流程不依赖这一步。

```bash
python scripts/recovery_phase2.py prepare --steps 3000 --seed 0
python scripts/recovery_phase2.py train
python scripts/recovery_phase2.py evaluate
python scripts/recovery_phase2.py export --name train-recovery3k-repro
```

## 4. truck与多预算实验

truck入口串行完成baseline、两种剪枝与50%六组恢复对照；已完成且校验通过的阶段会跳过。

```bash
python scripts/truck_suite.py plan
python scripts/truck_suite.py all
python scripts/truck_suite.py export --name truck-scoring-recovery3k-repro
```

两个场景贡献评分均完成后，运行70%和30%预算。每个预算从原始完整模型选点，不在50%模型上再剪一次。

```bash
(
set -e
for scene in train truck; do
  for keep in 0.7 0.3; do
    case "$keep" in 0.7) budget=70 ;; 0.3) budget=30 ;; esac
    python scripts/budget_phase5.py run --scene "$scene" --keep "$keep"
    python scripts/budget_phase5.py export --scene "$scene" --keep "$keep" \
      --name "${scene}-budget${budget}-repro"
  done
done
)
```

可先运行20步冒烟，指定独立目录，不纳入正式结果：

```bash
python scripts/budget_phase5.py run --scene train --keep 0.3 --steps 20 \
  --experiment outputs/compression/train_budget30_smoke20
```

## 5. 独立计时

六个正式质量实验全部完成后，对18个恢复后模型设置各启动3个独立进程计时：

```bash
python scripts/repeat_timing.py --name timing-raw-repro --rounds 3
python scripts/publish_timing.py --source results/timing-raw-repro \
  --destination results/timing-repeat-repro
```

该入口读取 `outputs/compression/{scene}_scoring_recovery_3k`（50%）及 `{scene}_budget{70|30}_recovery3k`，不改写历史计时。若改变这些模型路径，须相应调整来源收集逻辑，不能只改导出名称。

`repeat_timing.py`的原始目录用于断点续跑，`publish_timing.py`生成保留数值与来源哈希的可分享结果。计时期间保持GPU独占，避免其他任务影响测量。

恢复阶段支持已有断点；baseline续跑需指定checkpoint。各入口的完整参数见 `--help`。

## 结果、检查与报告

`results/`保存指标、协议和固定视角图片；数据、完整模型、日志与缓存不进入Git。结果导出与模型归档方法见[结果管理](docs/SYNC.md)。

```bash
python -m unittest discover -s tests -v
python scripts/static_check.py
```

测试覆盖范围见[验证说明](docs/STATUS.md)。图表由 `scripts/build_report_assets.py`从固定已发布结果生成；独立报告依赖与XeLaTeX编译见[报告说明](report/README.md)。使用新的实验结果时，需同步更新图表脚本的数据来源。

| 目录 | 内容 |
| --- | --- |
| `vendor/gaussian-splatting` | 官方算法和CUDA子模块 |
| `configs/`、`environment/` | 实验配置、依赖 |
| `scripts/`、`tests/` | 训练/压缩/评测/导出及测试 |
| `docs/`、`references/` | 实验文档、原理教程、论文 |
| `results/` | 小型结果及诊断图像 |
| `report/`、`output/pdf/` | 图表、LaTeX和PDF |
| `data/`、`outputs/`、`archives/`、`logs/` | 运行资产，不随Git提供 |
