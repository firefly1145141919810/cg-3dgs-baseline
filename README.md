# 3DGS baseline

这是计算机图形学大作业的 3DGS 工程：以固定官方 baseline 为对照，研究质量保持压缩。用多张有相机信息的照片，优化一个由彩色三维高斯组成的场景，再从训练中没用过的视角渲染图片、计算质量指标。

**状态：train、truck的70%/50%/30%预算实验均已云端完成并核查。50%贡献恢复组相对同预算全模型PSNR分别低0.0526、0.0085 dB，模型减半，渲染约快1.45、1.40倍；30%加速更多，但质量损失扩大。贡献度方法并非所有场景和预算下都全面优于不透明度方法。本地未运行 GPU 模型。**

当前任务入口：[压缩实验手册与持续结果记录](docs/COMPRESSION_EXPERIMENT.md)。先做保留100%的保存重载检查，再评测opacity保留50%，不会覆盖原baseline。

当前下一步：`mkdir -p logs`后`sbatch timing.slurm`，对已有最终模型做独立重复计时，不重新训练。最终验收范围与停止条件见实验手册第23节，运行及上传说明见第24节。

最新更新：两场景70%/30%预算实验已完成，提交b756c61核查通过。结合既有50%结果，推荐50%作为主要展示工作点，30%作为激进压缩边界；完整结果及限制见实验手册第22节。第21节保留运行命令，后续优先补独立重复计时和对比图。

具体已检查与未检查事项见 [验证范围](docs/STATUS.md)。

核心算法采用作者的官方实现，而不是另写一个外观相似的玩具版本。固定提交见 `baseline.lock.json`；`vendor/gaussian-splatting` 为 Git 子模块。当前分支不启用后来的曝光补偿、深度监督、抗锯齿或稀疏优化器。

## 建议阅读顺序

1. [零基础详细讲解](docs/BASELINE_GUIDE.md)：从“立体照片”一直讲到投影、混合、损失、增密和源码。
2. [云服务器安装与运行](docs/CLOUD.md)：准备独立环境、下载数据、先冒烟测试再正式实验。
3. [论文批注索引](references/READING_NOTES.md)：配合原文和中文批注版 PDF 阅读。
4. [实验协议与验收](docs/PROTOCOL.md)：避免测试集泄漏、分辨率变化和计时口径不一致。
5. [GitHub 与结果同步](docs/SYNC.md)：小文件入 Git，大模型用文件传输。

论文已下载：[作者原文](references/3dgs_original.pdf) · [逐页中文批注 PDF](output/pdf/3dgs_annotated_zh.pdf)。批注直接显示在侧栏，不需要打开 PDF 评论面板。

## 云端快速入口

```bash
git clone git@github.com:firefly1145141919810/cg-3dgs-baseline.git
cd cg-3dgs-baseline
python scripts/bootstrap.py
conda create -n cg3dgs python=3.10 -y
conda activate cg3dgs
# 仅当 nvcc --version 为 12.1，GPU 为该工具链支持的型号时：
bash scripts/setup_cloud.sh cu121
# 阅读数据集条款后，再执行下载（约 650 MB）：
python scripts/download_data.py --accept-dataset-terms
python scripts/run.py all --config configs/smoke.json
python scripts/run.py all --config configs/train.json
python scripts/benchmark.py --config configs/train.json
python scripts/export_results.py --config configs/train.json --name train-baseline-first
```

若只有 CUDA 11.8，改用 `cu118`。如果 CUDA 版本不是这两种，或者是 Blackwell 等更新架构，先看 CLOUD.md，不要强行安装旧环境。已存在的 LLM Conda 环境不要覆盖。

## 文件分工

| 路径 | 用途 |
| --- | --- |
| vendor/gaussian-splatting | 官方优化、渲染、评测及 CUDA 源码，保持不修改 |
| configs | 正式 30k 与快速 1k 配置 |
| scripts | 环境诊断、数据检查、流程封装、计时与结果导出 |
| docs | 中文原理、安装、实验与同步说明 |
| references | 原论文、学习批注及来源信息 |
| results | 真实实验的小型指标与协议，适合 Git |
| data、outputs、archives | 云端数据、模型及归档，不进入 Git |

默认先使用作者预处理的 Tanks and Temples `train` 场景。这里 `train` 是“火车场景”的名字，不等于训练集；该场景仍划分训练视角和测试视角。

## 完成 baseline 的标准

环境检查通过 → 1k 冒烟测试完成 → 30k 正式训练完成 → 留出测试视角渲染完成 → PSNR/SSIM/LPIPS 有效 → 保存参数、划分、版本与真实对比图。要比较速度，再完成独立推理计时。

只有完成上述云端步骤后，才能说 baseline 已经跑通。后续选定扩展后，以相同数据和协议与这个版本比较。来源和 AI 辅助范围见 [NOTICE.md](NOTICE.md)。
