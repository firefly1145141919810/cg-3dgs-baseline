# 3DGS baseline

这是计算机图形学大作业的第一阶段：建立可信、可重复的原始 3D Gaussian Splatting 对照系统，暂不实现扩展。用多张有相机信息的照片，优化一个由彩色三维高斯组成的场景，再从训练中没用过的视角渲染图片、计算质量指标。

**状态：代码与文档已准备，未在本地或云服务器执行训练。目标云服务器的安装、CUDA 编译、训练和评测仍待实际验证。不要把工程准备完成写成论文复现成功。**

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
