# 云服务器安装与运行

本教程面向Linux＋NVIDIA GPU，项目训练在Linux计算环境验证，未验证本地Windows训练。bash命令在相应Linux环境执行；下载论文、编辑文档与静态代码检查不需要GPU。

## 1 先确认你实际登录的环境

```bash
nvidia-smi
nvcc --version
which python
conda env list
g++ --version
```

这几个输出回答不同问题：

- `nvidia-smi` 显示 GPU 和驱动。右上角的 CUDA Version 是驱动支持的 CUDA 版本信息，不等于当前 `nvcc` 编译器版本。
- `nvcc --version` 显示 CUDA Toolkit 编译器版本。只安装 PyTorch CUDA wheel 并不会自动提供编译扩展所需的完整 Toolkit。
- Conda 决定 Python 和包安装在哪个隔离环境中；激活错环境会导致“刚装完却无法 import”。
- 3DGS 需要编译自定义 C++/CUDA 扩展，所以“另一个 PyTorch 项目能运行”不足以证明这些工具齐全。

不要直接照搬其他项目的环境快照；能运行一般PyTorch代码不等于能够编译本项目CUDA扩展。

## 2 获取仓库与固定版本源码

```bash
git clone https://github.com/firefly1145141919810/cg-3dgs-baseline.git
cd cg-3dgs-baseline
python scripts/bootstrap.py
```

公开仓库HTTPS克隆不需要SSH key或登录；推送才需要权限和身份验证。不要把token写进命令行URL、文档或聊天。

bootstrap 只拉取训练需要的两个 CUDA 子模块及它们的依赖，不下载/编译可选的 SIBR 桌面查看器。不要直接运行 `git submodule update --remote`，它会破坏固定版本。

## 3 使用独立 Conda 环境

```bash
conda create -n cg3dgs python=3.10 -y
conda activate cg3dgs
```

若该名称已经存在，先确认它确实属于此项目，或换名；不要删除你原来的环境。默认提供两个可检查的安装配置：

| 配置 | Python | PyTorch / torchvision | nvcc Toolkit |
| --- | --- | --- | --- |
| cu121 | 3.10 | 2.2.2 / 0.17.2 | 12.1 |
| cu118 | 3.10 | 2.2.2 / 0.17.2 | 11.8 |

这是工程兼容性起点，不是原论文的软件环境。cu121已在本项目RTX 4090 D环境验证，不保证其他硬件或cu118配置同样通过。原上游environment.yml使用较旧的Python/PyTorch；这里单独固定配套版本，实验时记录实际环境。

### nvcc 为 12.1 时

```bash
bash scripts/setup_cloud.sh cu121
```

### nvcc 为 11.8 时

```bash
bash scripts/setup_cloud.sh cu118
```

如果服务器安装了多个 Toolkit，用管理员提供的 module 命令选择版本，或设置实际存在的目录，例如：

```bash
# 仅在该目录实际存在时使用
export CUDA_HOME=/usr/local/cuda-12.1
export PATH="$CUDA_HOME/bin:$PATH"
nvcc --version
```

编译器需要与所选 Toolkit 兼容，不能仅升级到最新 GCC 就认为更好。缺少编译器时可请管理员安装 build-essential；没有 sudo 的共享集群应使用管理员提供的编译器模块。

### CUDA 为 12.4/12.8/13.x 或 GPU 为 Blackwell 等新架构

不要仅把脚本里的 cu121 改成新数字。需联合选择支持该 GPU 的 PyTorch、torchvision、Toolkit、编译器，并验证旧 CUDA 子模块能否编译。先保存 `nvidia-smi`、`nvcc --version`、`g++ --version` 的输出再做适配。诊断脚本对已知不适合的更新架构直接报错，不会暗中降级到 CPU。

安装后：

```bash
python scripts/doctor.py
```

该命令检查 PyTorch 能否执行一个很小的 CUDA 运算、能否导入两个扩展。它不是训练成功证明。首次 LPIPS 评测还会下载 VGG/LPIPS 权重，云服务器需要访问相关官方资源；无网络时需事先在可信环境缓存权重，不要用来历不明的 checkpoint。

## 4 准备第一个场景

下载脚本显示进度、速度与剩余时间，并自动尝试续传 `data/tandt_db.zip.part`。网络中断后重新运行同一命令即可，不要删除 `.part` 及其 `.part.json` 元数据，也不要同时启动两个下载进程。旧版本生成、没有元数据的 `.part` 也可尝试续传；下载完成会检查 ZIP CRC。服务器忽略 Range 或通过 If-Range 表示文件已变化时，会提示并从零下载，不会把完整响应追加到旧文件。若响应范围异常、版本不一致或 ZIP 校验失败，脚本停止并保留文件供排查；CRC 不是来源真实性校验。服务器未提供总大小时，只能显示已下载量和速度。

第一轮使用作者提供的 COLMAP 预处理数据，省去原始照片重建问题。来源与条款：

- 作者项目：https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- Tanks and Temples：https://www.tanksandtemples.org/
- Deep Blending：https://repo-sam.inria.fr/fungraph/deep-blending/

确认可以按相应条款使用后：

```bash
python scripts/download_data.py --accept-dataset-terms
python scripts/dataset.py data/tandt/train
```

预期结构：

```text
data/tandt/train/
  images/                 去畸变照片
  sparse/0/
    cameras.bin           相机内参
    images.bin            相机姿态与图像名称
    points3D.bin          稀疏三维点
```

支持对应的 COLMAP 文本文件。如果解压多了一层外目录，修改 JSON 配置中的 `source`，不要随意把 sparse 文件和另一套 images 拼在一起。下载器不覆盖已有文件，并记录下载文件哈希；该哈希用于日后复核，不冒充发布方的校验值。

正式配置：`resolution=-1`，沿用上游宽度大于 1600 时缩至 1600 的默认策略；否则保持原尺寸。`resolution=1` 表示原尺寸，2/4/8 表示缩小倍数。正式实验与后续扩展必须一致。

## 5 先做 1000 步冒烟测试

```bash
python scripts/run.py plan --config configs/smoke.json
python scripts/run.py all --config configs/smoke.json
```

plan 只打印将执行的命令。all 按顺序训练、渲染测试视角、算指标。冒烟测试降低分辨率，作用是验证安装、数据、扩展和评测链路；它不是高质量 baseline，结果不能与正式 30k 混在一起报告。

代码不会自动重复训练已有输出目录。若需要重跑，在配置中指定一个新 output；如果是中断恢复，使用下一节的 resume。

## 6 正式训练 30000 步

```bash
python scripts/run.py all --config configs/train.json
```

建议在 tmux 会话或集群调度器分配的 GPU 作业中执行，避免断开 SSH 导致任务退出；不要在集群登录节点启动训练。一次任务默认使用一张可见 GPU。多张 GPU 不会让原版脚本自动进行分布式训练；可以分别跑不同场景，并设置不同 output 和 GUI 端口。

```bash
# 单机示例：选择一张获准使用的 GPU
CUDA_VISIBLE_DEVICES=0 python scripts/run.py all --config configs/train.json
```

7k 和 30k 保存 Gaussian PLY；7k、15k、30k 保存训练 checkpoint。测试视角仅用于质量检查，不参与反向传播。种子沿用上游 0，但 CUDA 并行、库版本及硬件差异仍可能导致数值差异。

中断恢复示例：

```bash
python scripts/run.py train --config configs/train.json \
  --resume outputs/train_baseline_30k/chkpnt15000.pth
python scripts/run.py render --config configs/train.json
python scripts/run.py metrics --config configs/train.json
```

必须使用同一次运行的可信 checkpoint、相同配置和数据。旧版 checkpoint 不保存所有随机数状态和相机抽样队列，因此恢复不保证逐位等同不中断训练；日志会记录恢复命令。不要把 PLY 传给 --resume。

## 7 找到结果并进行速度评测

```text
outputs/train_baseline_30k/
  manifest.json、dataset.json、environment.json、packages.json
  train_*.log、render_*.log、metrics_*.log、stage_history.jsonl
  point_cloud/iteration_30000/point_cloud.ply
  test/ours_30000/renders/   模型预测图
  test/ours_30000/gt/        对应真实图
  results.json              汇总质量
  per_view.json             每视角质量
```

```bash
python scripts/benchmark.py --config configs/train.json
python scripts/export_results.py --config configs/train.json --name train-baseline-first
```

benchmark 预热后用 CUDA events 测试官方 Python 渲染入口，排除 PNG 编码和磁盘写入。不能把它的 FPS 写成论文 SIBR 查看器的 FPS。显存字段仅为 PyTorch allocator 的峰值，不是整个 GPU 的总占用或训练峰值。

## 8 常见问题

| 现象 | 优先检查 |
| --- | --- |
| nvcc 不存在 | 只有运行库，没有完整 Toolkit，或 PATH 未选择正确版本 |
| CUDA version mismatch | torch.version.cuda 与实际 nvcc 版本 |
| no kernel image | GPU 架构是否被当前 PyTorch/Toolkit 支持 |
| 编译 cc1plus 被杀 | 主机内存不足，降低 MAX_JOBS，不只是显存问题 |
| simple_knn / rasterizer 导入失败 | 是否在当前 Conda 中重新编译；是否残留别的环境安装 |
| only undistorted supported | 需要去畸变 COLMAP 数据，不支持直接把鱼眼/畸变参数送入 baseline |
| 输出模型很差 | 图像与相机是否对应，训练是否足够，分辨率是否误改 |
| 指标没生成但程序退出 0 | 旧 metrics.py 会吞异常；包装器额外检查输出，查看日志中的原因 |
| LPIPS 下载失败 | 评测权重网络/缓存问题，不是训练模型坏了 |
| 显存不足 | 数据图像缓存、Gaussian 数量、评测整套图片一起入 GPU 都会占显存 |

数据图像可用 `data_device=cpu` 降低常驻显存，但会影响时间口径；改变后应使用新的运行标签。先完成一个场景再扩大范围，不要一开始下载所有 benchmark。
