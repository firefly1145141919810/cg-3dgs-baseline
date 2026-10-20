# 3DGS 质量保持压缩实验手册与结果记录

我们要研究的问题是：一个已经训练好的 3DGS 场景，能否用更少的高斯、更小的模型文件，保持接近原来的新视角画质？这里的“压缩”首先指删除部分高斯，不是把 PLY 打成 ZIP。目标是建立画质与资源开销之间的量化关系，不预设压缩一定提高 FPS，也不把压缩当作修复原来重建失败区域的方法。

本手册是两位同学的操作指南和持续维护的实验记录。原理和计划写在前半部分，实际运行命令、验收与结果表写在后半部分。之后每次提交云端结果，应核对原始记录，再更新本文的结果和结论；失败实验也保留。模型代码或实验协议改变时必须新增记录，不能覆盖旧结果后沿用旧结论。

## 1 当前进展和本轮范围

| 工作 | 当前状态 |
| --- | --- |
| 官方 2023 baseline 的 train 场景 30k | 已由用户在云端完成，指标与六组图像已同步检查 |
| 第一阶段代码与 CPU 合成数据测试 | 已实现，21项CPU测试通过，云端结果已核查 |
| 第一阶段保留 100% 的参数与渲染检查 | 通过；云端记录38视角PNG像素完全一致 |
| 第一阶段 opacity 保留 50% 的剪枝评测 | 已完成；约1.63倍渲染速度，PSNR下降2.3863 dB |
| 可见贡献统计 | 第三阶段正式结果已核查；保留50%仅下降0.214577 dB |
| 恢复训练与不剪枝继续训练对照 | 第二阶段3k正式结果已核查；剪枝恢复后PSNR为21.175407 |
| 评分方法×恢复训练六组对照 | 第四阶段正式结果已核查；贡献恢复组与同预算全模型PSNR差0.052603 dB |
| truck 跨场景复现 | 0af71e9结果已核查，贡献恢复组与同预算全模型PSNR差0.008457 dB |
| 多预算完整实验 | 70%/30%代码已实现；既有50%结果保留，新增预算待云端执行 |

本轮不需要重新训练 baseline、不更换 CUDA 扩展、不下载新权重。已有的 numpy、Pillow、plyfile 和 baseline 环境即可使用。准备阶段只处理 PLY；评测阶段在 GPU 节点调用原版渲染与评测程序。整个第一阶段没有梯度更新。

课程第13题要求 baseline 加一个实质扩展，并报告 FPS、显存或模型大小与视觉质量。当前 opacity 剪枝是为后续扩展搭建可信对照，不应把“按透明度删一半”本身包装成完整的高分创新项目。

## 2 为什么可以尝试减少高斯

训练过程根据图像误差增加、拆分或删除高斯，主要目标是拟合照片，而不是找到参数数量最少的表示。训练结束后可能仍存在被遮挡、贡献很弱或作用重复的高斯。剪枝就是选择一个子集，保留其位置、颜色、尺度、旋转和透明度，其余高斯不再参与渲染。

删除并不总是无害。一个很小的高斯可能负责一根扶手的边缘；一个从多数视角看不到的高斯可能在某个视角十分重要。所以我们不只比较文件大小，还必须看测试视角的质量和失败案例。

当前 baseline 的高斯数为1,082,124，PLY为268,368,284字节，约255.94 MiB。保留一半时目标数量为541,062；同样的字段和精度下，文件大小应接近一半，但速度不一定翻倍。屏幕覆盖、排序、背景和调度开销也影响耗时。

## 3 第一阶段的两个模型

### 保留全部高斯的对照

`identity100` 不改变任何参数，也不改变高斯顺序，但实际经历“读取 PLY → 选择全部行 → 写出新的 PLY → 重新读取 → 原版 GPU 渲染”的过程。

这一关用于排除实现错误：例如把颜色系数顺序写乱、把对数尺度误当成真实尺度、把四元数各分量错位等。如果保留所有高斯时画面都变了，之后观察到的质量下降就不能归因于剪枝。

代码做两层检查：

1. CPU 层逐字段比较所有选中行的 float32 位模式，确认保存重载后参数不变。
2. GPU 层重新渲染所有测试视角，与原 baseline 已保存的 RGB PNG 逐像素比较；GT 也必须完全相同。

默认要求解码后 PNG 像素完全一致。它能验证实际输出图像，但不等于证明浮点渲染张量在 PNG 量化前逐位一致。如果更换硬件或软件导致微小差异，脚本会停止并报告差异，不会擅自放宽阈值。应先核查环境与参数再决定处理方法。

### 保留一半高斯的朴素剪枝

`opacity50` 按每个高斯自身的 opacity 排名，保留最高的50%。注意“保留50%”不是“透明度大于0.5”。

原版 PLY 的 `opacity` 字段保存的是原始参数 z，真正的 opacity 为：

    opacity = sigmoid(z) = 1 / (1 + exp(-z))

sigmoid 严格单调递增，因此按 z 排名与数学上按 opacity 排名相同。实现直接使用 z，避免很大或很小的数经过 sigmoid 后数值饱和，产生人为并列。

具体选择规则固定如下：

- 保留数量为 max(1, floor(N × 保留比例))。
- 分数相同时，原始行号较小的高斯优先保留。
- 选中以后按原始行号排序，再写出全部参数。
- 不修改任何选中高斯的参数值，保留SH三阶和float32精度。

为什么要恢复原始顺序？前后混合与排序有关；在深度并列等情况下，任意重排可能影响图像。尤其是100%对照不应该因为“按分数重新排列”而改变结果。

每个高斯的所有字段必须用同一组选中行一起裁剪，不能分别为位置、颜色等做不同选择。`kept_indices.npy` 保存原始行号，便于追踪和复核。

## 4 下一阶段的重要性评分

opacity 只反映高斯自身的不透明程度，不知道它是否被遮挡，也不知道它在最终画面里占多少权重。后续主要方法计划统计可见贡献。

对某个像素，高斯的颜色混合权重为：

    w_i(p) = T_i(p) × alpha_i(p)

alpha_i(p) 是 opacity 乘上该像素处的二维高斯权重；T_i(p) 是前方所有高斯混合后剩余的透过率。一个自身不透明、但被完全挡住的高斯，实际可见权重仍然很小。

计划中的基础评分是对选定训练视角和像素累积 w，并按视角像素数与视角数量归一化。它衡量平均可见贡献，但并不是“删掉该高斯后真实误差”的精确计算：删除以后，后面的高斯会变得更可见，图像并非简单少掉一项。

原计划是在CUDA前向渲染中新增统计输出。第三阶段改为复用现有渲染器的预计算颜色梯度：
它已经包含同一个 `T × alpha` 权重，因而无需改动或重编译CUDA。具体推导与数值关卡见第15节。
评分仍是独立离线步骤，正式测FPS调用未修改的原版渲染，不包含梯度统计开销。

重要性评分借鉴 LightGaussian 的剪枝与恢复思想，但上述评分是本课程项目拟实现的代理量；未证明与论文完整评分等价，也不包含论文的SH蒸馏或向量量化。来源：[LightGaussian 作者项目页](https://lightgaussian.github.io/)。报告必须标出实际采用、修改和未实现的部分。

## 5 后续恢复训练怎样设计

恢复训练让剩余高斯重新分工，弥补删除带来的图像误差。初步开发预算为3,000步，不是已验证最优值。

计划固定高斯数量，关闭增密、自动裁剪与 opacity reset，使用与 baseline 相同的图像损失。第一版从剪枝后的参数重新建立优化器；不剪枝继续训练的对照也使用同样的优化器重建和学习率策略，避免恢复收益被不公平比较夸大。

第一阶段保存的 `iteration_30000` 只是沿用原版渲染器的读取目录，表示来源模型训练到了30k，不表示剪枝模型又训练了30k。本轮记录 `recovery_steps=0`。这些派生目录不包含可恢复的优化器 checkpoint，不能把它们当作原版训练的 `--resume` 输入。

## 6 完整实验要回答的问题

| 对照 | 问题 |
| --- | --- |
| 原始模型与100%保存重载 | 工程处理是否改变了结果？ |
| opacity剪枝与贡献评分剪枝 | 更复杂的评分是否值得？ |
| 同一剪枝模型恢复前与恢复后 | 恢复训练能挽回多少质量？ |
| 剪枝恢复与不剪枝同预算继续训练 | 收益是否仅仅来自额外优化？ |
| 保留70%、50%、30% | 质量从哪个预算开始显著恶化？ |
| train与truck | 结论是否仅对单场景有效？ |

100%与50%无恢复实验已完成。根据首轮结果，先在50%预算实现恢复训练及不剪枝同预算优化对照，再加入贡献评分，最后铺开两个场景、三个预算。这样先隔离“删掉以后能否通过优化补回来”，再隔离“是否选错了要删的高斯”。可以把“保留一半时PSNR损失约不超过0.5 dB”设为探索目标，不能写成已经达到或保证达到的成绩。

### 数据使用边界

本轮保留比例和评分已经预先固定，不根据测试分数挑阈值。opacity排序不读取图像误差，评测仍沿用现有263训练视角和38测试视角。

开发新的评分权重、学习率或恢复步数时，应从训练视角中另划固定验证集。用于验证的照片不能再参与该开发实验的模型优化或评分统计；这意味着相应开发对照也要按一致划分重新建立，不能把训练过这些照片的旧模型冒充独立验证。冻结方案后，正式实验可按统一的完整训练视角协议重新运行，再评测保留测试集。

我们已经看过 `00033` 等测试图，因此这批图可作为透明披露的诊断案例，不能再围绕它们反复调参并宣称无偏泛化提升。第二个场景和预先固定的实验方案有助于减少这种偏差。

## 7 云端运行前需要保留什么

必须在云端保留原始 baseline 完整输出，而不只是 GitHub 上的指标：

    outputs/train_baseline_30k/
      manifest.json
      dataset.json
      results.json
      per_view.json
      point_cloud/iteration_30000/point_cloud.ply
      test/ours_30000/renders/00000.png ... 00037.png
      test/ours_30000/gt/00000.png ... 00037.png

还需要原始数据集 `data/tandt/train`。准备阶段会核对相机、图片哈希、划分、模型与版本，拒绝把不同数据或不同模型混到一起。不要修改原配置来强行绕过报错。

服务器使用现有 `cg3dgs` 环境。本轮无需新 CUDA 编译；权重缓存仍沿用 `data/torch-cache`。不要重新执行联网安装脚本。

## 8 第一阶段运行步骤

### 第一步 在联网登录节点更新

```bash
cd /path/to/cg-3dgs-baseline
git pull --ff-only
conda activate cg3dgs
python -m unittest discover -s tests -v
```

测试使用人工构造的小型PLY和图像，不执行GPU渲染或训练。第一阶段新增测试覆盖参数不变、选择规则、错误输入、图片比较、指标校验及100%关卡失败后禁止继续。

### 第二步 准备两个模型

该步骤是CPU任务，涉及读取和复制约256 MiB模型，需预留至少数GiB内存及磁盘空间。若集群不允许登录节点做此类处理，把这一步也放在获批的计算节点。

```bash
python scripts/compression_phase1.py prepare
```

默认输出到 `outputs/compression/train_phase1/`，创建 `identity100` 与 `opacity50`。原 baseline 不会被改写。CPU参数检查通过时打印 Prepared；这还不代表GPU渲染验证通过。

若目录已存在，prepare会拒绝覆盖。已经准备完成的实验直接执行evaluate；如果prepare中断、没有生成 `experiment.json`，保留现场并换一个目录：

```bash
python scripts/compression_phase1.py prepare \
  --experiment outputs/compression/train_phase1_retry
```

使用自定义目录后，evaluate和export也要传入同一个 `--experiment`。路径默认为相对仓库根目录，而不是当前终端目录。

### 第三步 在已分配GPU节点评测

```bash
cd /path/to/cg-3dgs-baseline
conda activate cg3dgs
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_HOME="$PWD/data/torch-cache"

python scripts/doctor.py
python scripts/compression_phase1.py evaluate
```

执行顺序固定为：

1. 检查环境、源码、数据、模型和参考图片未改变。
2. 渲染identity100，核对所有GT及渲染PNG。
3. 100%检查通过后，计算其画质指标和新的渲染计时。
4. 渲染opacity50、核对GT，计算其指标和计时。
5. 两组均成功才写入 `evaluation.json` 的 `complete: true`。

100%检查失败时，脚本停止，50%模型虽然已经准备好但不会继续评测。不要为了继续运行而修改passed标志。上传 `identity_check.json` 与相关日志排查即可。

如果只是评测阶段中断，可以再次运行evaluate：它会重新评测两个固定模型，不训练、不改模型；旧指标文件会另存为带时间标记的备份。不能同时对同一个实验目录运行两份脚本。

使用tmux可以减少SSH断线影响，但不能延长GPU作业时限。请在有效GPU分配内运行。

### 第四步 导出结果和图片

两个评测均成功后执行：

```bash
python scripts/compression_phase1.py export \
  --name train-opacity-phase1-first
```

导出目录为 `results/train-opacity-phase1-first/`，包含：

- 两组指标、相对原始baseline的画质差值和模型大小；
- 每组114次计时样本、逐视角指标；
- 100%重载检查结果；
- 版本、模型哈希、保留规则和测试协议；
- 六个固定视角的两组渲染图，以及同名GT。

本轮自动导出图片，避免此前只提交指标而遗漏图片的问题。视角0、14、24、31、33、35包含此前看过的较好与较差案例，不声称是随机样本。PLY、行号数组和完整运行日志仍保留在outputs，不进入Git。

导出标签必须唯一。已经成功导出后无需重复执行；重新评测的结果使用新标签，保留历史。

### 第五步 在联网登录节点提交

```bash
git status --short
git add results/train-opacity-phase1-first
git commit -m "Add opacity pruning phase 1 results"
git push origin main
```

GPU节点离线时，不在GPU节点执行push。上传后告知“第一阶段结果已推送”，再由本地拉取、核查与维护本文。不要强制添加outputs，不要上传私钥或模型checkpoint。

## 9 如何读输出

`identity_check.json` 的 `passed: true` 表示所有测试PNG像素一致；`changed_channels` 是发生变化的颜色通道数量，不是高斯数。`mean_abs_8bit` 的单位为0到255的像素值，不是归一化图像误差。

`summary.csv` 每行对应一个模型。PSNR与SSIM越高通常越好，LPIPS越低通常越好；delta均为“本组减原baseline”，因此负的delta_PSNR通常是退化，负的delta_LPIPS通常是改善。

剪枝前后速度优先比较本轮新测的identity100和opacity50，而不是把旧baseline另一时间段的201 FPS作为唯一分母。新计时同样预热20帧、测试38视角重复3轮，记录114个样本。运行时避免其他任务占用同一GPU；小幅差异需要重复测量，不能直接归因于算法。

`mean_ms` 是CUDA event覆盖的原版Python渲染路径耗时，不含PNG保存和磁盘IO，不是完整交互系统帧率。`peak_torch_allocated_bytes` 是推理过程中PyTorch分配器记录的峰值，`reserved` 是其保留内存；二者都不是训练峰值，也不等于nvidia-smi总显存。

模型大小统一用PLY实际字节数；MiB = bytes / 1,048,576。高斯数量压缩比 = 原数量 / 新数量，文件压缩比 = 原PLY字节数 / 新PLY字节数。只压缩ZIP体积而不改变运行时模型，不计为本项目的高斯剪枝效果。

## 10 验收标准与失败处理

| 检查 | 合格标准 | 不合格时 |
| --- | --- | --- |
| PLY结构 | 官方SH3的62个float32字段，非空且有限 | 检查是不是错误的普通点云或其他模型 |
| 参数重载 | 选中字段逐位不变，顺序不变 | 先修文件处理，禁止评判剪枝质量 |
| 100%图像检查 | 38张GT和渲染图逐像素一致 | 检查版本、设备、参考文件和数值差异 |
| 50%数量 | 当前train模型保留541,062个高斯 | 检查预算和模型来源 |
| 评测 | 38个视角全部有效，逐视角均值吻合汇总 | 排查漏图、LPIPS缓存、评测异常 |
| 结果导出 | complete为true且所有产物哈希未变 | 不导出陈旧或被修改的结果 |

50%模型质量明显变差并不一定是程序错误：朴素opacity剪枝本来就是对照组。只有先通过100%关卡，再确认配对与指标完整，才讨论方法本身的局限。第一阶段即使失败，也不要直接提高训练步数，因为本阶段没有恢复训练。

准备后若更新了相关脚本，evaluate会提示代码哈希改变。应使用原代码版本完成旧实验，或新建实验目录；不要手改来源记录。这避免同一个实验标签对应不同实现。

## 11 已核查结果

### 原始baseline 已核查

来源：`results/train-4090d-30k-first/`。原始结果提交665ae60，配对图像提交f54af9c。用户云端运行，非本地训练。

| 指标 | train baseline |
| --- | --- |
| 迭代数与视角 | 30,000；263训练，38测试 |
| GPU与实际分辨率 | RTX 4090 D；980×545 |
| PSNR / SSIM / LPIPS | 21.810524 / 0.812434 / 0.208012 |
| 高斯数量与PLY | 1,082,124；268,368,284字节 |
| 平均耗时与FPS | 4.972951 ms；201.087833 |
| 推理PyTorch已分配峰值 | 834,924,032字节 |

六组上传图片的PSNR已由像素重新计算，与记录吻合。车身和文字较清楚，但天空色块、近地碎石拉伸、底盘细节模糊仍明显。00033是严重失败视角；这些现象没有被确诊为同一个原因，压缩也不承诺修复它们。

### 第一阶段 云端实测与本地复核（2026-10-20）

来源：[完整汇总](../results/train-opacity-phase1-first/summary.json)、
[协议](../results/train-opacity-phase1-first/protocol.json)、
[一致性检查](../results/train-opacity-phase1-first/identity_check.json)。
结果提交 `58c83b7`，执行代码提交 `f4672af`；官方核心版本未变。
用户在RTX 4090 D上执行，263训练视角／38测试视角，980×545，SH3，恢复步数为0。

| 指标 | identity100 | opacity50 | 解释 |
| --- | --- | --- | --- |
| 高斯数 | 1,082,124 | 541,062 | 恰好保留50% |
| PLY字节数 | 268,368,284 | 134,184,907 | 文件头使字节数不是精确一半 |
| PLY大小（MiB） | 255.936 | 127.969 | 约2倍模型压缩 |
| PSNR ↑ | 21.810524 | 19.424191 | 下降2.386333 dB |
| SSIM ↑ | 0.812434 | 0.774688 | 下降0.037746 |
| LPIPS ↓ | 0.208012 | 0.232863 | 上升0.024851，变差 |
| 平均渲染耗时（ms） | 4.976176 | 3.057673 | 耗时下降38.55% |
| 由平均耗时换算FPS | 200.958 | 327.046 | 速度约1.627倍 |
| 推理PyTorch已分配峰值（MiB） | 796.246 | 397.099 | 下降50.13%，不是总显存或训练峰值 |

核查范围：

- 云端参数记录显示两组所选字段逐位不变，identity100的PLY哈希与来源一致；本地没有完整PLY，未重新执行模型级核验。
- 云端一致性报告包含38视角，最大像素差、平均像素差、变化通道数全部为0；identity100三项汇总指标与原baseline完全相同。
- 本地检查两组各38视角的三项指标，重算均值与汇总一致；两组各114次计时样本均值与汇总一致；脚本哈希与执行协议一致。
- 上传的18张图片（6张GT、两组各6张渲染）与记录的哈希一致，尺寸均为980×545。独立从像素重算12张渲染图的PSNR，与逐视角记录差异小于0.0001 dB。本地未重新计算SSIM或LPIPS，也未运行GPU模型。
- 38个视角全部出现PSNR、SSIM下降和LPIPS上升，不是少数坏图拉低了平均值。PSNR下降范围为0.188816至6.849558 dB；最严重的是00009，但该图没有上传，不能据此描述其视觉内容。
- 三轮计时均值：identity100为4.98897／4.97399／4.96557 ms；opacity50为3.06565／3.05205／3.05533 ms。同一会话内趋势稳定，但仍不是多次独立运行的统计置信区间。

### 配对渲染图观察

逐一查看了全部六组GT／identity100／opacity50。以下是固定诊断视角，不是随机抽样：

| 视角 | identity100 PSNR | opacity50 PSNR | 目视观察 |
| --- | --- | --- | --- |
| 00000 | 24.13342 | 17.77581 | 车头与文字主体仍在，但剪枝后天空更亮、远处车厢与路面亮部更突兀，整体明暗偏离GT |
| 00014 | 18.07940 | 15.49735 | 植被明暗反差增强；baseline本有的近地碎石拉伸没有消失 |
| 00024 | 18.23728 | 18.01685 | 两组较接近，车身文字保留；底盘模糊与地面失真仍然存在 |
| 00031 | 26.77513 | 23.64983 | 车身文字轮廓保留，但左侧背景、天空出现更明显的色块和亮暗偏差 |
| 00033 | 14.69416 | 11.22705 | baseline已经模糊，剪枝后踏板和底盘更暗，与较明亮的GT偏差扩大 |
| 00035 | 16.85299 | 15.78031 | 车头仍可辨认，天空及远景变化明显；未解决原有车身亮度与GT不一致 |

例如，[00000的GT](../results/train-opacity-phase1-first/gt/00000.png)、
[未剪枝](../results/train-opacity-phase1-first/identity100/renders/00000.png)、
[剪枝后](../results/train-opacity-phase1-first/opacity50/renders/00000.png)可并排查看。

结论：第一阶段工程检查通过，但朴素剪枝尚未达到“质量保持压缩”。它证明模型更小、当前渲染路径更快，也量化了明显的画质代价，作为对照很有价值。

解释与假设要分开：删除高斯会改变alpha混合和后方高斯的可见权重，这是渲染公式的直接结果；低opacity高斯合在一起可能仍对大面积颜色有贡献，因此不能把“每个都不太不透明”理解成“都没有用”。不过，哪一类被删高斯具体导致天空或车身变化，本轮没有做贡献定位，不能宣称已证实。画面有时显得更锐、更有反差，也不代表更接近GT。

### 下一步执行计划：先恢复，再改评分

本节为第二阶段实验设计；对应代码已实现，运行命令见第13节。真实GPU训练尚未验证。

1. 保留当前baseline、两个派生PLY、原始数据、配置和日志，不覆盖第一阶段结果。已提交的导出目录足够做本次图像与指标检查，不需要现在重跑。
2. 新增独立恢复训练入口，从PLY加载全部参数并新建优化器，不把PLY当作包含优化器的checkpoint。先跑短冒烟测试，验证损失有限、能保存重载、渲染参数齐全、高斯数始终不变。
3. 固定3,000步，先做两个同预算实验：identity100继续优化与opacity50恢复优化。两组都关闭增密、自动裁剪和opacity reset；使用相同损失、学习率策略与训练视角抽样种子。记录学习率具体数值和调度、初始PLY哈希、恢复耗时及最终数量，不沿用从零训练的增密流程。
4. 用四组对照回答问题：原模型、原模型+3k、opacity50、opacity50+3k。恢复收益看后两者之差；恢复后与原模型比较判断保质程度；与原模型+3k比较控制“多训练了3k步”的优势。训练只使用训练视角，测试视角仅评测；不根据本轮测试图反复挑学习率或最佳checkpoint。
5. 若恢复仍不够，不自动增加训练预算直到测试分数好看。继续实现原计划的训练视角可见贡献评分，在同样50%数量与3k恢复预算下比较评分方法。评分统计应与正式计时分离，并先验证关闭统计时不改变渲染。
6. 单预算流程可靠后，扩展70%／50%／30%与truck场景，并做独立重复计时。若要调参，遵守第6节独立验证集要求。

本轮后优先实现恢复，是为了在不更换CUDA扩展的情况下先量化“剩余高斯能否重新分工补偿误差”，不是放弃可见贡献评分。3,000步和0.5 dB目标均为预先规划的探索条件，不保证达到。

## 12 后续维护与最终报告

### 2026-10-20：首次评测配置错误与重跑

首次云端 `identity100` 渲染在加载相机后报错：
`AttributeError: 'GroupParams' object has no attribute 'white_background'`。
这是实验包装脚本的配置生成遗漏，不是CUDA或模型损坏。旧版写入空的
`Namespace()`，而原版渲染器的参数合并会忽略未传入的布尔开关；
不传 `--white_background` 并不自动补上 `False`。
修复后保存完整模型配置，显式记录黑色背景 `False`，并增加调用原版参数解析器的
CPU回归测试，覆盖黑／白背景两种情况。该测试不代表真实GPU渲染已经成功。

更新代码后，旧实验的脚本哈希不再匹配。保留旧目录和失败日志，使用新目录重建派生模型：

```bash
# 登录节点更新代码
git pull --ff-only
# GPU节点，已激活 cg3dgs 环境，在仓库根目录执行
python scripts/compression_phase1.py prepare --experiment outputs/compression/train_phase1_v2
python scripts/compression_phase1.py evaluate --experiment outputs/compression/train_phase1_v2
# 仅在 evaluate 成功后导出
python scripts/compression_phase1.py export --experiment outputs/compression/train_phase1_v2 --name train-opacity-phase1-first
```

不需要重新训练baseline、重新下载数据或重新编译CUDA扩展。
如果新目录已经存在，不覆盖它；按实验版本继续选择新的目录名。

每次用户推送结果后，按下列顺序更新本文：

1. 登记结果目录、提交、代码版本、场景与实验条件。
2. 核对关卡、哈希、视角数量、均值、计时样本与配对图片。
3. 将“待测”替换为实测数值，附对应结果路径；保留失败记录。
4. 写清楚观察、解释和仍未验证的假设，不能把推测写成根因。
5. 若修改方案，新增协议版本与实验行，不覆盖旧结论。

未来报告建议结构：问题与动机 → baseline → opacity对照 → 可见贡献评分 → 固定数量恢复 → 多预算多场景比较 → 失败案例 → 局限与总结。主要图表为质量对模型大小、质量对速度，以及GT／原模型／剪枝后／恢复后的同视角对比。

两位同学分别负责云端实验与版本记录、原理整理与结果展示，但都要理解alpha混合、为什么opacity不等于可见贡献、为什么恢复时不能增密，以及为什么单次FPS不足以证明加速。

本阶段代码入口：

- `scripts/prune_ply.py`：字段校验、稳定排序、整行裁剪与重载检查。
- `scripts/compression_phase1.py`：准备、评测关卡、来源检查、结果与图像导出。
- `tests/test_compression.py`：CPU合成数据与失败关卡测试。
- 原版 `render.py`、`metrics.py` 与现有 `benchmark.py`：沿用统一评测路径，核心算法不改动。

## 13 第二阶段：固定数量恢复训练代码与运行

### 代码范围和四组对照

`scripts/recovery_phase2.py` 提供prepare／train／evaluate／export；
`scripts/recovery_train.py` 是独立训练入口，调用固定版本官方的模型、可微渲染器和损失函数，
不修改vendor源码，不需要重新编译CUDA或联网安装依赖。

| 名称 | 初始化 | 实际新增优化步数 | 正式最终PLY目录标号 |
| --- | --- | --- | --- |
| identity100 | 第一阶段identity100 | 0 | 30000 |
| opacity50 | 第一阶段opacity50 | 0 | 30000 |
| identity100_recovery | 第一阶段identity100 | 3000 | 33000 |
| opacity50_recovery | 第一阶段opacity50 | 3000 | 33000 |

两组训练均从PLY新建Adam优化器，并不是恢复baseline原有的Adam动量。
每轮随机打乱训练视角后不放回抽取，种子固定为0；两组共享相同相机名称序列，
评测前核对序列哈希。训练只访问训练相机，测试相机不用于损失、挑checkpoint或调学习率。
GPU浮点计算不保证跨运行逐位确定，因此相同随机种子不等于完全确定性。

损失固定为 `0.8 × L1 + 0.2 × (1 - SSIM)`。完整SH3从第一步启用，
位置、颜色SH、opacity、尺度、旋转都可优化；高斯数量不变。
不调用增密、自动删除、透明度重置和SH逐级开启。
每一步都执行优化器更新，包括第3000步，然后才保存最终模型。

### 学习率预设（并未根据测试分数调优）

| 参数组 | 学习率 |
| --- | --- |
| xyz位置 | 常数 `1.6e-6 × cameras_extent` |
| SH直流颜色 | 0.00025 |
| SH其余系数 | 0.0000125（直流的1/20） |
| opacity原始参数 | 0.005 |
| 对数尺度 | 0.0005 |
| 旋转四元数 | 0.0001 |

Adam使用`eps=1e-15`，其余默认参数沿用官方训练设置。位置使用baseline训练末期的
学习率，其他组使用官方默认值的十分之一，作为预先固定的保守恢复方案；
这不是已证实的最佳超参数，也不保证恢复到0.5 dB以内。

关键工程细节：官方`load_ply`不会恢复`spatial_lr_scale`，
所以显式将其设为场景的`cameras_extent`。否则位置学习率会乘以默认0而停止更新。
模型报告会记录实际学习率和空间尺度。

### 运行前提

保留baseline完整输出、数据、已成功评测的`outputs/compression/train_phase1_v2`。
默认parent就是这个目录；若你实际使用不同目录，prepare时明确传入`--parent`。
第二阶段会验证第一阶段的代码、输入及评测产物哈希，不能只从Git导出的六张图重建训练。
本次未改动第一阶段脚本，原有哈希应继续匹配。

准备额外至少10 GiB可用磁盘空间作为保守预留：四组初始PLY、两组最终PLY、
含Adam状态的断点、断点写入临时副本、四组完整渲染图和日志都在outputs。
实际占用随模型和图片大小变化。不要将这些大文件强制加入Git。
训练图片保留在CPU，每步仅选定GT转到GPU，仍需足够主机内存。

登录节点更新（仓库根目录）：

```bash
git pull --ff-only
conda activate cg3dgs
python -m unittest discover -s tests -v
```

所有后续GPU操作均在有效GPU作业分配内进行，不能直接在登录节点训练。

### 先跑20步冒烟测试

在GPU节点、仓库根目录：

```bash
conda activate cg3dgs
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_HOME="$PWD/data/torch-cache"
python scripts/doctor.py

python scripts/recovery_phase2.py prepare \
  --experiment outputs/compression/train_recovery_smoke20 --steps 20
python scripts/recovery_phase2.py train \
  --experiment outputs/compression/train_recovery_smoke20
python scripts/recovery_phase2.py evaluate \
  --experiment outputs/compression/train_recovery_smoke20
```

冒烟测试也完整渲染／评测四组，因而评测耗时可能超过20步训练本身。
目的是验证反向传播、保存重载、模型数量、旧对照图像一致性和指标链路；
不要求20步就有质量提升。协议中的`formal: false`防止把它误当正式结果。
全部成功后，继续以下正式实验；如果报错，保留日志排查，不要跳过检查。

### 正式3,000步实验

```bash
python scripts/recovery_phase2.py prepare
python scripts/recovery_phase2.py train
python scripts/recovery_phase2.py evaluate
python scripts/recovery_phase2.py export --name train-recovery3k-first
```

逐条执行，上一条成功才执行下一条。默认目录为
`outputs/compression/train_recovery_3k`。正式实验重新从第一阶段PLY开始，
不接着20步冒烟模型继续训练，避免训练预算混淆。
train顺序执行两个恢复模型，不同时占用GPU；evaluate对四组重新计时，
并核对两组未训练对照的PNG仍与第一阶段完全一致。
脚本不自动挑选PSNR最高的训练中间点，正式评测固定最终3000步。

### 中断与续跑

每500步以及最后一步保存本脚本专用的`recovery_checkpoint.pth`，
包含模型、Adam状态、随机状态、已完成步数、损失记录和实验哈希。
先写临时文件，再替换旧断点；中断后重新执行同一个train命令即可。
若尚未到第一次保存，则从初始PLY重新开始。已完成且哈希通过的训练组自动跳过。
断点只加载本脚本在可信本地outputs中生成的文件；不要下载并加载来路不明的pth。
训练步数记录的是累计优化次数，不会把恢复后执行的第一步重复计数。
断点续跑的GPU数值结果仍不承诺逐位一致。

不要对同一实验目录同时运行多个train/evaluate，也不要在实验进行中更新代码。
更新相关代码会触发哈希保护，应保留旧目录并建立新实验；不手改哈希。
评测中断可单独重跑evaluate，不需要重新训练；旧指标会保留带时间戳的备份。
prepare拒绝覆盖已有目录。新实验使用新目录名，并在后续各命令带相同`--experiment`。

### 上传与验收

成功导出后，在联网登录节点执行：

```bash
git add results/train-recovery3k-first
git commit -m "Add fixed-count recovery results"
git push origin main
```

导出包含四组指标、逐视角分数、计时、固定六视角渲染、GT、训练损失轨迹、
训练耗时和学习率；不包含PLY或优化器断点。训练耗时排除断点写入，
也不包含场景加载、最终PLY保存和评测，不能当作端到端作业时长。

验收首先确认两组恢复的数量仍分别为1,082,124和541,062、优化步数恰为3000、
相机序列一致，且全部指标有限。然后比较opacity50恢复前后、恢复后与原模型、
恢复后与未剪枝同预算优化。只有这些结果返回，才能判断是否恢复有效。

第二阶段交付时24项CPU测试通过，涵盖固定抽样序列、预算校验、无拓扑修改调用、
模拟四组prepare/evaluate/export及产物篡改拦截；静态解析通过。
现已收到正常完成的云端训练与评测结果，见第14节；实际中断后恢复的行为尚无云端记录。

## 14 第二阶段结果核查

结果提交`58d6689`，执行代码`a4327cc`，来源
[train-recovery3k-first](../results/train-recovery3k-first/summary.csv)。
RTX 4090 D，同一263训练／38测试划分与980×545分辨率；seed=0，恢复3000步。

| 模型 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 高斯数量 | ms | FPS |
| --- | --- | --- | --- | --- | --- | --- |
| identity100 | 21.810524 | 0.812434 | 0.208012 | 1,082,124 | 4.967687 | 201.301 |
| opacity50 | 19.424191 | 0.774688 | 0.232863 | 541,062 | 3.057293 | 327.087 |
| identity100_recovery | 22.139515 | 0.818247 | 0.203941 | 1,082,124 | 5.024080 | 199.041 |
| opacity50_recovery | 21.175407 | 0.802914 | 0.221098 | 541,062 | 3.086492 | 323.992 |

**结论：恢复明显有效，但还不能称为近无损压缩。**

- opacity50恢复增加1.751217 dB，证明让留下的高斯重新优化可以补回相当一部分误差。
- 与原baseline仍差0.635117 dB，未达到此前约0.5 dB的探索目标。
- 与同样多优化3000步的未剪枝组差0.964108 dB。后者自己也提升了0.328991 dB，因此不能把全部恢复收益都当成压缩方法的优势。
- 恢复后数量不变，PLY仍约127.97 MiB；相对未剪枝同预算组渲染速度约1.628倍。训练耗时报告分别为58.64秒和36.12秒，不含加载、断点写入、最终PLY保存和评测，不是作业总耗时。

核查：四组各38视角指标均值、各114次计时样本与汇总一致；上传的GT和四组六视角渲染共30张图片哈希一致。24张渲染的PSNR由像素独立重算，与记录差异小于0.0001 dB。
两组训练报告各有3000条连续、有限的损失记录，最终数量正确，相机序列哈希相同，
实际学习率相同（xyz为1.192282104492188e-5，含场景尺度）。本地没有重新跑GPU、SSIM／LPIPS或完整PLY核验，也没有证据确认用户曾测试中断续跑。

查看了两组恢复后的全部六张诊断图。00000的剪枝模型已比无恢复时更接近原模型，
但天空偏亮和右侧远景色块仍在；00024的文字与主体轮廓仍保留；
00014、00031和00035的远景/植被差异尚存；00033的底盘、踏板仍模糊偏暗，
恢复没有消除原始baseline的地面拉伸。这些是定性观察，不是对各区域误差来源的定量定位。

下一步不根据现有测试分数不断增加恢复步数或调学习率。先固定50%预算、0步恢复，
比较两种选点规则，单独检验“是否选错了要保留的高斯”。这一步完成后，再给可见贡献剪枝
加入相同3000步恢复，形成评分×恢复的完整对照；不会直接拿无恢复的新方法与已恢复旧方法宣布优劣。

## 15 第三阶段：可见贡献评分与剪枝

### 为什么可用颜色梯度算贡献

固定几何、opacity和相机时，像素颜色对每个高斯的预计算RGB颜色是线性的：

    I_c(p) = sum_i [w_i(p) * color_i,c] + T_end(p) * background_c
    w_i(p) = T_i(p) * alpha_i(p)
    d I_c(p) / d color_i,c = w_i(p)

令标量L为渲染图所有像素和RGB通道的均值，则把某个高斯RGB三个梯度相加：

    sum_c dL / d color_i,c = (1 / HW) * sum_p w_i(p)
    score_i = 对所有训练视角的上述值取平均

这正是第4节预先定义的平均可见混合权重（受原渲染器自身的裁剪、alpha阈值与提前终止规则限制）。
它不包含颜色误差、GT、体积因子或测试视角，也不是删掉高斯后的真实误差。
不能将它宣称为完整LightGaussian复现。

实现依据是本仓库固定版本的`cuda_rasterizer/backward.cu`：
`dchannel_dcolor = alpha * T`，然后累积到`dL_dcolors`。
`gaussian_renderer.render`已有`override_color`接口，可以获得这些梯度。
脚本冻结原模型参数，不建优化器，只对单独的临时RGB变量求梯度，原PLY绝不改写。

评分时所有临时颜色设为0.5、背景设为0，**仅用于计算和校验权重**。
这不是把保存的模型变成灰色，也不改变最终评测的背景或SH颜色。
两个评分验证关卡：

1. 每个视角要求`sum(score_i) ≈ 2 × mean(灰色诊断渲染图)`，权重非负、有限且总和不超过1（允许数值误差）。
2. 首个视角对固定每17个高斯抽一个的子集增加0.01颜色，比较实际图像均值变化与梯度预测。

失败即停止，不带着未经验证的评分去评测。GPU内部梯度为float32原子累加，
跨视角在CPU上用float64累加；不承诺每次评分逐位确定。相同分数按原始行号打破并列，
选中后恢复原行顺序，完整保留62个字段。评分不读取GT像素；
沿用的输入完整性检查仍会读取已有参考图片的文件哈希，这不参与评分。

### 三组实验与验收

| 名称 | 保留数 | 评分 | 恢复步数 |
| --- | --- | --- | --- |
| identity100 | 1,082,124 | 全保留 | 0 |
| opacity50 | 541,062 | 原始opacity排序 | 0 |
| contribution50 | 541,062 | 训练视角平均可见权重 | 0 |

三组都从原30k模型派生，而非从第二阶段恢复后的参数派生。
正式评分遍历全部263训练视角，视角等权、每视角按像素数归一化。
准备时记录与opacity50保留集合的重叠比例；这能说明选点是否不同，但重叠低不自动代表更好。
评测时重新计时三组，旧两组PNG必须与第一阶段完全一致，新组也必须匹配GT及视角数量。
选择评分的耗时独立记录，不计入推理FPS；需在最终报告另外披露。

第三阶段不实现贡献剪枝后的恢复训练，目的是先隔离评分的效果；第四阶段实现见第17节。
若新评分有效或暴露有解释价值的取舍，再在固定3k恢复预算下验证；不预先保证新评分一定更优。
平均贡献可能偏向覆盖大区域的高斯，仍可能损失细线或少数视角才出现的结构。
后续两个场景、多预算及验证集约束不变。

### 运行步骤

登录节点仓库根目录：

```bash
git pull --ff-only
conda activate cg3dgs
python -m unittest discover -s tests -v
```

GPU节点沿用cg3dgs、CUDA_HOME与TORCH_HOME设置。无需重装任何扩展。
保留原数据、baseline和第一阶段`outputs/compression/train_phase1_v2`；
默认parent指向该目录，不是`train_recovery_3k`。

先用2个训练视角验证评分、选择和评测链路：

```bash
python scripts/contribution_phase3.py score \
  --experiment outputs/compression/train_contribution_smoke2 --views 2
python scripts/contribution_phase3.py prepare \
  --experiment outputs/compression/train_contribution_smoke2
python scripts/contribution_phase3.py evaluate \
  --experiment outputs/compression/train_contribution_smoke2
```

2视角只用于冒烟，协议`formal: false`，不评价其分数好坏。成功后从新目录完成正式实验：

```bash
python scripts/contribution_phase3.py score
python scripts/contribution_phase3.py prepare
python scripts/contribution_phase3.py evaluate
python scripts/contribution_phase3.py export --name train-contribution50-first
```

默认目录`outputs/compression/train_contribution50`，`--views 0`表示全部训练视角。
每条成功后再执行下一条；评分每完成一个视角原子保存pickle-free断点，
中断后重跑完全相同的score命令继续。已完整评分会验证后跳过。
prepare拒绝覆盖，若准备中断需新目录；evaluate可重跑并备份旧指标。
不要同时对同一目录运行多份命令，不要在运行中更新代码或手改哈希。

导出包含三组逐视角指标、计时、六组对照图、评分校验记录、选点重叠和协议；
评分数组、索引、PLY仍留在outputs，不上传Git。评分记录可能包含训练图片文件名。
在联网登录节点上传：

```bash
git add results/train-contribution50-first
git commit -m "Add visible contribution pruning results"
git push origin main
```

第三阶段交付时本地验证：27项CPU测试（含线性混合的梯度公式、合法输入、训练视角调用约束、
选点字段不变、三组模拟评测导出及篡改拦截）与静态检查。
现已收到云端正式结果，数值关卡核查见第16节。本地仍未执行GPU模型。

## 16 第三阶段正式结果与分析

来源：[train-contribution50-first](../results/train-contribution50-first/summary.csv)，
结果提交`2d43d1e`。正式评分遍历263个训练视角，模型来源仍为原30k baseline。
测试38视角，RTX 4090 D，980×545，SH3；三组均没有恢复训练。

| 指标 | identity100 | opacity50 | contribution50 |
| --- | --- | --- | --- |
| 高斯数量 | 1,082,124 | 541,062 | 541,062 |
| PLY大小 MiB | 255.936 | 127.969 | 127.969 |
| PSNR ↑ | 21.810524 | 19.424191 | 21.595947 |
| SSIM ↑ | 0.812434 | 0.774688 | 0.803524 |
| LPIPS ↓ | 0.208012 | 0.232863 | 0.217566 |
| 平均渲染 ms | 4.974824 | 3.046757 | 3.413233 |
| FPS | 201.012 | 328.218 | 292.977 |
| 推理PyTorch已分配峰值 bytes | 834,924,032 | 416,388,608 | 454,330,880 |

相对原模型，可见贡献剪枝PSNR下降0.214577 dB，已在本场景、本预算、无恢复条件下
达到此前“不超过约0.5 dB”的探索目标；SSIM仍下降0.008910、LPIPS仍上升0.009554，
所以不能宣称无损。相对opacity剪枝，PSNR高2.171757 dB，三项汇总画质指标均更好。
这支持“选哪些高斯保留很重要”，不是单靠额外训练的收益。

代价：contribution50渲染比opacity50多耗时约12.03%，但仍比原模型快约1.458倍。
两者保留数量、字段与精度完全相同，模型字节数相同；因此高斯数不能单独预测渲染速度。
覆盖面积、tile分配、遮挡与混合工作量可能造成差异；本轮未记录tile计数，
不能把具体瓶颈当成已验证事实。两种选择的交集为413,172个，占各自保留集合的76.36%。

评分校验：263条训练视角记录连续完整，权重总和与诊断覆盖率最大差为
4.7684e-6；首视角有限差分为0.0773847103，梯度预测0.0773852691，误差约5.59e-7。
评分计时报告约3.12秒，排除每视角断点写入、加载等开销，不是端到端耗时。
本地复核报告内数值，没有重新在GPU计算梯度或读取完整评分数组。

评测核查：三组各38视角均值及各114个计时样本一致；上传的24张图片
（6张GT+18张渲染）与哈希记录一致；18张渲染的PSNR独立重算与记录差异小于0.0001 dB。
没有本地重算SSIM／LPIPS或完整PLY。查看全部六张contribution50诊断图：
车头、文字和大面积天空外观比之前opacity50更接近未剪枝图；
00024底盘细节仍模糊，00033的踏板、地面拉伸问题仍在。
该方法主要降低额外剪枝损伤，并未解决原baseline的重建缺陷。

不要把本轮未恢复的21.595947直接与上一轮opacity恢复后的21.175407作唯一结论。
下一轮同时比较恢复前后，并重新统一计时，补齐“评分方法×恢复训练”实验矩阵。

## 17 第四阶段：六组评分与恢复消融

### 设计与已实现范围

本轮只改变是否恢复，不改变评分公式、50%预算、学习率、种子或数据划分：

| 组别 | 初始模型（第三阶段） | 新增优化 |
| --- | --- | --- |
| identity100 | 全高斯 | 0步 |
| opacity50 | opacity选点 | 0步 |
| contribution50 | 可见贡献选点 | 0步 |
| identity100_recovery | 全高斯 | 3,000步 |
| opacity50_recovery | opacity选点 | 3,000步 |
| contribution50_recovery | 可见贡献选点 | 3,000步 |

三个恢复组分别新建Adam，使用第13节完全相同的训练算法、学习率与损失；
随机种子0，训练相机序列一致，关闭增密、裁剪和opacity reset。
从第三阶段的未恢复PLY开始，**不读取第二阶段的最终PLY或优化器断点**，
所以正式总预算不会变成额外6000步。此前两个恢复组重新训练，
用于在同一协议下补齐三组；GPU浮点计算仍不保证与历史结果逐位相同。

新入口为`recovery_phase4.py`和`recovery_train_phase4.py`。
为保留旧实验脚本哈希，不改动第一至第三阶段源文件。第四阶段训练代码沿用第二阶段逻辑，
仅更换实验协议接口；CPU回归测试比较函数AST，防止副本之间训练算法悄悄分叉。
旧源文件继续是历史实验可追溯的依据。32项CPU测试与静态检查已通过，
包括六组模拟评测导出、父实验必须正式且完成、配置/产物篡改拦截及训练逻辑一致性；
当时新入口尚未运行GPU；现已收到正式云端结果，见第18节。本地始终未运行GPU模型。

### 运行命令

保留完整第三阶段`outputs/compression/train_contribution50`，以及它依赖的第一阶段
和baseline数据。第四阶段拒绝以2视角冒烟评分结果作为父实验。
不需要重新评分，不需要安装依赖、编译CUDA或重训baseline。
建议额外预留至少15 GiB磁盘空间，实际占用随图片和断点而变。

登录节点：

```bash
git pull --ff-only
conda activate cg3dgs
python -m unittest discover -s tests -v
```

GPU节点仓库根目录：

```bash
conda activate cg3dgs
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_HOME="$PWD/data/torch-cache"

python scripts/recovery_phase4.py prepare \
  --experiment outputs/compression/train_scoring_recovery_smoke20 --steps 20
python scripts/recovery_phase4.py train \
  --experiment outputs/compression/train_scoring_recovery_smoke20
python scripts/recovery_phase4.py evaluate \
  --experiment outputs/compression/train_scoring_recovery_smoke20
```

20步验证新入口的三组训练和六组评测链路，不要求画质提升。逐条成功后，再跑正式：

```bash
python scripts/recovery_phase4.py prepare
python scripts/recovery_phase4.py train
python scripts/recovery_phase4.py evaluate
python scripts/recovery_phase4.py export --name train-scoring-recovery3k-first
```

默认parent为`outputs/compression/train_contribution50`，正式输出为
`outputs/compression/train_scoring_recovery_3k`。若自定义parent，在prepare时传`--parent`；
若自定义输出，每一步传相同`--experiment`。

训练顺序为三个恢复组，评测重新渲染并计时全部六组。
三组未训练对照必须与第三阶段PNG逐像素相同，三个恢复组必须具有相同训练视角序列。
每500步及最后一步保存可信本地断点，重复train可继续中断任务并跳过已完成的组；
不要同时对同一目录运行多份脚本。prepare不覆盖已有目录；
更新代码后不手改哈希，按新目录重建实验。

导出包含六组指标与逐视角分数、36张固定视角渲染、6张GT、三组训练记录与协议，
不会上传PLY或checkpoint。成功后在联网登录节点：

```bash
git add results/train-scoring-recovery3k-first
git commit -m "Add scoring by recovery six-arm results"
git push origin main
```

### 返回结果后如何决策

1. 看contribution50恢复前后，量化恢复的增益或退化，不能预设一定提高。
2. 看contribution50_recovery与opacity50_recovery，同样数量和优化预算比较选点。
3. 同时与identity100及identity100_recovery比较，分别报告对原baseline和同预算对照的损失。
4. 同时报告PSNR、SSIM、LPIPS、模型大小、推理显存口径、速度及评分/训练开销，
   不因某一项好就忽略其余项目。
5. 冻结本轮方案后，下一阶段优先在truck复现实验，并扩展70%/50%/30%预算，
   加入独立重复计时。不要围绕已经看过的train测试图反复调评分权重或恢复步数。

本轮不开发新评分、不调参、不扩展预算和场景；先把当前核心方法的公平对照补齐。
跨场景与多预算仍是最终报告必需的后续证据，单场景成功不等于已完成整个大作业。

## 18 第四阶段结果：核心六组消融完成

来源：[train-scoring-recovery3k-first](../results/train-scoring-recovery3k-first/summary.csv)，
结果提交`d433184`，执行代码`113c856`。RTX 4090 D，980×545，
263训练／38测试视角，seed=0，50%保留预算；恢复组三组均为3000步。

| 模型 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 平均ms | FPS |
| --- | --- | --- | --- | --- | --- |
| identity100 | 21.810524 | 0.812434 | 0.208012 | 4.968767 | 201.257 |
| opacity50 | 19.424191 | 0.774688 | 0.232863 | 3.043336 | 328.587 |
| contribution50 | 21.595947 | 0.803524 | 0.217566 | 3.428053 | 291.711 |
| identity100_recovery | 22.139402 | 0.818250 | 0.203952 | 5.010918 | 199.564 |
| opacity50_recovery | 21.175035 | 0.802914 | 0.221105 | 3.084995 | 324.150 |
| contribution50_recovery | 22.086800 | 0.814473 | 0.210707 | 3.447501 | 290.065 |

主要发现：

1. 贡献剪枝恢复后比恢复前提高0.490852 dB。恢复有帮助，但提升小于opacity组，
   因为它初始剪枝损伤已经较小；这不是对优化难度的严格证明。
2. 同为541,062个高斯、相同3000步预算，贡献组比opacity组高0.911764 dB，
   SSIM更高、LPIPS更低。选点方法的优势在恢复后仍然存在。
3. 相对同预算未剪枝组，贡献恢复组PSNR仅低0.052603 dB，
   模型约127.97 MiB而非255.94 MiB，渲染约快1.4535倍。
   但SSIM低约0.003777，LPIPS高0.006755，不能写成无损或所有指标完全一致。
4. 相对原30k baseline，贡献恢复组PSNR高0.276276 dB、SSIM高0.002038，
   但LPIPS仍高0.002695。这说明“超过原baseline”必须说明具体指标和额外优化预算。

本地核查六组各38视角的三项均值、各114次计时；三组各3000条训练记录的步数、
有限损失、数量与相机序列；42张上传图片哈希与36张渲染PSNR独立重算均通过。
模型完整PLY、SSIM／LPIPS和GPU训练未在本地重新执行。
查看全部六张贡献恢复诊断图：车头与文字保持较好，视觉上接近未剪枝结果，
但00024底盘纹理和00033踏板依然模糊，近地碎石拉伸仍在。
不能用压缩成功掩盖baseline本身的重建缺陷。

当前可以支持的结论是“在train场景、50%预算下，贡献评分加固定数量恢复具有较好的质量—大小—速度折中”。
还不能据此推广到其他场景或更激进预算。下一步优先保持方案不变复现truck，
再做70%／50%／30%曲线，而不是继续围绕train测试图调整评分或学习率。

## 19 下一轮：truck跨场景固定协议复现

### 冻结哪些条件，允许哪些自然差异

新增`configs/truck.json`：官方30k训练，seed=0，留出每8张相机中的一张，
原图配置`resolution=-1`、SH3、黑背景。贡献评分使用全部训练视角，
保留比例50%，恢复3000步，学习率与损失完全沿用第二/四阶段，不针对truck调参。
先跑同样的全保留关卡，再比较两种剪枝，最后形成完整六组。

新场景的相机数量、原始图片大小和高斯数自然不同，不应强行填成train的
263／38、980×545或1,082,124。`-1`也会遵循原版对大图的自动缩放规则。
视角数量、实际分辨率与最终数量由数据验证和benchmark记录；
跨场景不直接比较绝对FPS来宣称某方法加速，主要比较每个场景内同口径比值。
truck也属于当前数据包中的另一个场景，不是另一个完全独立数据集。

新增`scripts/truck_suite.py`只负责调度、校验和跳过已完成阶段。
底层沿用已验证的`run.py`、`compression_phase1.py`、
`contribution_phase3.py`、`recovery_phase4.py`，未修改历史实验算法或脚本哈希。
这是一轮跨场景复现实验，不是新算法开发。

### 数据和环境检查

默认数据路径为`data/tandt/truck`。先确认此前下载的数据包中是否已解压该目录，
不要因默认路径不存在就重复下载train数据。数据位于别处时，
只修改`configs/truck.json`的`source`；其他训练与分辨率参数受固定协议检查保护。
运行开始后不要再修改配置。

在登录节点、仓库根目录执行：

```bash
git pull --ff-only
conda activate cg3dgs
python -m unittest discover -s tests -v
python scripts/truck_suite.py plan
python scripts/truck_suite.py check
```

plan只打印命令，不训练、不生成实验目录。check校验COLMAP相机、图片和初始点云，
打印实际训练／测试数量及独立输出目录，不需要GPU。check会读取图片计算哈希，
若集群限制登录节点I/O，请放到获批计算节点。数据检查失败时先处理路径或缺文件。

### GPU执行：可逐阶段，也可一条命令串行

在有效GPU分配内、仓库根目录：

```bash
conda activate cg3dgs
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_HOME="$PWD/data/torch-cache"
python scripts/doctor.py

python scripts/truck_suite.py baseline
python scripts/truck_suite.py opacity
python scripts/truck_suite.py contribution
python scripts/truck_suite.py recovery
```

逐条成功后继续，或用以下命令串行执行相同四阶段：

```bash
python scripts/truck_suite.py all
```

all不自动导出或推送。它不会在本地运行，不会并行抢占多个GPU。
本轮复用已云端验证的训练/评分核心，不再重复增加20步冒烟实验；
仍保留环境检查、baseline完整性、100%像素一致性、贡献评分数值关卡以及六组评测检查。

| 阶段 | 独立输出 |
| --- | --- |
| baseline | outputs/truck_baseline_30k |
| opacity | outputs/compression/truck_phase1 |
| contribution | outputs/compression/truck_contribution50 |
| recovery | outputs/compression/truck_scoring_recovery_3k |

baseline必须为新场景重新训练，不复用train场景的高斯参数。
后续三个恢复组都从truck的30k派生模型重新建立优化器。
不要对同一输出并发运行all和单阶段命令，不要在运行中拉取新代码。
预留数十GiB磁盘余量；具体取决于truck实际模型和视角数量，不能根据train结果保证上限。

### 中断与重跑规则

- 已完成阶段只有通过来源、指标及相关产物检查后才会跳过，不因“目录存在”就认为成功。
- 评分沿用逐视角断点，恢复训练沿用每500步断点；重复相同阶段或all即可继续。
- baseline训练断点与恢复训练断点格式不同，不能混用。baseline部分完成时，
  入口要求你明确指定可信的原版checkpoint，不自动选择“最新文件”：

```bash
# 仅当此文件确实存在且来自本次truck baseline时使用
python scripts/truck_suite.py baseline \
  --checkpoint outputs/truck_baseline_30k/chkpnt15000.pth
```

baseline恢复沿用原版行为，不保证随机状态精确复现。若尚未生成任何checkpoint，
保留失败目录并先排查；脚本不会自动删除或覆盖已有运行。
最终PLY已存在但评测未完成时，重新执行baseline仅补渲染、指标和计时，不重训。
若某上游阶段不完整但已经有下游实验，入口拒绝重写上游，避免使下游来源失效。
此时提交错误信息排查，不手改哈希或强制覆盖。

### 导出与提交

六组都完成后：

```bash
python scripts/truck_suite.py export
```

默认导出`results/truck-scoring-recovery3k-first`。包含六组结果、训练记录、
最多六个固定测试视角的渲染和GT、评分校验记录及`cross_scene.json`，
后者记录truck实际划分、原版版本、原baseline计时、选点重叠和冻结协议。
视角不足时只导出有效编号。PLY、断点、完整数据不进入Git。
若导出目录已存在或导出中断，用新的`--name`保留旧目录，不覆盖：

```bash
python scripts/truck_suite.py export --name truck-scoring-recovery3k-second
```

成功后在联网登录节点提交实际导出的目录：

```bash
git add results/truck-scoring-recovery3k-first
git commit -m "Add frozen truck cross-scene replication"
git push origin main
```

### 验收与之后的实验

先核查新场景完整性、像素一致性、评分关卡和同预算训练记录，再比较六组。
无论truck支持还是不支持train结论，都完整记录，不先调参数把分数“修好”。
下一轮再以冻结评分与恢复设置跑两个场景的70%／50%／30%预算曲线，
配合独立重复计时和失败案例。多预算代码本轮未实现，不能把未来计划写成已完成。

本地38项CPU测试与静态检查通过，包括冻结配置、独立路径、命令顺序、
已完成阶段跳过、部分恢复续跑、baseline显式断点要求和保护下游来源。
truck数据与GPU实验尚未在本地执行，也尚无云端结果；实际视角数和指标暂不填写。

## 20 truck跨场景实测结果（上传提交0af71e9）

本节更新第19节当时的“待执行”状态。结果目录为
`results/truck-scoring-recovery3k-first`。实际219个训练视角、32个测试视角，
分辨率979×546，RTX 4090 D。原始2,590,158个高斯，50%组1,295,079个；
PLY分别642,360,716和321,181,124字节。

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | 毫秒/帧↓ | FPS↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| 完整模型 | 25.417843 | 0.882305 | 0.146760 | 6.2613 | 159.711 |
| 不透明度50% | 24.896971 | 0.877248 | 0.149610 | 4.0677 | 245.838 |
| 贡献度50% | 25.374870 | 0.881424 | 0.147770 | 4.4546 | 224.486 |
| 完整模型+恢复3k | 25.561270 | 0.884788 | 0.143576 | 6.2941 | 158.879 |
| 不透明度50%+恢复3k | 25.524910 | 0.883636 | 0.145088 | 4.0715 | 245.612 |
| 贡献度50%+恢复3k | 25.552813 | 0.884649 | 0.143931 | 4.4799 | 223.217 |

### 怎样理解

贡献恢复组与“也多训练3000步”的完整模型相比，PSNR低0.008457 dB、
LPIPS高0.000355，PLY约减半，渲染约快1.405倍。
这支持“50%压缩在第二个场景也能保持接近完整模型的质量”，
不是证明没有任何损失，也不是证明压缩提升了重建质量。

与不透明度恢复组相比，贡献度只高0.027903 dB，差距很小；
不透明度组FPS还高约10%。不能把train上的明显优势推广为所有场景都明显更好，
也不能忽略速度代价。高斯数量相同不代表渲染工作量相同：位置、投影覆盖、
重叠和可见性都会影响光栅化耗时。当前单次训练与计时不能证明小差异的统计显著性。

查看上传的00014后侧视角，两恢复组的车身、木板、栏杆和建筑总体接近，
未见明显结构性破坏；细小背景、树叶与地面纹理仍应结合GT比较。
00031侧面视角与GT比较，车身主体保持良好，但地面细纹明显更平滑，
远处路牌与树叶细节仍有损失；这些是重建结果的残余误差，不能都归因于剪枝。
固定诊断图不是随机采样的画质证明，不能代替全32视角指标。

### 核查范围

已核对六组32视角指标均值、各96次计时样本、28张上传图片哈希、
24张渲染的像素PSNR，以及三组连续3000步、固定数量、同相机序列的训练记录。
全219训练视角评分记录及其质量守恒/覆盖检查通过，评分文件与跨场景协议哈希对应。
本地没有重跑GPU训练、SSIM/LPIPS或云端完整PLY；不能将此检查写成独立复现实验。

## 21 下一轮：冻结方法，补齐70%与30%保留率

### 问题和控制变量

50%只回答“删一半能否工作”。70%检验较温和压缩，30%检验更激进压缩。
两场景都做，不能只挑更好看的场景或预算。
现有50%正式结果原样保留，最终形成70%/50%/30%的质量—大小—速度表。
这里是**保留率**：30%表示删掉70%，不是删掉30%。

每个新增预算仍有六组：完整、不透明度剪枝、贡献度剪枝，
以及这三者各恢复3000步。共四个任务、24组评测、12组恢复训练。
每次都重新测同任务的完整模型计时对照；不跨任务直接拿FPS比值当可靠加速比。
这不是四个独立随机种子的统计重复，独立计时重复及必要的多种子验证留待后续。

- 始终从原始30k完整PLY直接选取，不从50%模型再次剪枝，不从恢复后的模型剪枝。
- 贡献评分直接复用所有训练视角得到的原始完整评分，不引入测试图像或新超参数。
- 不透明度与贡献度使用同样数量：`max(1, floor(N * keep))`；
  分数相同时按原始索引打破平局，保存时保持原始行序和全部62个字段。
- 三个恢复组同seed=0、同相机序列、同学习率、fresh Adam、固定高斯数量、SH3，
  损失仍为0.8 L1 + 0.2 DSSIM。不新增增密、透明度重置或测试集调参。
- 新脚本为`budget_phase5.py`和`recovery_train_phase5.py`，历史脚本不改，
  避免破坏已有实验的代码哈希。新目录中组名不再含“50”，保留率写入协议及汇总表。

预算不同的排名可能不一样。若30%画质明显恶化，也要原样记录；
下一步讨论压缩极限或改进方法，而不是改测试视角或悄悄重调参数。

### 先做20步冒烟（GPU节点）

保持已工作的cg3dgs环境和缓存配置；不需要联网安装、重新评分或重训baseline。
联网节点先`git pull --ff-only`，GPU节点再执行：

```bash
conda activate cg3dgs
export TORCH_HOME="$PWD/data/torch-cache"
python scripts/budget_phase5.py run --scene train --keep 0.3 --steps 20 \
  --experiment outputs/compression/train_budget30_smoke20
```

`run`依次准备、训练、渲染、评测和计时，不自动导出。冒烟会评测六组，
不只是运行20次迭代；它验证管线，不能用于正式结论。
非3000步必须显式指定独立目录，以防与正式实验混用。

### 正式运行及导出

先确认冒烟成功、GPU作业时限和scratch磁盘余量充足。每个任务保存六份初始模型、
三份最终模型及恢复断点，四个任务可能占用数十GB；不会自动删除旧结果。
在调度器分配的GPU节点执行，不在登录节点训练：

```bash
(
set -e
for scene in train truck; do
  for keep in 0.7 0.3; do
    case "$keep" in
      0.7) budget=70 ;;
      0.3) budget=30 ;;
    esac
    python scripts/budget_phase5.py run --scene "$scene" --keep "$keep"
    python scripts/budget_phase5.py export --scene "$scene" --keep "$keep" \
      --name "${scene}-budget${budget}-recovery3k-first"
  done
done
)
```

括号内遇错立即停止；不要在失败后仍提交不完整结果。可分别运行四个任务，
但计时时避免其他GPU任务干扰，不建议同一张卡同时运行多组。

### 中断与重新运行

训练完成且哈希验证通过的组会跳过，未完成组使用已有500步间隔断点恢复；
实际跨中断一致性仍需云端验证。重新执行某个任务的`run`会重新评测、计时六组，
不重新训练已验证完成的组。已导出的结果目录不可覆盖，重新导出用新名称。
若`prepare`意外中断尚未生成完整`experiment.json`，不要直接复用残缺目录：
保留排错证据，使用新的`--experiment`路径重新准备。

已完成的任务不要因为其他任务中断而全部重跑；从未完成的scene/keep命令继续即可。
若只需重新评测可用`evaluate`代替`run`；若只需上传则直接`export`。
手工修改原始PLY、评分、代码、配置或索引将触发校验失败，不应删除校验绕过。

### 提交与后续验收

联网登录节点，仅提交成功导出的四个小结果目录：

```bash
git add results/train-budget70-recovery3k-first results/train-budget30-recovery3k-first \
        results/truck-budget70-recovery3k-first results/truck-budget30-recovery3k-first
git commit -m "Add train and truck multi-budget compression results"
git push origin main
```

导出包含六组指标、逐视角指标、计时样本、固定诊断图、协议和三组训练记录，
不包含模型/缓存。后续检查实际数量、全视角指标均值、训练来源、图片PSNR和哈希，
再将新数据与既有50%结果合并。重点报告相对**同预算恢复完整模型**的质量损失、
同任务加速比、PLY大小及失败区域，不只展示最好的单个分数。

本轮本地44项CPU单元测试、29个Python文件静态解析及`git diff --check`通过。
新增检查包括预算合法性、嵌套选择、62字段逐项保存、三组初始化对应、
索引篡改拒绝、模拟六组评测导出，以及恢复训练函数与已验证版本的AST一致性。
这些检查不等于新增预算已完成GPU验证。
