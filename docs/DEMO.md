# 项目演示

## 无需GPU的结果展示

克隆完整仓库后即可查看下列图像与报告，不需要安装训练环境。此路线展示已完成实验，不模拟实时渲染。

| 顺序 | 展示内容 | 讲解重点 |
| --- | --- | --- |
| 1 | [方法与原理](BASELINE_GUIDE.md) | 高斯表示、投影和alpha合成；baseline如何学习场景 |
| 2 | [质量曲线](../report/figures/quality_curves.png) | 完整、不透明度和贡献度三种策略，各自是否恢复；完整模型也恢复3k，保证公平 |
| 3 | [质量与速度](../report/figures/quality_speed.png) | 50%保留率的折中；30%更快但细节损失扩大 |
| 4 | [train局部对比](../report/figures/visual_train_00033.png) | GT、完整恢复与两种剪枝的同视角比较；地面条纹也是baseline的失败区域 |
| 5 | [truck局部对比](../report/figures/visual_truck_00031.png) | 另一个场景中的主体稳定性、纹理损失与方法差异 |
| 6 | [性能曲线](../report/figures/performance_curves.png) | 三轮独立进程计时；渲染器FPS、PyTorch分配峰值的测量边界 |

约5—8分钟讲解可分配为：原理1分钟、方法与公平对照1分钟、质量/性能2分钟、失败案例1分钟、结论与局限1分钟。完整数据、配置与结论见[实验文档](COMPRESSION_EXPERIMENT.md)。

## 现场运行

准备Linux CUDA环境、对应场景数据与训练输出后，可按[README](../README.md)运行渲染和评测。仅有Git中的小型结果不能重新渲染，需保留匹配的PLY、配置和相机数据。

baseline示例（不重新训练）：

```bash
python scripts/run.py render --config configs/train.json
python scripts/run.py metrics --config configs/train.json
```

输出图片位于 `outputs/train_baseline_30k/test/ours_30000/renders/`，参考图位于相邻 `gt/` 目录。展示时说明这是新视角合成的批量渲染，不将离线PNG播放速度当作渲染FPS。

源码讲解可依次查看 `scripts/contribution_phase3.py` 的训练视角评分、`scripts/prune_ply.py` 的固定数量选择和 `scripts/recovery_train_phase4.py` 的恢复优化；其中CUDA光栅化直接使用官方实现。
