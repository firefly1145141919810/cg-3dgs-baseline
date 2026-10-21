# 来源与贡献说明

## 官方实现

高斯表示、baseline训练、密度控制、CUDA渲染器、kNN初始化与质量评测采用Inria / MPII的官方Gaussian Splatting实现，固定提交为 `2eee0e26d2d5fd00ec462df47752223952f6bf4e`。

源码以Git子模块保存在 `vendor/gaussian-splatting`，版权与使用条件见其中的 `LICENSE.md`。

## 项目实现与AI辅助

本项目实现训练视角可见贡献评分、稳定Top-K剪枝、固定数量恢复、六组消融、多预算实验、独立重复计时、来源校验、结果导出，以及配套测试、文档与报告。

开发过程中使用Codex辅助工程代码、测试、资料讲解和报告整理。官方模块与项目新增模块的边界、成员分工见技术报告的贡献说明。

## 论文与数据

- Kerbl, Kopanas, Leimkühler, Drettakis. *3D Gaussian Splatting for Real-Time Radiance Field Rendering*. ACM TOG 42(4), 2023. [DOI](https://doi.org/10.1145/3592433)
- 原论文及图片版权归原作者或出版方。中文批注为学习说明，不是官方译本，未改动论文原文。
- 数据与场景图像版权归各提供方，使用与传播须遵守相应数据集条款。

第三方材料保留各自许可，本仓库不为其授予新的使用权限。
