# 来源与贡献边界

核心算法完整采用 Inria / MPII 作者的 gaussian-splatting 官方实现，固定在 `2eee0e26d2d5fd00ec462df47752223952f6bf4e`，未把他人的渲染器、优化器或 CUDA 内核冒充为自己的实现。

- 上游：`vendor/gaussian-splatting`，版权与使用条件见该目录的 `LICENSE.md`。本仓库以 Git submodule 保留原始提交、来源及许可证。
- 本项目新增：云端安装与诊断、固定实验配置、训练视角可见贡献评分、稳定Top-K剪枝、固定数量恢复、六组消融、多预算实验、独立重复计时、来源核验、导出与实验报告。由 AI 协助编写；两场景三预算的质量实验及最终模型重复计时已在用户云端执行并回传核查。本地未重跑GPU模型。
- 论文：Kerbl, Kopanas, Leimkühler, Drettakis. *3D Gaussian Splatting for Real-Time Radiance Field Rendering*. ACM TOG 42(4), 2023. https://doi.org/10.1145/3592433
- 原论文及图片版权归原作者/出版方。中文批注是学习解释，不是作者认可的译本；保留未改动原文。公开传播前应另核对相应授权。
- 数据版权归各提供方。数据不进入 Git；请按各数据集条款使用。

当前已完成baseline加压缩扩展的实验主线，不宣称复现原论文全部分数或提出全新理论。成员姓名、实际分工与贡献说明必须由组员确认；报告明确披露官方直接使用模块及AI辅助范围。课程要求的演示视频或现场Demo仍需确认完成。
