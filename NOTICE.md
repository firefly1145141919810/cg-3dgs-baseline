# 来源与贡献边界

核心算法完整采用 Inria / MPII 作者的 gaussian-splatting 官方实现，固定在 `2eee0e26d2d5fd00ec462df47752223952f6bf4e`，未把他人的渲染器、优化器或 CUDA 内核冒充为自己的实现。

- 上游：`vendor/gaussian-splatting`，版权与使用条件见该目录的 `LICENSE.md`。本仓库以 Git submodule 保留原始提交、来源及许可证。
- 本次新增：云端安装与诊断、固定实验配置、数据核验、运行记录、推理计时、结果导出、中文教学文档及论文学习批注。由 AI 协助编写，尚未在目标云 GPU 验证。
- 论文：Kerbl, Kopanas, Leimkühler, Drettakis. *3D Gaussian Splatting for Real-Time Radiance Field Rendering*. ACM TOG 42(4), 2023. https://doi.org/10.1145/3592433
- 原论文及图片版权归原作者/出版方。中文批注是学习解释，不是作者认可的译本；保留未改动原文。公开传播前应另核对相应授权。
- 数据版权归各提供方。数据不进入 Git；请按各数据集条款使用。

这个阶段只完成 baseline 工程准备，不是大作业的实质扩展，也不是“已复现论文分数”。后续必须增加经确认的扩展、真实对比和个人贡献说明。
