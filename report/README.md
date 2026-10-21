# 最终实验图表与技术报告

主文档是 `main.tex`，导出的提交候选PDF为仓库中的 `output/pdf/3dgs_final_report.pdf`。
本文不是“已经全部验收完成”的声明：实验正文与图表已整理，姓名、学号、实际分工和5–8分钟演示仍需确认。

## 文件分工

- `figures/`：7张图各提供矢量PDF与PNG。三张定量总图、四张固定视角的全图/局部对比。
- `generated/`：6张数值表的LaTeX源、审计JSON（实际有两场景质量表、两场景计时表、核心结论表、开销表，共6张）。不要手改数值，修改生成脚本后重建。
- `main.tex`：中文正文、公式、来源与贡献边界、局限和提交前清单。
- `requirements.txt`：仅用于可选的独立报告环境，不得直接覆盖云端cg3dgs依赖。

## 重建图表

仓库根目录、独立Python 3.12环境：

```bash
python -m pip install -r report/requirements.txt
python scripts/build_report_assets.py
```

脚本固定读取六个质量实验和 `results/timing-repeat-job4380448`，先校验来源与数据，
再重建图表与表格；失败会停止，不会修改原实验。所有读取的输入哈希写入 `generated/audit.json`。
未来使用新结果时需显式修改数据目录与相应来源说明，不自动选择最新/最好的一次实验。
本流程不运行GPU、不访问云端完整PLY。

## 编辑及编译

在 `main.tex` 中搜索“待填写”，填写两人的信息及真实分工，不能虚构完成比例。
编辑器可直接打开本项目文件夹预览。命令行使用XeLaTeX编译两次：

```bash
cd report
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

产生的 `report/main.pdf` 是本次重编译结果；将核查过的版本另存/复制为 `output/pdf/3dgs_final_report.pdf`。
中文字体优先使用可用的SimSun/SimHei并嵌入；其他系统退回Fandol。
本机MiKTeX的Fandol PDF曾出现显示正常但缺少中文ToUnicode映射的问题，故最终导出使用TrueType中文字体，
同时检查了中文文字可提取。换机器编译后也应检查字体嵌入、文字复制和所有页面。

## 解读图表时注意

所有最终性能来自独立复测，而不是旧单轮benchmark。误差棒是三进程均值的最小/最大范围，非置信区间。
质量是单种子全测试视角均值；50%工作点推荐为事后工程结论，完整预算曲线必须一并保留。
局部图坐标在脚本中固定，图像只用于对齐展示与裁剪，不做颜色修改或去噪。
评分耗时在三预算之间共享，开销表重复列出便于阅读，不应把它累加三次。

## 仍需完成

1. 成员核对姓名、学号、实际分工，并能够解释关键公式与源码。
2. 按课程要求制作5–8分钟演示视频或准备现场Demo，包含一个失败案例。
3. 核对最终提交清单，确认数据/模型是否需要单独交付；本次没有自动打包大模型或提交到课程系统。
