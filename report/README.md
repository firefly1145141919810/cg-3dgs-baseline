# 最终实验图表与技术报告

主文档是 `main.tex`，导出的PDF为仓库中的 `output/pdf/3dgs_final_report.pdf`。

## 文件分工

- `figures/`：7张图各提供矢量PDF与PNG。三张定量总图、四张固定视角的全图/局部对比。
- `generated/`：6张数值表的LaTeX源、审计JSON（实际有两场景质量表、两场景计时表、核心结论表、开销表，共6张）。不要手改数值，修改生成脚本后重建。
- `main.tex`：中文正文、公式、来源与贡献边界、研究局限。
- `requirements.txt`：用于独立的报告构建环境。

## 重建图表

仓库根目录、独立Python 3.12环境：

```bash
python -m pip install -r report/requirements.txt
python scripts/build_report_assets.py
```

脚本固定读取六个质量实验和 `results/timing-repeat-job4380448`，先校验来源与数据，
再重建图表与表格；失败会停止，不会修改原实验。所有读取的输入哈希写入 `generated/audit.json`。
未来使用新结果时需显式修改数据目录与相应来源说明，不自动选择最新/最好的一次实验。
图表重建只读取JSON与PNG，不需要GPU或完整PLY；计时文件的元数据与哈希映射记录在对应目录的 `public_redaction.json` 中。

## 编辑及编译

在 `main.tex` 中搜索“待填写”，填写成员信息及实际分工。
编辑器可直接打开本项目文件夹预览。命令行使用XeLaTeX编译两次：

```bash
cd report
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

产生的 `report/main.pdf` 是本次重编译结果；将核查过的版本另存/复制为 `output/pdf/3dgs_final_report.pdf`。
中文字体优先使用可用的SimSun/SimHei并嵌入；其他系统退回Fandol。
编译后应检查字体嵌入、文字提取和页面布局。

## 解读图表时注意

所有最终性能来自独立复测，而不是旧单轮benchmark。误差棒是三进程均值的最小/最大范围，非置信区间。
质量是单种子全测试视角均值；50%工作点推荐为事后工程结论，完整预算曲线必须一并保留。
局部图坐标在脚本中固定，图像只用于对齐展示与裁剪，不做颜色修改或去噪。
评分耗时在三预算之间共享，开销表重复列出便于阅读，不应把它累加三次。
