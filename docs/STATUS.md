# 测试与验证

研究设计与结果见[实验文档](COMPRESSION_EXPERIMENT.md)，运行方法见[README](../README.md)。

## 实验验证

- train、truck两场景的官方30k baseline，以及100%保存重载正确性检查。
- 不透明度与训练视角贡献度剪枝、固定数量3k恢复、同预算完整恢复对照。
- 两场景70%/50%/30%预算，共36组最终质量记录、18组恢复训练记录。
- 18个恢复后模型设置各3个独立进程计时，共54份样本；最大相对耗时范围0.536%。
- 数据审计覆盖全测试视角均值、训练记录、210张图片哈希、180张渲染像素PSNR及重复计时统计。

## 自动化检查

```bash
python -m unittest discover -s tests -v
python scripts/static_check.py
python scripts/check_public_repo.py
```

67项CPU测试覆盖数据读取、配置、稳定选点、评分与恢复约束、结果导出、来源校验及发布工具；静态检查覆盖39个Python文件。发布检查用于识别常见凭据与机器特定路径，不构成完整安全审计。

GitHub Actions在PR和推送时执行普通及优化模式的CPU测试、静态检查和已发布实验的完整审计，无需CUDA或场景数据。

报告资产生成器校验实验输入及来源哈希，生成 `report/generated/audit.json`。只核验、不重建图表时运行 `python scripts/build_report_assets.py --audit-only`。PSNR可由导出的渲染与参考图片复算；SSIM与LPIPS采用GPU评测记录。

## 验证边界

- 质量实验为单种子；三轮独立计时不是三个训练种子，不能用于证明微小画质差异具有统计显著性。
- 中断续跑逻辑有CPU测试，但未覆盖所有真实GPU中断情形的逐位一致性。
- 当前性能只代表记录中的硬件与软件配置；其他GPU、CUDA版本和操作系统尚未完成同等验证。
- CPU测试与结果审计不替代GPU训练和渲染验证；中断故障测试验证文件发布和重试逻辑，不模拟所有GPU或存储硬件故障。
