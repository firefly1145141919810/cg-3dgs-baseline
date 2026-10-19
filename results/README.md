# 实验结果

此目录只保存通过 `scripts/export_results.py` 导出的真实小型结果记录，适合提交到 GitHub。

当前没有云端训练结果，不提供虚构的 PSNR、SSIM、LPIPS 或 FPS。首次云端验证完成后再新增以日期、场景、硬件命名的结果目录。

原始图片、checkpoint、Gaussian PLY、TensorBoard 日志留在 `outputs/`，通过 `archive_run.py` 打包，再用 scp、对象存储或团队共享盘传输；不要用 `git add -f` 把它们提交进来。
