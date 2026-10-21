# GitHub 与云端结果同步

公开仓库：https://github.com/firefly1145141919810/cg-3dgs-baseline

## 代码同步

任何人均可通过HTTPS克隆。直接推送仍需要写权限；合作同学可由仓库所有者邀请为Collaborator，或通过fork和PR贡献。不要共享密码或SSH私钥。云端推送使用自己的SSH key或受限凭据，私钥不进入Git。

```bash
git pull --ff-only
python scripts/bootstrap.py
```

如果本地有修改导致 pull 失败，先 `git status`，保存并提交自己的修改，不要用 reset --hard。首次打开仓库后不要把 vendor 子模块切到最新 main；它是我们的固定算法版本。

## 实验结果分两类

**小文件通过 GitHub 同步：** 指标 JSON/CSV、固定配置、报告、学习文档、小量经过检查的示例图。

```bash
python scripts/export_results.py --config configs/train.json --name train-baseline-first
git status
git add results/train-baseline-first
git commit -m "Record measured train baseline results"
git push
```

导出器只接收真正存在的指标与PLY，不生成模拟结果。运行标签必须唯一，防止覆盖历史实验。不能假设所有导出器都已自动脱敏：原始重复计时协议记录模型绝对路径，样本记录节点名，须使用 `publish_timing.py` 生成公开副本，具体命令见README。其他导出也应检查后再提交。

**大文件通过文件传输同步：** 输入照片、Gaussian PLY、checkpoint、全套渲染图与 TensorBoard 日志。它们被 .gitignore 排除，不要用 `git add -f` 绕过。Git仓库不适合无限存储大型模型。

```bash
python scripts/archive_run.py --config configs/train.json
```

得到 `archives/train_baseline_30k.tar.gz` 和对应 `.sha256`。脚本仅本地打包，不会自动上传。用服务器提供的文件管理器、团队对象存储或 scp 获取，例如在本地终端执行：

```text
scp 用户名@服务器地址:/云端仓库绝对路径/archives/train_baseline_30k.tar.gz 本地保存目录
```

把示例中的用户名、地址、路径替换为实际值。共享集群也可能需要跳板机/指定端口，按平台说明操作。下载后验证 SHA256：Linux 使用 `sha256sum`，Windows PowerShell 使用 `Get-FileHash -Algorithm SHA256`。

实验 archive 不包含输入数据；另保留数据来源、下载哈希和清单。若把归档放在团队对象存储，可在结果记录中写不含凭据的对象路径；不要提交带短期签名 token 的下载链接。

## 保留可复现记录

需要备份：代码 commit、官方子模块 commit、环境版本、配置、数据划分与哈希、checkpoint、最终模型、指标和对照图片。GPU 开始运行后再改代码会破坏版本对应，建议每次实验从干净 commit 启动。

仓库公开不等于第三方材料获得了新的授权。上游代码、论文、数据与图像保留各自权利和使用条件；不要把它们统一宣称为本项目自由授权的原创内容。

## 公开前检查与历史边界

运行 `python scripts/check_public_repo.py` 检查已跟踪文件中的常见凭据、机器路径和个人作业脚本，再人工检查 `git diff --cached`。扫描器不会打印命中的秘密值，也不能保证发现所有敏感信息。

本次已从当前版本移除个人Slurm脚本，并将计时记录替换为脱敏公开副本。原始计时证据仅保存在本地忽略目录 `private/`；公开副本附 `public_redaction.json`，记录原始/公开文件SHA256及变更类型，不改变任何数值。它不是原始文件的逐字节副本。

旧克隆拉取删除提交时可能删除原来已跟踪的Slurm脚本。**在云端首次拉取此次清理前**，先把个人作业文件和原始计时结果复制到 `private/` 或其他私下备份位置；不要仅依靠 `.gitignore` 保护已有的已跟踪文件。之后按需要从备份恢复脚本，它们将保持不受Git跟踪。

后续已按授权重写公开main历史，清除历史Slurm及目标云端信息。**旧克隆不要直接pull或合并回新main**，本地/云端迁移及旧提交映射见[历史迁移说明](HISTORY_MIGRATION.md)。fork、他人克隆和GitHub缓存不一定随分支更新而清除；如曾泄露有效凭据，应先撤销/轮换，不能只删历史。
