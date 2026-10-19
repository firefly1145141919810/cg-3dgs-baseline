# GitHub 与云端结果同步

私有仓库：https://github.com/firefly1145141919810/cg-3dgs-baseline

## 代码同步

合作同学需要自己的 GitHub 账号，由仓库所有者在 Settings → Collaborators 邀请。不要共享密码或 SSH 私钥。云端使用自己的 SSH key 或 `gh auth login`。

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

导出器只接收真正存在的指标与 PLY，不生成模拟结果；它避免把服务器绝对路径和环境变量写进提交。运行标签必须唯一，防止覆盖历史实验。

**大文件通过文件传输同步：** 输入照片、Gaussian PLY、checkpoint、全套渲染图与 TensorBoard 日志。它们被 .gitignore 排除，不能用 `git add -f` 绕过。私有 Git 仓库也不适合无限存储大型模型。

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

第一阶段不需要把结果发布成 GitHub Release，更不需要公开数据。将来确需公开或使用 LFS/Release 时，再核对许可、体积和访问权限。
