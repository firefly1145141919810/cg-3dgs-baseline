# Git历史清理后的同步方式

## 清理范围与证据保留

2026-10-21对唯一公开分支main执行历史重写，清除全部历史Slurm脚本、个人云端工作路径与节点信息。历史中含节点名的报告使用已核验的脱敏报告替代。未改动训练算法或实验数值。

重写前后核对29条提交映射、5431次结果文件对应关系：图像及其他非元数据结果字节未变；56对计时JSON仅路径、主机别名和相关文件校验值变化。清理后全历史扫描包含文本与PDF可提取文本，无目标信息命中，Slurm历史路径为空。清理操作完成时，最新文件树与重写前完全一致；本说明及同步文档随后作为正常新提交加入。

`HISTORY_COMMIT_MAP.json`用于将报告、协议里记录的旧代码版本追溯到新历史。值为null表示该提交在删除Slurm后变为空提交，不再独立保留。上游子模块提交未重写。文件SHA256与Git提交ID是两套不同的标识，不能互相替换。

完整旧历史bundle和原始计时记录仅保留在本机被忽略的private目录。它们仍含旧信息，**不得上传、加入公开分支或作为恢复远程历史的来源**。

GitHub分支已重写不代表他人fork、克隆、缓存或旧提交直链立即消失。若仍能通过旧SHA访问，需联系GitHub Support请求清理缓存/不可达对象；本项目不能撤回他人的副本。此项不等于发现凭据泄露，若另发现有效凭据仍应先轮换。

## 本台Windows电脑

当前项目已经原地切换到清理后的main，并保留工作树和本地运行文件，不需要重新下载数据、模型或重新安装环境。后续正常 `git pull --ff-only` / `git push` 即可。临时清理镜像和旧bundle位于private，不能推送其中的旧历史。

本地reflog及私下备份可能仍保留旧对象，这些不是公开分支。为了可恢复性没有自动销毁它们；不要创建指向旧提交的公开分支，也不要将旧bundle导入远程。

## 云端及合作同学：推荐重新克隆

**不要在旧仓库直接pull、merge或强推旧main。** 那可能产生无关历史合并，或将被删除的历史重新上传。

先停止写入该仓库的训练/导出作业。检查 `git status`，记录未提交改动和未推送成果；以下流程保留整个旧目录，因此不会直接删除这些内容。为了让已保存配置中的绝对路径继续有效，新克隆应使用原来的目录名/位置。

在旧仓库的父目录执行（示例目录名与项目一致；若实际不同，请先调整）：

```bash
# 停止相关作业后，离开旧仓库
cd ..
# 若备份名已存在，先另选备份名，不要覆盖
test ! -e cg-3dgs-baseline-before-history-clean || exit 1
mv cg-3dgs-baseline cg-3dgs-baseline-before-history-clean
git clone https://github.com/firefly1145141919810/cg-3dgs-baseline.git cg-3dgs-baseline
cd cg-3dgs-baseline
python scripts/bootstrap.py
```

克隆失败时不要删除旧目录；先排查联网问题。旧副本仍可作为备份。`bootstrap.py`需要联网拉取固定子模块，可在联网登录节点执行；现有Conda环境无需重装。

确认新克隆成功后，选择下列一种方式恢复大文件：

- 有足够空间：复制旧目录中的 `data/`、`outputs/`、`archives/`、`logs/`，保留双份备份。
- 不希望复制大型模型：下面仅移动这四个明确目录到新仓库，不移动旧 `.git`。操作后它们位于原来的绝对位置，旧代码和Git历史仍保留在备份目录。

```bash
# 新仓库根目录；有同名目标时停止，不覆盖
for dir in data outputs archives logs; do
  if [ -d "../cg-3dgs-baseline-before-history-clean/$dir" ]; then
    test ! -e "$dir" || exit 1
    mv "../cg-3dgs-baseline-before-history-clean/$dir" "./$dir"
  fi
done
mkdir -p private
```

把个人 `exp.slurm`、`timing.slurm` 以及其他私下配置从旧副本手工复制回来（或放入private）。`*.slurm`已忽略，不会重新提交；使用脚本前检查工作目录、环境和日志路径。旧private目录和原始计时证据应私下保留，不要整目录覆盖新仓库。

**不要把旧的results、report、README或.git整体复制到新克隆。** 已公开的实验结果已在新历史中保留；这样复制会重新带入原始路径或覆盖脱敏文件。如有尚未上传的新实验，只选择该新结果目录，脱敏、检查后再正常提交。未提交代码改动也应逐文件审查移植，不合并整个旧分支；含私密信息的补丁不应发布。

检查新仓库：

```bash
git status --short
python scripts/check_public_repo.py
conda activate cg3dgs
python -m unittest discover -s tests -v
# 只有在已分配GPU时才执行环境诊断
python scripts/doctor.py
```

HTTPS克隆仅解决读取。云端继续用原deploy key推送时，可将新仓库的origin改为SSH；私钥仍留在用户SSH配置目录，不复制进项目：

```bash
git remote set-url origin git@github.com:firefly1145141919810/cg-3dgs-baseline.git
```

建议对旧备份执行 `git -C ../cg-3dgs-baseline-before-history-clean remote remove origin`，降低误推风险。完成迁移并确认结果可用之前，不删除旧备份。所有协作者都应完成同样迁移，避免重新引入旧历史。
