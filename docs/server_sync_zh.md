# GitHub、本地与服务器代码同步

## 代码与数据的位置

- GitHub 目标仓库：`https://github.com/bv74238/topmimic.git`，本地远端名为 `topmimic`。
- 本地：`/home/kevin/Uni-SuperMimic`，专用开发分支 `codex/pace-star`。
- SSH：`ssh -p 31069 root@183.147.142.40`。
- 服务器部署目录：`/root/Uni-SuperMimic-pace-star`，分支 `codex/pace-star`。
- 服务器独立裸仓库：`/root/Uni-SuperMimic-pace-star.git`，本地远端名为 `pace-server`。
- 原上游 `origin`（Motphys/Uni-SuperMimic）保留不变，不向其推送。
- 原服务器项目 `/root/Uni-SuperMimic` 未覆盖；部署目录的 `.venv` 是指向原项目已有环境的符号链接，未上传、复制或提交 Python 环境。

Git 只管理源码、配置、测试、文档与依赖清单/锁文件。**权重、run、输出、环境、缓存、动作/机器人数据及 ZIP 不进 Git**。它们保留在本地和服务器，不因代码同步删除。

本次整理前已对比两端全部 69 个代码/文档文件，内容一致。GitHub 当前登录曾返回 401；只有推送命令成功后才表示 GitHub 同步完成，不能把新增远端当成已上传。

## 日常同步

首次使用 GitHub 若认证失效，在本地终端重新登录，勿把访问令牌写进仓库或聊天：

```bash
gh auth login -h github.com
gh auth setup-git
```

在本地审查并提交需要同步的代码，然后推送同一个提交到 GitHub 和服务器中转仓库：

```bash
cd /home/kevin/Uni-SuperMimic
git status --short
# 先按文件审查、git add 并 git commit，不要强制添加被忽略的大文件。
git push topmimic HEAD:codex/pace-star
git push pace-server HEAD:codex/pace-star
```

这是独立分支，不强推或覆盖目标仓库已有 `main`。若要把代码并入 GitHub 的默认分支，在 GitHub 审查并合并相应分支；已有历史不一致时不要使用强推。

服务器收到推送后更新部署目录：

```bash
ssh -p 31069 root@183.147.142.40
cd /root/Uni-SuperMimic-pace-star
git status --short
# 工作区干净时才拉取；有修改先审查并备份，不要强制覆盖。
git pull --ff-only origin codex/pace-star
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python -m unisupermimic.scripts.pace --help
```

服务器通过裸仓库中转获取与 GitHub 相同的提交，无需把本机 GitHub 凭据或私钥复制到服务器。服务器也配置 `topmimic` 远端，若未来要直接从 GitHub 拉取，须确保服务器具备对应仓库的读取权限。

## 训练数据不走 Git

- 基础模型：`/root/Uni-SuperMimic-pace-star/runs/supermimic_mjwarp_001/ckpt_012000.pt`，同目录保留匹配的 `last.pt` 和 run 元数据。
- 标准 40 段动作：`/root/Uni-SuperMimic-pace-star/src/unisupermimic/assets/motions/g1/lafan1_supermimic/`。
- 三个自定义动作：`/root/GMR/retargeted_motions/csv/`。

仅在动作有新增或修改时单独上传 CSV，不上传权重或环境：

```bash
scp -P 31069 /home/kevin/GMR/retargeted_motions/csv/*.csv root@183.147.142.40:/root/GMR/retargeted_motions/csv/
```

第二阶段准备、训练与前后各 20 次/动作复测见 [PACE/STAR 手册](pace_star_stage2_zh.md)。部署与 Git 同步不表示训练完成，也不表示已达到提升目标。
