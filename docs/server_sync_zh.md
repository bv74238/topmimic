# GitHub、本地与服务器代码同步

## 代码与数据的位置

- GitHub：`https://github.com/bv74238/topmimic.git`，本地远端名为 `topmimic`；默认分支 `main` 与开发分支 `codex/pace-star` 同步本轮代码。
- 本地：`/home/kevin/Uni-SuperMimic`，当前开发分支 `codex/pace-star`。
- SSH：`ssh -p 31069 root@183.147.142.40`。
- 服务器部署目录：`/root/Uni-SuperMimic-pace-star`，分支 `codex/pace-star`。
- 服务器独立裸仓库：`/root/Uni-SuperMimic-pace-star.git`，本地远端名为 `pace-server`。
- 原上游 `origin`（Motphys/Uni-SuperMimic）保留不变，不向其推送。
- 原服务器项目 `/root/Uni-SuperMimic` 未覆盖；部署目录的 `.venv` 是指向原项目已有环境的符号链接，未上传、复制或提交 Python 环境。

Git 只管理源码、配置、测试、文档与依赖清单/锁文件。**权重、run、输出、环境、缓存、动作/机器人数据及 ZIP 不进 Git**。它们保留在本地和服务器，不因代码同步删除。

同步前已对比两端全部 69 个代码/文档文件，内容一致。服务器原工作快照另外保存在 `git stash` 中以便追溯。本轮保留 GitHub 原 `main` 初始化提交并合并其历史，不强制推送或丢弃远端历史；README 保留项目说明并增加 topmimic 的 PACE/STAR 入口。

## 日常同步

在本地审查并提交需要同步的代码，然后推送同一个提交到 GitHub 和服务器中转仓库：

```bash
cd /home/kevin/Uni-SuperMimic
git status --short
# 先在干净工作区同步 GitHub 可能新增的提交，禁止强行覆盖。
git fetch topmimic
git merge --ff-only topmimic/main
# 修改后按文件审查、git add 并 git commit，不要强制添加被忽略的大文件。
git push --atomic topmimic HEAD:main HEAD:codex/pace-star
git push pace-server HEAD:codex/pace-star
```

若远端存在未合并变更，上述命令会拒绝非快进更新；先审查合并，不要使用 `--force`。若需要发起代码审查，可先只推开发分支，再通过 GitHub 合并到 `main`，最后将最终合并提交同步至本地及服务器。

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

服务器通过裸仓库中转获取与 GitHub 相同的提交，无需把本机 GitHub 凭据或私钥复制到服务器。服务器也配置 `topmimic` 远端；若直接读取公开仓库，可执行 `git fetch topmimic`，但部署更新仍以上述中转流程为准。

Git 推送和 `gh` CLI 的登录状态可能不同；以实际 Git 推送结果为准。如果 Git 推送确实提示身份认证失败，可在本地重新登录（勿把令牌写进仓库或聊天）：

```bash
gh auth login -h github.com
gh auth setup-git
```

## 训练数据不走 Git

- 基础模型：`/root/Uni-SuperMimic-pace-star/runs/supermimic_mjwarp_001/ckpt_012000.pt`，同目录保留匹配的 `last.pt` 和 run 元数据。
- 标准 40 段动作：`/root/Uni-SuperMimic-pace-star/src/unisupermimic/assets/motions/g1/lafan1_supermimic/`。
- 三个自定义动作：`/root/GMR/retargeted_motions/csv/`。

仅在动作有新增或修改时单独上传 CSV，不上传权重或环境：

```bash
scp -P 31069 /home/kevin/GMR/retargeted_motions/csv/*.csv root@183.147.142.40:/root/GMR/retargeted_motions/csv/
```

第二阶段准备、训练与前后各 20 次/动作复测见 [PACE/STAR 手册](pace_star_stage2_zh.md)。部署与 Git 同步不表示训练完成，也不表示已达到提升目标。
