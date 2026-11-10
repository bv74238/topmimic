# GVHMR 视频提取人体姿态

```
conda activate gvhmr
cd ~/GVHMR
python tools/demo/demo.py --video=docs/VideoClips/1_freekick.mp4 -s
```
输出权重hmr4d_results.pt

# GMR 人体姿态重定向到机器人

```
conda activate gmr
cd ~/GMR
python scripts/gvhmr_to_robot.py   --gvhmr_pred_file ~/GVHMR/outputs/demo/1_freekick/hmr4d_results.pt   --robot unitree_g1   --save_path retargeted_motions/freekick_g1.pkl   --record_video
python scripts/batch_gmr_pkl_to_csv.py   --folder /home/kevin/GMR/retargeted_motions
```
输出动作csv文件

# Supermimic 执行参考轨迹

```
cd ~/Uni-SuperMimic
MUJOCO_GL=egl uv run unisupermimic-video   --backend mujoco   --ckpt runs/supermimic_mjwarp_001/ckpt_012000.pt   --motion /home/kevin/GMR/retargeted_motions/csv/freekick_g1.csv   --seconds 6
```

## 评估动作质量

```
MUJOCO_GL=egl uv run unisupermimic-eval   --backend mujoco   --ckpt /home/kevin/Uni-SuperMimic/runs/supermimic_mjwarp_001/ckpt_012000.pt   --motion /home/kevin/GMR/retargeted_motions/csv/tennis_g1.csv   --output /home/kevin/Uni-SuperMimic/outputs/evaluations/tennis_g1_eval.json
```

# 增加新动作
使用新数据训练新动作
目标奖励加上遵循动作约束
与物体的交互
