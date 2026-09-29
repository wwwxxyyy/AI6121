# AI6121 SfM 项目环境与工作流

> 当前基础函数版视频入口见 [BASIC_SFM_GUIDE_ZH.md](BASIC_SFM_GUIDE_ZH.md)，使用 Python 3.11 和 `requirements-core.txt`。下文保留原 Notebook/可选基准工作流；新的 `run_sfm.py` 不调用这些外部重建后端。

本目录以 `zaarAli/i-sfm` 为唯一主代码库。原上游已经配置为 Git remote
`upstream`；数学模块、评估逻辑和流程参考代码放在 `references/`，不会混入主线提交。

## 1. 一次性安装

本机完整安装（含 CUDA 13.0 COLMAP）：

```bash
bash scripts/bootstrap.sh --with-colmap
bash scripts/fetch_references.sh
conda activate ai6121-sfm
python -m pytest -q tests/test_environment.py
```

只安装 Python baseline 时，省略 `--with-colmap`。Python 环境固定为 3.10，
并包含 NumPy 1.26、OpenCV/SIFT、Open3D、SciPy、JupyterLab、FFmpeg 等依赖。
COLMAP 使用独立环境，避免其 Qt/CUDA/C++ 依赖改变 baseline 的 Python 环境。

## 2. 目录约定

```text
datasets/
├── templeRing -> ../templeRing     # 仓库自带 baseline
├── House -> ../House               # 仓库自带 baseline
├── park_gate -> ../park_gate       # 仓库自带 baseline
├── scene_a/                         # 开发场景：图像 + K.txt
├── scene_b/                         # 固定算法后的泛化测试
└── scene_c/                         # 固定算法后的泛化测试
outputs/                              # i-sfm 结果（Git 忽略）
colmap-workspaces/                    # COLMAP 结果（Git 忽略）
references/                           # 固定版本的参考仓库（Git 忽略）
```

每个 Scene 目录直接放按拍摄顺序排列的图像和一个 3×3 的 `K.txt`：

```text
00.jpg  01.jpg  02.jpg  ...  K.txt
```

不要混用不同焦段、数码变焦或分辨率。每张相邻图像建议保留约 60%–80% 重叠，
围绕场景平移拍摄，避免只站在原地旋转。视频可先抽帧：

```bash
bash scripts/extract_frames.sh input.mp4 datasets/scene_a 2
```

第三个参数是每秒帧数；输出目录必须为空，以防覆盖已有数据。

## 3. 标定与原始 baseline

1. 激活环境并启动 Notebook：

   ```bash
   conda activate ai6121-sfm
   jupyter lab
   ```

2. 用 `camera calibration.ipynb` 处理同一手机、同一焦段和分辨率拍摄的棋盘格图像，
   将最终内参保存为对应 Scene 的 `K.txt`。
3. 在 `main.ipynb` 的配置单元把 `dataset_path` 改为 `./datasets/scene_a`。
4. 选择 `Python (AI6121 SfM)` kernel，按顺序运行全部 SfM 单元；点云写入 `outputs/`。

仓库自带的最小验证数据已经映射到 Notebook 期望的位置。可先保持默认的
`./datasets/templeRing` 跑通，再接入 Scene A。

## 4. COLMAP 基准

以下命令使用同一个 `K.txt`、PINHOLE 模型和 exhaustive matching 生成稀疏基准：

```bash
bash scripts/run_colmap.sh \
  datasets/scene_a \
  colmap-workspaces/scene_a \
  datasets/scene_a/K.txt \
  exhaustive
```

输出包括 `sparse/0` 模型、`sparse.ply` 点云、`txt/` 文本模型以及 `metrics.txt`。Scene 图片较多且按视频
顺序采样时，可将最后一项换成 `sequential`。报告中至少比较：

- 注册图像数 / 总图像数；
- 稀疏 3D 点数；
- 平均重投影误差（pixel）；
- Scene A/B/C 的成功率和失败案例；
- 运行时间（相同机器、相同输入分辨率）。

### 前景掩码与稠密点云

当目标物体周围的树木、行人或地面产生大量无关特征时，先为每帧生成前景掩码：

```bash
python scripts/remove_background.py datasets/scene_a outputs/scene_a/preprocessed \
  --model isnet-general-use --threshold 48 --close-size 7 --dilate 5 --suppress-green
```

用原图和掩码进行稀疏重建。掩码只控制特征提取，不会引入黑色边界特征：

```bash
bash scripts/run_colmap.sh \
  datasets/scene_a \
  colmap-workspaces/scene_a-masked \
  datasets/scene_a/K.txt \
  exhaustive \
  outputs/scene_a/preprocessed/masks
```

需要表面更完整的点云时，使用生成的黑底前景帧运行稠密匹配：

```bash
bash scripts/run_colmap_dense.sh \
  outputs/scene_a/preprocessed/masked_rgb \
  colmap-workspaces/scene_a-masked/sparse/0 \
  colmap-workspaces/scene_a-masked-dense \
  photometric

python scripts/clean_point_cloud.py \
  colmap-workspaces/scene_a-masked-dense/fused.ply \
  colmap-workspaces/scene_a-masked-dense/fused-clean.ply
```

`photometric` 较快；可改为 `geometric` 做更严格但更慢的几何一致性筛选。

## 5. Git/Fork 工作流

当前没有可用的个人 GitHub Fork 凭据，因此只配置了只读上游：

```bash
git remote -v
# upstream  https://github.com/zaarAli/i-sfm.git
```

在 GitHub 创建个人 Fork 后执行：

```bash
git remote add origin git@github.com:<YOUR_ACCOUNT>/i-sfm.git
git push -u origin main
```

之后用 `git fetch upstream` 获取原项目更新；项目改进只提交到个人 `origin`。
