# 基础函数版 SfM：实现与运行

本仓库只有一套命令行重建流程。图像/几何基础函数来自 OpenCV；矩阵运算、稀疏矩阵与线性方程求解来自 NumPy/SciPy；Matplotlib 生成无窗口预览。特征匹配模块改编自 i-sfm，来源及论文引用见 [ATTRIBUTION.md](ATTRIBUTION.md)。

## 流程与实现

1. **抽帧和内参**：`run_sfm.py` 在时间轴上每 0.5 秒选最近的源帧，保留原始分辨率；`--max-frames 0` 保留全部采样帧。工作副本、源帧号、时间戳和视频哈希写入独立输出目录。消防栓共 33 帧，花园共 61 帧。
2. **特征匹配**：`matching.py` 使用 SIFT、BFMatcher 和 0.75 Lowe ratio 筛选双向一致匹配，跳过空描述子和不足两个近邻的情况；`track_graph.py` 用 F-MAGSAC/RANSAC 做几何检查。
3. **跨帧轨迹**：`track_graph.py` 按匹配距离排序，用并查集连接多帧特征。一条轨迹中每帧最多一个关键点，冲突边被拒绝。可选特征缓存只包含本程序计算的 SIFT/BF 数据，绑定图像像素、参数和匹配源码，不包含相机或点云。
4. **初始化与增量注册**：`incremental_sfm.py` 用 Essential/recoverPose 生成基线候选，检查正深度和视差，并优先有第三视图支持的点。增量注册使用 PnP RANSAC 和位姿精化；至少 12 个且不少于 20% 的内点。失败不赋予位姿，注册后异常会回滚状态；轨迹补全和 BA 后可重试。
5. **多视图三角化**：显式构造 DLT 方程并用 SVD 求点，以跨帧内点数量和重投影误差选优，再用内点做多视图 DLT。每个有效观测必须为正深度、误差不超过 3 px；最终每点至少有 3 个视图支持、最大有效三角化夹角至少 1.5°。颜色在浮点数中平均，避免 uint8 溢出。
6. **Bundle Adjustment**：`sparse_ba.py` 自己实现 Schur 补 Levenberg–Marquardt。优化旋转、平移、XYZ，以及可选共享焦距；OpenCV `projectPoints` 提供基础投影导数，代码自行构造点/焦距导数、块矩阵、阻尼及回代。采用径向鲁棒代价 `sum(sqrt(1+dx²+dy²)-1)`，仅接受降低代价的更新。固定首个相机 6 个参数与与首相机基线最长的另一台相机的一个平移分量，消除整体坐标/尺度的 7 个自由度。
7. **BA 调度与输出**：新增第 3 个视角时、之后每新增 5 个视角及结束时执行 BA。单次最多 100 次残差函数评估；到达预算不算收敛。最终在 BA 和筛选之间迭代，直到观测集合稳定；10 轮仍不稳定则报错。`basic_backend.py` 用最终参数计算误差，分别保存场景点云与相机位姿。

## 内参假设

默认固定针孔模型、零畸变：`fx=fy=1.2*max(width,height)`，主点居中。

| 视频 | 分辨率 | 初始 fx/fy | cx/cy |
|---|---:|---:|---:|
| 消防栓 | 544×960 | 1152 | 272 / 480 |
| 花园标牌 | 1280×720 | 1536 | 640 / 360 |

`--refine-focal` 先从当前匹配的 F 矩阵搜索共享焦距，使 `K.T @ F @ K` 的两个非零奇异值接近，再在 BA 中调整。它是针孔模型假设下的数值估计；主点与畸变不优化，也没有真实尺度。初值、搜索记录、最终内参分别写入 `intrinsics.json`、`focal_initialization.json`、`K.txt`。省略该选项则固定输入内参。

## 可复制命令

安装步骤见 [README](README.md)。在仓库根目录执行，输出目录必须新建或为空：

```bash
export MPLCONFIGDIR=/tmp/ai6121-sfm-mpl
export OPENBLAS_NUM_THREADS=1

.venv/bin/python run_sfm.py --video demos/hydrant/input.mp4 --fps 2 --max-frames 0 --refine-focal --nfeatures 3000 --output outputs/hydrant
.venv/bin/python run_sfm.py --video demos/garden/input.mp4 --fps 2 --max-frames 0 --refine-focal --nfeatures 6000 --output outputs/garden

.venv/bin/python scripts/verify_reconstruction.py outputs/hydrant
.venv/bin/python scripts/verify_reconstruction.py outputs/garden
```

对于图片目录：

```bash
.venv/bin/python run_sfm.py --images path/to/images --calibration path/to/K.txt --output outputs/images
```

`--calibration` 是原分辨率下的 3×3 内参文本文件；图片目录中的 `K.txt` 会自动读取。没有内参时使用上述估计。`--feature-cache path/to/cache.npz` 可复用特征；`--ba-max-nfev` 设置单次 BA 预算；`--threads` 控制特征计算线程数。用 `--help` 查看完整参数。

## 可选立体点云和预览

独立检查通过后，执行 OpenCV SGBM 立体匹配。代码自行检查左右视差误差 ≤1 px、至少 3 个深度图支持、相对深度差 <2%，并用 NumPy/SciPy 做体素融合和距离筛选：

```bash
.venv/bin/python scripts/stereo_cloud.py outputs/hydrant --rectify-alpha -1 --disparities 224
.venv/bin/python scripts/stereo_cloud.py outputs/garden
.venv/bin/python scripts/render_cloud_views.py outputs/hydrant --title Hydrant --stereo
.venv/bin/python scripts/render_cloud_views.py outputs/garden --title 'Yunnan Garden' --stereo
```

`make test`（或 `make demos`）将上述稀疏重建、独立检查、立体点云与渲染串起来，每次为两段视频创建新的 `outputs/video_tests_*/` 目录。无纹理、反光、遮挡区域可能留洞；立体点云与稀疏 SIFT 点云是不同产物。稀疏点的重投影误差不证明立体点云的真实三维精度。

## 验收、测试与文件

“使用一帧”要求求得真实位姿，且最终 BA 使用该帧至少 10 个二维观测。通过条件还包含：全部采样帧均注册；点云非空且坐标有限；平均重投影误差 <2 px；正深度和视差检查通过；每对连续帧至少共享 15 个三维点、20 个几何匹配，位姿推导的 Sampson 误差中位数 ≤3 px。程序不通过时返回非零状态。

`verify_reconstruction.py` 针对本项目的完整 2 fps 视频演示，核对输入/源码哈希，从 PLY 独立复算投影，并重新提取相邻帧 SIFT 进行检查。它要求所有 2 fps 帧均保留，不适用于裁剪图片集或其他抽帧率。它是交叉检查，不是三维真值。

核心输出：`reconstruction.ply`、`preview.png`、`cameras.json`、`observations.json`、`frame_usage.json`、`ba_history.json`、`quality.json`、`metrics.json`、内参、抽帧及 `run.log`。立体阶段额外生成 `stereo_cloud.ply`、`stereo_metrics.json` 和 `stereo_views.png`。本地完整运行保存在忽略的 `outputs/`，GitHub 演示摘要见 [demos/README.md](demos/README.md)。

仓库仅保留两个真实视频作为测试用例，由 `scripts/test_videos.py` 顺序运行；不再包含 pytest、合成测试或旧数据集测试。每次从原视频重新计算 SIFT/BF 匹配、位姿、三维点和 BA，再执行独立检查、立体点云与预览。测试额外要求消防栓 33 帧、花园 61 帧全部参与，以及每次 BA 收敛且不超过 100 次评估。失败会保留日志并返回非零状态，测试摘要写入输出根目录的 `test_results.json`。
