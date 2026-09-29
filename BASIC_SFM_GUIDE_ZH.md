# 基础函数版 SfM：实现、结构与运行

本版响应“不能使用 COLMAP 等完整重建库”的限制。当前 `run_sfm.py` 只接受 `--engine basic`，不调用 PyCOLMAP、COLMAP、Ceres、OpenMVG、GTSAM 或 Open3D。开发过程中的旧后端与运行快照仅保留在本地，不属于此提交。仓库已有的可选基准脚本保留，但本入口不会调用它们。

使用的外部计算能力限于 OpenCV 的图像/几何基础函数，以及 NumPy/SciPy 的通用数值函数；Matplotlib、Plotly 只负责显示。代码没有读取旧 COLMAP 输出的相机、点云或焦距作为本版初值。

## 代码结构

```text
run_sfm.py                 参数、抽帧、内参、日志、版本/源码指纹
  matching.py             OpenCV SIFT + 双向 BF/Lowe ratio 匹配
  feature_cache.py         仅缓存像素绑定的 SIFT/BF 数据
  basic_backend.py         编排重建、验收、输出
    track_graph.py         F-RANSAC、并查集轨迹、共享焦距初始估计
    incremental_sfm.py     初始化、PnP 注册、DLT 三角化、观测筛选
      sparse_ba.py         自编 Schur 补 Levenberg–Marquardt BA
    quality_checks.py      全帧、共享轨迹、深度、夹角、邻接位姿检查
    ply_io.py              二进制/ASCII PLY、NumPy 体素融合
scripts/
  verify_reconstruction.py 独立复算投影与新的相邻帧匹配
  stereo_cloud.py          可选 OpenCV SGBM + 自写多视图深度一致性
  render_ply.py            点云 PNG 预览
  render_cloud_views.py    用最终相机渲染导出的点云
  point_cloud_gallery.py   静态对比、离线旋转查看器
```

`reconstruction.py`、`bundle_adjustment.py` 保留原教学实现与原有回归测试；当前命令行调用上面的新模块。`utilis.py`、`visualize_sfm.py` 的显示也改为 Matplotlib，核心依赖清单移除了 PyCOLMAP 和 Open3D。

## 具体修改与原因

### 1. 保留全部 2 fps 采样帧

使用 OpenCV 读取原视频。在时间轴上每 0.5 秒选最近的源帧，默认 `--max-frames 0`，不再限制 40 帧。源视频不改动；每一帧独立复制到新结果的 `frames/`，时间戳和源帧号写入 `frames.json`。消防栓 33 帧、花园 61 帧。

“使用一帧”必须有求得的真实位姿，并且最终 BA 至少使用该帧的 10 个二维观测。仅完成抽帧、加载或保存位姿均不算通过。

### 2. 自己维护跨帧轨迹

先用 SIFT、BFMatcher 和 0.75 ratio 筛选双向一致的匹配，再用 `findFundamentalMat` 的 MAGSAC/RANSAC 检查两视图几何。缺少描述子或不足两个近邻的匹配直接跳过。

`track_graph.py` 把所有通过检查的匹配边按描述子距离排序，用并查集将同一特征跨帧合并。一条轨迹最多包含每张图片的一个关键点；合并会造成“同帧两个关键点”时拒绝该边。这样每个三维点有明确的多视图观测，不再把各图像对的点孤立地堆叠。

### 3. 基线选择与真实 PnP 注册

用 `findEssentialMat`、`recoverPose` 产生候选基线，检查两个相机中的正深度、至少 1.5° 的视线夹角与候选中位夹角至少 4°。评分优先有第三视图支持的点，并降低高单应性支持的候选权重。

Temple 控制实际暴露了一个问题：单纯按两视图点数选出的 6–7 基线与其他帧连接很弱。加入第三视图支持后选择 0–1，8 帧可以连续注册。开发时保留了失败运行和修复依据，结果摘要见本仓库的验证报告。

增量阶段按已有 2D–3D 对应数量排序，用 `solvePnPRansac` 和 `solvePnPRefineLM` 注册；至少需要 12 个、且不少于 20% 的 RANSAC 内点，随后再次检查正深度与 3 px 重投影阈值。注册失败不赋予位姿；注册后的三角化发生异常会回滚位姿和观测。全局 BA/轨迹补全后重新尝试暂时失败的帧。

### 4. 显式多视图三角化和筛选

`triangulate_dlt` 自己构造每个观测的 DLT 方程，并用 `numpy.linalg.svd` 求齐次三维点。每条轨迹从有足够视差的图像对形成候选，以跨帧内点数量和重投影误差选优，再用内点做多视图 DLT。

每一个有效观测都必须在对应相机前方，误差不超过 3 px；最终三维点必须有至少 3 个有效视图，最大有效三角化夹角至少 1.5°。三视图限制会减少点数，其目的是移除缺乏多视图支持的深度估计。RGB 在浮点数中平均，避免 uint8 相加溢出。

### 5. 自编 BA，不调用成品 BA 求解器

`sparse_ba.py` 显式优化每台相机的 Rodrigues 旋转、平移、每个点的 XYZ，以及可选的全局共享焦距。残差是像素预测与观测之差，采用径向鲁棒损失：

`sum(sqrt(1 + dx² + dy²) - 1)`。

OpenCV `projectPoints` 提供投影对旋转、平移、焦距的基础导数；代码通过链式法则构造对三维点和 log(f) 的导数。回归测试用中央差分验证这些雅可比。

代码自行组装相机块 U、点块 V 和交叉块 W，加入 LM 阻尼，用 Schur 补消去三维点：

`(U - W V^-1 W.T) dc = -gc + W V^-1 gp`

然后回代点增量，只在真实鲁棒代价下降时接受更新。SciPy 仅用于稀疏矩阵表示和线性方程求解。固定一台相机的 6 个参数及另一台相机的一个平移分量，消除整体坐标和尺度的 7 个自由度。

每增加 5 台相机进行一次全局 BA，首次注册第三台相机也执行 BA；结束时 BA 与筛选交替，直到三维点和二维观测集合不再变化，才允许导出；若 10 轮内无法稳定则明确报错。每次最多 100 次残差函数评估，并且迭代数也不超过 100。日志记录接受/拒绝步、代价、终止原因及是否收敛；到达预算不会被记录为收敛。

### 6. 内参从估计值出发

默认仍为 `fx=fy=1.2*max(width,height)`、主点居中、畸变为零。没有真实标定时，`--refine-focal` 先在当前图像匹配算得的 F 矩阵上搜索一个共享焦距，使 `K.T @ F @ K` 的两个非零奇异值接近；再在自编 BA 中联合调整这个焦距。

这只是受针孔模型假设约束的数值估计，不等于完成真实相机标定。主点和畸变不优化，模型仍无真实尺度。初始公式、搜索过程和最终数值分别记录在 `intrinsics.json`、`focal_initialization.json` 和 `K.txt`。

### 7. 自己读写与检查点云

`ply_io.py` 直接读写彩色 PLY。可选立体点云使用 OpenCV `stereoRectify`、`StereoSGBM` 和 `reprojectImageTo3D`，多视图深度一致性检查、体素均值融合、近邻距离筛选均由项目代码实现。没有调用完整的 MVS/点云重建封装。

立体点要求左右视差误差不超过 1 px，至少 3 个深度图支持，相对深度差低于 2%。它增加可见表面的采样密度，不补造遮挡或无纹理区域。稀疏点重投影误差不是立体点的精度证明。

## 可复制的命令

在仓库根目录执行以下命令。输出目录必须是新建或空目录。两段样例视频不随源码提交；复现时先将它们放入根目录的 `test_video/`，也可以用 `--video` 指定自己的视频。

```bash
cd AI6121
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-core.txt
export MPLCONFIGDIR=/tmp/ai6121-sfm-mpl
export OPENBLAS_NUM_THREADS=1

.venv/bin/python -m pytest -q
.venv/bin/python run_sfm.py --images datasets/templeRing --max-frames 8 --output outputs/temple_basic_new

.venv/bin/python run_sfm.py --video test_video/255_1788600394.mp4 --fps 2 --max-frames 0 --refine-focal --nfeatures 3000 --output outputs/hydrant_basic_new
.venv/bin/python run_sfm.py --video test_video/256_1788600415.mp4 --fps 2 --max-frames 0 --refine-focal --nfeatures 6000 --output outputs/garden_basic_new

.venv/bin/python scripts/verify_reconstruction.py outputs/hydrant_basic_new
.venv/bin/python scripts/verify_reconstruction.py outputs/garden_basic_new

.venv/bin/python scripts/stereo_cloud.py outputs/hydrant_basic_new --rectify-alpha -1 --disparities 224
.venv/bin/python scripts/stereo_cloud.py outputs/garden_basic_new
.venv/bin/python scripts/render_cloud_views.py outputs/hydrant_basic_new --title Hydrant --stereo
.venv/bin/python scripts/render_cloud_views.py outputs/garden_basic_new --title 'Yunnan Garden' --stereo
```

`--calibration K.txt` 可指定真实内参；省略 `--refine-focal` 则完全固定输入内参。`--feature-cache 路径.npz` 可选，缓存只包含本程序计算的 SIFT/BF 数据，绑定图像像素哈希、特征设置和匹配源码；像素或设置不匹配会拒绝复用。

## 验收和输出

稀疏输出有 `reconstruction.ply`、`preview.png`、`cameras.json`、`observations.json`、`tracks.json`、`registration.json`、`ba_history.json`、`frame_usage.json`、`quality.json`、`metrics.json` 和 `run.log`。

所有采样帧必须参加最终 BA。除了均值误差小于 2 px、深度与夹角门槛，连续帧之间还必须至少共享 15 个三维点，至少有 20 个几何匹配，位姿推导的 Sampson 误差中位数不超过 3 px。独立脚本从 PLY 重新计算投影，并重新提取 SIFT/估计 F 验证邻接关系；它是独立实现的交叉检查，不是三维真值。

开发时的代码快照、逐次运行日志和证据树保存在本地，并未上传到仓库。已记录的结果、复现边界及测试范围见 [BASIC_SFM_RESULTS_ZH.md](BASIC_SFM_RESULTS_ZH.md)。

本次实现与调试使用了生成式 AI 辅助，代码通过自动化回归测试、已知内参控制和视频实跑验证。
