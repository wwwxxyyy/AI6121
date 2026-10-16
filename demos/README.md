# 两个视频演示

原视频由项目使用者提供，分别是消防栓与花园标牌。以下结果来自 **2026-10-16 清理后的代码实跑**，保留每秒两帧的全部采样，均通过独立验证和最终 BA。本仓库仅以这两段真实视频作为测试用例；`make test` 的两项测试均通过。

| 指标 | [消防栓](hydrant/) | [花园标牌](garden/) |
|---|---:|---:|
| 原视频 | [input.mp4](hydrant/input.mp4) | [input.mp4](garden/input.mp4) |
| 分辨率 | 544×960 | 1280×720 |
| 使用帧数 / 采样帧数 | 33 / 33 | 61 / 61 |
| 稀疏三维点数 | 1,343 | 4,199 |
| 最终平均重投影误差 | 0.493 px | 0.465 px |
| BA 次数（全部收敛） | 11 | 17 |
| 完整立体点云点数 | 287,209 | 2,399,891 |
| 随仓库提供的立体预览点数 | 100,000 | 100,000 |
| 稀疏重建耗时（含特征提取与匹配） | 44.24 s | 415.43 s |
| 立体阶段耗时 | 6.76 s | 24.39 s |
| 单视频完整测试耗时 | 60.72 s | 462.16 s |

本次未使用特征缓存，从原视频重新计算了 SIFT/BF 匹配、位姿、焦距、三维点和 BA；独立检查再次提取相邻帧特征。完整测试耗时包括稀疏重建、独立检查、立体阶段和预览，具体速度取决于机器配置。

## 文件说明

每个目录均包含：

- `input.mp4`：原视频的逐字节副本。
- `reconstruction.ply`：完整的彩色 SIFT 稀疏点云。
- `stereo_preview.ply`：从完整立体点云确定性均匀抽取的 100,000 个真实点，便于下载查看；未插值或生成几何。
- `preview.png`：完整立体点云从两个重建相机视角的投影，无原照片叠加。
- `K.txt`、`cameras.json`：最终内参与相机位姿。
- `summary.json`：帧数、误差、耗时、BA、源码/输入/演示产物哈希及复现参数。
- `independent_verification.json`：从导出的 PLY 独立复算投影，以及重新匹配相邻帧后的验证结果。

内参来自针孔模型下的数值估计，畸变为零，场景包含背景且没有真实尺度。稀疏重投影误差不能证明立体点云的实际三维精度；无纹理、反光和遮挡区域仍可能出现缺失或噪声。

## 重现完整结果

在安装依赖后，从仓库根目录运行：

```bash
make test
```

测试每次自动创建新的 `outputs/video_tests_*/`，其下分别保存 `hydrant/` 和 `garden/` 的完整立体点云 `stereo_cloud.ply`、抽帧与中间记录。根目录的 `test_results.json` 汇总两项测试结果，日志按视频与阶段保存。已有结果不会被覆盖；单独运行视频可参照 [中文说明](../BASIC_SFM_GUIDE_ZH.md)。

## 预览

### 消防栓

![消防栓点云](hydrant/preview.png)

[稀疏 PLY](hydrant/reconstruction.ply) · [立体预览 PLY](hydrant/stereo_preview.ply) · [统计](hydrant/summary.json)

### 花园标牌

![花园标牌点云](garden/preview.png)

[稀疏 PLY](garden/reconstruction.ply) · [立体预览 PLY](garden/stereo_preview.ply) · [统计](garden/summary.json)
