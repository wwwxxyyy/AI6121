# 两个视频演示

原视频由项目使用者提供，分别是消防栓与花园标牌。以下结果来自 **2026-10-16 清理后的代码实跑**，保留每秒两帧的全部采样，均通过独立验证和最终 BA。回归测试：31 项通过。

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
| 本次稀疏重建耗时（已缓存特征） | 15.61 s | 51.03 s |
| 立体阶段耗时 | 6.66 s | 23.79 s |

本次只复用了绑定图像像素、参数和匹配源码的 SIFT/BF 缓存，位姿、焦距、三维点与 BA 均重新计算；独立检查还重新提取了相邻帧特征。首次运行命令不使用缓存，需要额外计算全部特征和匹配，因此耗时更长。以上耗时不包含独立检查与最后渲染，不能当作首次运行总耗时。

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
make demo-hydrant
make demo-garden
```

结果分别进入 `outputs/hydrant/` 和 `outputs/garden/`。完整立体点云保存在各目录的 `stereo_cloud.ply`；抽帧、日志、中间记录也在同一输出目录。目录必须为空；已有运行要保留时，可参照 [中文说明](../BASIC_SFM_GUIDE_ZH.md) 使用新的 `--output` 路径。

## 预览

### 消防栓

![消防栓点云](hydrant/preview.png)

[稀疏 PLY](hydrant/reconstruction.ply) · [立体预览 PLY](hydrant/stereo_preview.ply) · [统计](hydrant/summary.json)

### 花园标牌

![花园标牌点云](garden/preview.png)

[稀疏 PLY](garden/reconstruction.ply) · [立体预览 PLY](garden/stereo_preview.ply) · [统计](garden/summary.json)
