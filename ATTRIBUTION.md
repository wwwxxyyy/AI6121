# 开源来源与致谢

本项目基于 [zaarAli/i-sfm](https://github.com/zaarAli/i-sfm) 的教学型增量 SfM 实现继续开发，由 [yitong0322/AI6121](https://github.com/yitong0322/AI6121) 的初始项目导入版本延续而来。

`matching.py` 保留并修改了 i-sfm 的特征提取与匹配实现，增加空描述子/不足近邻处理、双向一致性匹配和可配置线程。本项目新增了视频入口、跨帧轨迹管理、增量注册与回滚、多视图三角化、自编 Schur-LM Bundle Adjustment、质量检查及两个视频演示。不能将本仓库全部代码视为从零编写。

原项目采用 MIT License；其版权声明 `Copyright (c) 2025 Zaar Ali` 和完整许可文本保留于 [LICENSE](LICENSE)。

## 原项目论文引用

上游请求引用 [3D Reconstruction via Incremental Structure From Motion](https://arxiv.org/abs/2508.01019)：

```bibtex
@article{zeeshan2025incrementalSfM,
  title={3D Reconstruction via Incremental Structure From Motion},
  author={Muhammad Zeeshan and Umer Zaki and Syed Ahmed Pasha and Zaar Khizar},
  journal={arXiv preprint arXiv:2508.01019},
  year={2025},
  url={https://arxiv.org/abs/2508.01019}
}
```

## 学习参考与基础依赖

初始项目记录的学习参考包括 [sakshikakde/SFM](https://github.com/sakshikakde/SFM)、[RISHIT7/sfm-pipeline](https://github.com/RISHIT7/sfm-pipeline) 和 [muneebaadil/structure-from-motion](https://github.com/muneebaadil/structure-from-motion)。保留这些历史致谢；当前代码不导入或调用这些仓库，运行时无需下载它们。

图像与几何基础操作使用 [OpenCV](https://docs.opencv.org/4.x/)，数值运算使用 [NumPy](https://numpy.org/doc/) 与 [SciPy](https://docs.scipy.org/doc/scipy/)，预览使用 [Matplotlib](https://matplotlib.org/stable/)。各依赖保留其各自许可证。两段演示视频由本项目使用者提供。

本项目后续实现、调试与文档整理使用了生成式 AI 辅助。
