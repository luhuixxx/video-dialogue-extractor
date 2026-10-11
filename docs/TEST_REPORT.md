# 测试报告

测试视频：`D:\\1.mp4`，时长约 649.36 秒；原有长视频仍保留为历史回归样本。

| 检查 | 结果 |
| --- | --- |
| `python -m pytest -q` | 11 passed |
| `python -m compileall -q src scripts` | passed |
| 字幕变化检测，120 秒，0.25 秒采样 | 168 条 metadata，168 张 PNG |
| 音频 smoke test，20 秒 | 3 条 JSONL，UTF-8，无空文本 |
| 音频动态分段，120 秒 | 12 条 JSONL，包含两名说话人切换 |
| 整段音频转写 | 914 条 JSONL，最后时间戳 15425.455 秒 |
| `D:\\1.mp4` 自动多人转写 | 74 条 JSONL，角色1-角色6，未判定 0，最后时间戳 647.63 秒；每行含 `alignment_quality` |
| `D:\\1.mp4` `--resume` 回归 | manifest 识别 3 个已完成块，未重复写入 |

整段输出核对了：JSONL 可用 UTF-8 解码、时间戳单调递增、空文本为 0、TXT 行数与 JSONL 一致，所有记录含 `alignment_quality`。音频只由 FFmpeg 解码一次，声纹 embedding 只提取一次，后续 ASR 从缓存 WAV 分块读取。RTX 5060 Ti 当前安装的 PyTorch 只编译到 `sm_90`，而显卡是 `sm_120`，因此 `--device auto` 按预期记录警告并使用 CPU；安装支持该架构的 PyTorch/CUDA 后可重新使用 GPU。

边界回归覆盖了负起始时间、非正时长、非正字幕采样间隔和负变化阈值；字幕视频句柄在异常路径也会释放。

声纹测试使用本机 enrollment 文件 `artifacts/test_speakers.npz`，角色名为 `角色1`、`角色2`，没有写入源代码，也不应提交到公开仓库。`D:\\1.mp4` 使用自动聚类生成角色1-角色6。准确率没有人工真值集，不宣称 99.99%；多人或重叠说话仍应抽查并按视频重新建立 enrollment。
