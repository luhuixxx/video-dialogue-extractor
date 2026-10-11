# 视频对话提取工具

这是一个面向通用视频的中文优先对话文字提取工具。它从视频中提取音频，使用 FunASR 识别语音，再结合声纹特征区分说话人，以 `角色1`、`角色2`、`角色3` 等通用名称输出。工具适合多人对话、会议、课程和访谈，不依赖特定视频平台。

项目还提供独立的字幕画面分段功能：程序按设定间隔采样画面下方区域，在画面变化时保存截图，供后续文字识别使用。长视频支持增量写入和按音频块恢复。

## 功能

- 使用 FunASR Paraformer-zh 识别中文语音。
- 使用 Resemblyzer 提取声纹，自动估计说话人数；也可以预先录入指定说话人的语音样本。
- 以 `角色1`、`角色2` 等标签标注发言；声纹证据不足时标记为 `未判定`。
- 音频只解码一次，声纹特征只提取一次，并逐块写入 JSONL 和 TXT。
- 记录已完成的音频块，支持中断后继续处理。
- 检测字幕区域的画面变化并保存截图。此步骤只负责分段，不进行文字识别。

## 环境要求

- Python 3.10 或更高版本。
- 已安装 FFmpeg，并确保 `ffmpeg` 和 `ffprobe` 可在命令行调用。
- 已安装适合本机处理器或显卡的 PyTorch。
- 首次运行会从 ModelScope 下载语音识别模型，需要联网和足够的磁盘空间。

建议在独立环境中安装：

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Linux 和 macOS 使用 `source .venv/bin/activate` 激活环境。`--device auto` 会检查显卡与当前 PyTorch 构建是否兼容；不兼容时自动使用 CPU。

## 转写视频

没有说话人样本时，程序会自动聚类并估计 2 到 6 个角色：

```powershell
video-extract transcribe input.mp4 `
  --output outputs/transcript.jsonl `
  --chunk-seconds 300 `
  --device auto
```

如果已知说话人数，可以用 `--speakers` 指定：

```powershell
video-extract transcribe input.mp4 `
  --speakers 3 `
  --output outputs/transcript.jsonl
```

对于固定的说话人，可以为每人选择一段干净的独立语音，建立声纹样本。区间顺序对应 `角色1`、`角色2` 等标签：

```powershell
python scripts/enroll_speakers.py input.mp4 artifacts/roles.npz `
  --interval 120,150 `
  --interval 300,330

video-extract transcribe input.mp4 `
  --enrollment artifacts/roles.npz `
  --output outputs/transcript.jsonl
```

长任务中断后，使用相同输入、分块长度和输出路径，添加 `--resume` 跳过已完成的音频块：

```powershell
video-extract transcribe input.mp4 `
  --output outputs/transcript.jsonl `
  --chunk-seconds 300 `
  --resume
```

程序会生成三个文件：

- `transcript.jsonl`：每行一条 UTF-8 编码的记录，便于程序处理。
- `transcript.txt`：带时间戳和角色标签的可读文本。
- `transcript.manifest.json`：已完成块及输入参数，用于断点恢复。

JSONL 记录包含 `start`、`end`、`speaker`、`text`、`alignment_quality` 和 `engine` 字段。使用 `--start` 和 `--duration` 可只处理指定时间段，适合先做小范围测试。

## 提取字幕画面

该命令检测画面下方字幕区域的变化，并把变化后的区域保存为 PNG 图片。它不会识别图片中的文字：

```powershell
video-extract segment-subtitles input.mp4 `
  --output outputs/subtitle-segments `
  --sample-interval 0.25
```

输出目录包含 `segments.jsonl` 和按字幕持续时间命名的图片文件夹。后续可以将图片交给 OCR 或视觉模型处理。

## 项目结构

```text
src/video_dialogue_extractor/   核心流程和命令行入口
scripts/enroll_speakers.py      建立说话人声纹样本
scripts/segment_subtitles.py    字幕分段命令行封装
tests/                          单元测试
docs/ARCHITECTURE.md             模块职责说明
docs/TEST_REPORT.md              本地测试记录
```

## 开发与测试

在项目根目录运行：

```powershell
python -m pytest -q
python -m compileall -q src scripts
```

单元测试使用合成数据，不会下载模型，也不需要测试视频。视频、音频、声纹样本、转写结果和截图均应保存在版本控制之外。

## 准确率与隐私

说话人区分结果属于估计。多人同时说话、背景音乐、短句和相似声线都可能影响结果。重要内容建议人工抽查；换视频时可以重新建立声纹样本。项目没有带人工标注真值的评测集，因此不宣称固定的准确率。

仓库不包含视频、音频、个人声纹、模型文件或转写结果。模型缓存和生成文件保留在本地，并由 `.gitignore` 排除。

## 许可证

本项目采用 [MIT 许可证](LICENSE)。
