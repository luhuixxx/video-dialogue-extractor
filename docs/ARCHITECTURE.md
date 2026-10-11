# Architecture

`media.py` 负责 FFmpeg/ffprobe；`speakers.py` 负责一次性 embedding 提取、enrollment、自动聚类和滑动窗口声纹分段；`asr.py` 只负责 FunASR 调用与时间戳标准化；`output.py` 负责 UTF-8 增量落盘和块级 manifest；`pipeline.py` 编排一次音频抽取、动态分段和从缓存 WAV 按块识别；`subtitles.py` 是独立的字幕画面变化检测。

模型缓存由运行环境管理，不写死在代码中。ModelScope、Hugging Face 和 PyTorch 分别读取 `MODELSCOPE_CACHE`、`HF_HOME` 和 `TORCH_HOME`；Resemblyzer 声纹权重可通过 `VIDEO_DIALOGUE_RESEMBLYZER_WEIGHTS` 指定。未设置声纹路径时，使用 Resemblyzer 安装包自带的默认权重。

角色标签来自 `artifacts/*.npz` 的 profile 名称，默认按 enrollment 顺序生成 `角色1`、`角色2`、`角色3`。没有 enrollment 时，`cluster_profiles` 使用滑动声纹 embedding 和轮廓系数估计角色数量，再生成相同的通用角色名。没有覆盖到有效声纹区段时保留 `未判定`。源代码不包含具体个人的 embedding。
