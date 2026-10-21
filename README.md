# Xuke Clip Factory — 直播切片自动化工厂

将 3 小时视频号直播录像自动变成**即发布的短视频发布包**：转录 → AI 高光检测 → FFmpeg 剪辑 → 封面生成 → 多平台文案 → 多规格输出。

## 功能特性

- **自动转录**：FunASR Paraformer（默认，中文更准）或 OpenAI Whisper，本地运行
- **三路高光评分**：LLM 语义分析 + 音频能量验证 + 弹幕热度信号（可选）
- **自动剪辑**：FFmpeg 切片、ASS 字幕烧录、淡入淡出、横屏→竖屏裁切
- **封面自动生成**：切片高光帧 + 标题大字 + 头像角标，输出 1080x1920 竖屏封面
- **多平台文案**：按徐可人设生成视频号/抖音/小红书专属标题、描述、话题标签、前3秒钩子
- **多规格发布包**：视频号 / 抖音 / 小红书 / X / B站 各平台尺寸与时长规格适配
- **断点续跑**：Pipeline 状态持久化，已完成的步骤自动跳过

## 工作流程

```
直播录像 (3h mp4)
   │
   ├─ Step 1  transcribe    音频提取 + FunASR/Whisper 转录 → transcript.json
   │
   ├─ Step 2  highlight     LLM 语义分析 (gpt-4.1-mini)
   │                        + 音频能量 (RMS/过零率/动态)
   │                        + 弹幕热度 (可选, B站XML)      → highlights.json
   │
   ├─ Step 3  clip          FFmpeg 切片 + 字幕烧录 + 封面  → clips/*.mp4 + *_cover.jpg
   │
   ├─ Step 4  copywriting   徐可人设多平台文案              → copywriting.json
   │
   └─ Step 5  publish       多平台发布包 (尺寸/时长适配)    → publish_packages.json
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

系统依赖：`ffmpeg` / `ffprobe`（`brew install ffmpeg`）。

FunASR 首次运行会自动从 ModelScope 下载模型（paraformer-zh + fsmn-vad + ct-punc）。

### 2. 配置

复制并修改 `config.yaml` 关键项：

```yaml
paths:
  source_video_dir: "/path/to/recordings"   # 直播录像目录
  output_dir: ".../output"

transcription:
  engine: "funasr"          # funasr (推荐) / whisper

highlight:
  llm_api_key_env: "OPENAI_API_KEY"   # API Key 走环境变量，不落盘
  danmaku_file: ""          # 可选：B站 XML 弹幕文件，填了才启用弹幕评分
```

### 3. 设置 API Key

```bash
export OPENAI_API_KEY=your_key
```

### 4. 运行

```bash
python pipeline.py <直播录像.mp4>          # 处理单个视频
python pipeline.py --scan                  # 扫描源目录，处理所有新视频
python pipeline.py --watch                 # 持续监控 + 自动处理
python pipeline.py <视频> --steps clip     # 只重跑某个步骤
python pipeline.py <视频> --no-skip        # 忽略断点，全部重跑
```

产物输出到 `output/<视频名>/`：

```
output/livestream_0414/
├── transcript.json           # 带时间戳的转录
├── highlights.json           # 高光片段 + 三路评分明细
├── clips.json                # 切片清单
├── *_clip01_标题.mp4         # 成片（字幕已烧录）
├── *_clip01_cover.jpg        # 竖屏封面
├── copywriting.json          # 多平台文案
├── publish_packages.json     # 发布包索引
└── _pipeline_state.json      # 断点状态
```

## 配置速查

| 配置段 | 作用 | 常用项 |
|---|---|---|
| `transcription` | 转录引擎 | `engine` (funasr/whisper), `device` (cpu/mps/cuda) |
| `highlight` | 高光检测 | `min_total_score` 过滤阈值, `audio_energy_weight`, `danmaku_file`/`danmaku_weight` |
| `clipping` | 剪辑参数 | `padding_before/after`, `fade_in/out`, `video_crf` |
| `cover` | 封面生成 | `enabled`, `width/height`, `font_size`, `badge_image` |
| `subtitle` | 字幕样式 | `font_name`, `font_size`, `margin_bottom` |
| `copywriting` | 文案 LLM | `llm_model`, `llm_api_key_env` |
| `platforms` | 平台规格 | 各平台 `aspect`/`max_duration`/`enabled` |
| `obs` | OBS 自动录制 | `port`, `password`（配合 auto_record.py） |

### 弹幕热度评分（可选）

高光综合分 = 语义 x 0.5 + 能量 x 0.3 + 弹幕 x 0.2。
不提供弹幕文件时自动退化为 语义 x 0.7 + 能量 x 0.3 两路，行为与旧版一致。

弹幕文件用 B站 XML 格式（事实标准），可由录播姬、DanmakuRender 等工具从直播录制中导出。片段内弹幕密度达全片平均 = 25 分，2 倍 = 40 分，4 倍以上 = 50 分封顶。

## 项目结构

```
clip-factory/
├── pipeline.py           # 主编排器（5 步流水线 + 状态管理）
├── transcribe.py         # 转录：FunASR / Whisper 双引擎
├── highlight_detect.py   # 高光：语义 + 能量 + 弹幕 三路评分
├── clip_editor.py        # 剪辑：切片/字幕/拼接/封面
├── copywriter.py         # 文案：徐可人设多平台生成
├── platform_adapter.py   # 多平台规格适配与发布包
├── auto_record.py        # OBS WebSocket 直播自动录制
├── config.yaml           # 全部配置
└── assets/               # 角标等静态素材
```

各模块也可独立运行（`python transcribe.py <视频>` 等），通过 JSON 中间产物串联。

## 致谢

设计参考了这些优秀的开源项目：

- [hotclip](https://github.com/xixihhhh/hotclip) — 弹幕热度进爆点判断的思路
- [AutoClip](https://github.com/zhouxiaoka/autoclip) — 封面自动生成 + 发布文案一体化的思路
- [OpenClip](https://github.com/linzzzzzz/openclip) — 面向直播回放的轻量流水线

差异化保留：OBS 自动录制、徐可 persona 文案、视频号规格适配为自建能力。
