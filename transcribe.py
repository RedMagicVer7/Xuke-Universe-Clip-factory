"""
Xuke Clip Factory — 转录模块
支持 Whisper (默认) 和 FunASR (可选) 两种引擎
输出统一的带时间戳的转录结果
"""

import os
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import yaml

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────

@dataclass
class Segment:
    """一段转录文本"""
    start: float        # 开始时间（秒）
    end: float          # 结束时间（秒）
    text: str           # 转录文本
    confidence: float = 0.0  # 置信度 0~1

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TranscriptResult:
    """完整转录结果"""
    video_path: str
    engine: str                    # "whisper" or "funasr"
    language: str
    segments: List[Segment] = field(default_factory=list)
    full_text: str = ""
    duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "video_path": self.video_path,
            "engine": self.engine,
            "language": self.language,
            "duration_seconds": self.duration_seconds,
            "full_text": self.full_text,
            "segment_count": len(self.segments),
            "segments": [s.to_dict() for s in self.segments],
        }

    def save_json(self, output_path: str):
        """保存为 JSON 文件"""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"转录结果已保存: {output_path}")

    def get_text_in_range(self, start: float, end: float) -> str:
        """获取某个时间范围内的文本"""
        texts = []
        for seg in self.segments:
            if seg.end > start and seg.start < end:
                texts.append(seg.text)
        return " ".join(texts)


# ──────────────────────────────────────────────
# 音频提取
# ──────────────────────────────────────────────

def extract_audio(video_path: str, output_path: Optional[str] = None,
                  sample_rate: int = 16000) -> str:
    """
    从视频中提取音频为 WAV 格式
    Returns: 音频文件路径
    """
    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_path = tmp.name
        tmp.close()

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",                    # 不要视频
        "-acodec", "pcm_s16le",   # 16-bit PCM
        "-ar", str(sample_rate),  # 采样率
        "-ac", "1",               # 单声道
        output_path
    ]

    logger.info(f"提取音频: {Path(video_path).name} -> {Path(output_path).name}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg 音频提取失败:\n{result.stderr}")

    return output_path


def get_video_duration(video_path: str) -> float:
    """获取视频时长（秒）"""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"无法获取视频时长: {video_path}")
    return float(result.stdout.strip())


# ──────────────────────────────────────────────
# Whisper 引擎
# ──────────────────────────────────────────────

class WhisperEngine:
    """OpenAI Whisper 转录引擎"""

    def __init__(self, model_name: str = "medium", device: str = "cpu",
                 language: str = "zh"):
        self.model_name = model_name
        self.device = device
        self.language = language
        self._model = None

    def _load_model(self):
        if self._model is None:
            import whisper
            logger.info(f"加载 Whisper 模型: {self.model_name} (device={self.device})")
            self._model = whisper.load_model(self.model_name, device=self.device)
        return self._model

    def transcribe(self, audio_path: str) -> List[Segment]:
        """转录音频文件，返回 Segment 列表"""
        model = self._load_model()

        logger.info(f"Whisper 开始转录: {Path(audio_path).name}")
        result = model.transcribe(
            audio_path,
            language=self.language,
            task="transcribe",
            verbose=False,
            word_timestamps=False,
            condition_on_previous_text=True,
            fp16=False if self.device == "cpu" else True,
        )

        segments = []
        for seg in result.get("segments", []):
            segments.append(Segment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"].strip(),
                confidence=seg.get("avg_logprob", 0.0),
            ))

        logger.info(f"Whisper 转录完成: {len(segments)} 段")
        return segments


# ──────────────────────────────────────────────
# FunASR 引擎 (可选)
# ──────────────────────────────────────────────

class FunASREngine:
    """阿里 FunASR Paraformer 转录引擎（中文更准确）"""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from funasr import AutoModel
                logger.info("加载 FunASR Paraformer 模型...")
                self._model = AutoModel(
                    model="paraformer-zh",
                    vad_model="fsmn-vad",
                    punc_model="ct-punc",
                    device=self.device,
                )
            except ImportError:
                raise ImportError(
                    "FunASR 未安装。请运行: pip install funasr modelscope\n"
                    "或在 config.yaml 中切换为 whisper 引擎"
                )
        return self._model

    def transcribe(self, audio_path: str) -> List[Segment]:
        """转录音频文件"""
        model = self._load_model()

        logger.info(f"FunASR 开始转录: {Path(audio_path).name}")
        result = model.generate(input=audio_path, batch_size_s=300)

        segments = []
        for item in result:
            if "sentence_info" in item:
                for sent in item["sentence_info"]:
                    segments.append(Segment(
                        start=sent["start"] / 1000.0,  # ms -> s
                        end=sent["end"] / 1000.0,
                        text=sent["text"].strip(),
                        confidence=sent.get("confidence", 0.0),
                    ))
            else:
                # 回退: 只有完整文本没有分句
                segments.append(Segment(
                    start=0.0,
                    end=0.0,
                    text=item.get("text", "").strip(),
                    confidence=0.0,
                ))

        logger.info(f"FunASR 转录完成: {len(segments)} 段")
        return segments


# ──────────────────────────────────────────────
# 主转录函数
# ──────────────────────────────────────────────

def transcribe_video(video_path: str, config: dict) -> TranscriptResult:
    """
    转录视频文件的主入口
    
    Args:
        video_path: 视频文件路径
        config: 从 config.yaml 加载的配置字典
    
    Returns:
        TranscriptResult 包含所有转录段落
    """
    tc = config.get("transcription", {})
    engine_name = tc.get("engine", "whisper")
    language = tc.get("language", "zh")
    device = tc.get("device", "cpu")

    # 1. 获取视频时长
    duration = get_video_duration(video_path)
    logger.info(f"视频时长: {duration:.1f}s ({duration/3600:.1f}h)")

    # 2. 提取音频
    audio_path = extract_audio(video_path)

    try:
        # 3. 选择引擎并转录
        if engine_name == "funasr":
            engine = FunASREngine(device=device)
        else:
            model_name = tc.get("whisper_model", "medium")
            engine = WhisperEngine(
                model_name=model_name,
                device=device,
                language=language,
            )

        segments = engine.transcribe(audio_path)

        # 4. 构建结果
        full_text = " ".join(seg.text for seg in segments)
        result = TranscriptResult(
            video_path=video_path,
            engine=engine_name,
            language=language,
            segments=segments,
            full_text=full_text,
            duration_seconds=duration,
        )

        return result

    finally:
        # 清理临时音频文件
        if os.path.exists(audio_path) and "/tmp" in audio_path:
            os.unlink(audio_path)
            logger.debug(f"已清理临时音频: {audio_path}")


# ──────────────────────────────────────────────
# CLI 入口
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from tqdm import tqdm

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if len(sys.argv) < 2:
        print("用法: python transcribe.py <视频文件路径> [配置文件路径]")
        sys.exit(1)

    video_file = sys.argv[1]
    config_file = sys.argv[2] if len(sys.argv) > 2 else "config.yaml"

    if not os.path.exists(video_file):
        print(f"错误: 视频文件不存在 - {video_file}")
        sys.exit(1)

    with open(config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    result = transcribe_video(video_file, cfg)

    # 保存结果
    out_name = Path(video_file).stem + "_transcript.json"
    out_path = os.path.join(
        cfg.get("paths", {}).get("output_dir", "output"),
        out_name
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    result.save_json(out_path)

    print(f"\n转录完成!")
    print(f"  引擎: {result.engine}")
    print(f"  时长: {result.duration_seconds/3600:.1f} 小时")
    print(f"  段落: {len(result.segments)}")
    print(f"  输出: {out_path}")
