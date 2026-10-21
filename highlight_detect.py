"""
Xuke Clip Factory — AI 高光检测模块
三阶段检测：LLM 语义分析 + 音频能量验证 + 弹幕热度信号(可选)
"""

import os
import json
import logging
import re
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

import yaml
import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────

@dataclass
class Highlight:
    """一个高光片段"""
    start: float               # 开始时间（秒）
    end: float                 # 结束时间（秒）
    title: str                 # 片段标题
    reason: str                # 为什么是高光
    category: str              # 分类: 金句 / 故事 / 互动 / 情感 / 争议 / 教学
    semantic_score: float      # 语义评分 (0-50)
    energy_score: float = 0.0  # 音频能量评分 (0-50)
    danmaku_score: float = 0.0  # 弹幕热度评分 (0-50, 无弹幕数据时为0)
    total_score: float = 0.0   # 综合评分 (0-100, 加权)
    transcript_excerpt: str = ""  # 对应转录文本片段

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HighlightResult:
    """高光检测结果"""
    video_path: str
    highlights: List[Highlight] = field(default_factory=list)
    total_analyzed_minutes: float = 0.0

    def to_dict(self) -> dict:
        return {
            "video_path": self.video_path,
            "total_analyzed_minutes": self.total_analyzed_minutes,
            "highlight_count": len(self.highlights),
            "highlights": [h.to_dict() for h in self.highlights],
        }

    def save_json(self, output_path: str):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"高光检测结果已保存: {output_path}")


# ──────────────────────────────────────────────
# 阶段一：LLM 语义分析
# ──────────────────────────────────────────────

HIGHLIGHT_SYSTEM_PROMPT = """你是一个专业的短视频内容编辑，专门分析直播回放并找出最适合做成短视频的精彩片段。

你的分析对象是"徐可"(Xu Ke)，一位视频号情感类头部主播，她的内容特点：
- 以犀利、直白的方式谈论情感和人生话题
- 经常爆出金句，语言风格辛辣有力
- 会讲述个人经历（离婚、创业、带女儿等）
- 与观众互动活跃，常有即兴反应
- 偶尔涉及争议话题，观点鲜明

请根据以下维度给每个潜在高光片段评分（每项0-10分，总分0-50）：
1. 情绪冲击力：这段内容能否在3秒内抓住观众注意力？
2. 金句密度：是否包含容易传播的金句或经典表达？
3. 故事完整性：片段是否有完整的起承转合？
4. 互动/共鸣：是否容易引发评论区互动？
5. 传播潜力：观众是否会主动分享？

重要规则：
- 每个片段时长应在15秒~3分钟之间
- 片段需要有自然的开始和结束点
- 优先选择能独立成篇的内容（不需要上下文就能理解）
- 不要选择含有敏感政治内容的片段"""

HIGHLIGHT_USER_PROMPT_TEMPLATE = """以下是直播转录文本，时间范围 {start_time} - {end_time}：

---
{transcript_chunk}
---

请找出其中所有适合做成短视频的高光片段。对每个片段，请严格按以下 JSON 格式输出：

```json
[
  {{
    "start_time": "HH:MM:SS",
    "end_time": "HH:MM:SS",
    "title": "片段标题（10字以内）",
    "reason": "为什么选这段（一句话）",
    "category": "金句/故事/互动/情感/争议/教学",
    "scores": {{
      "emotion_impact": 8,
      "quote_density": 9,
      "story_completeness": 7,
      "interaction_potential": 8,
      "share_potential": 9
    }}
  }}
]
```

如果这段没有值得剪辑的高光内容，请返回空数组 `[]`。
只返回 JSON，不要其他解释。"""


def _format_time(seconds: float) -> str:
    """秒数转 HH:MM:SS"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _parse_time(time_str: str) -> float:
    """HH:MM:SS 转秒数"""
    parts = time_str.strip().split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(parts[0])


def _chunk_segments(segments: list, chunk_minutes: int = 10) -> List[dict]:
    """
    将转录段落按时间分块，每块约 chunk_minutes 分钟
    返回: [{start, end, text}, ...]
    """
    if not segments:
        return []

    chunk_seconds = chunk_minutes * 60
    chunks = []
    current_chunk = {"start": segments[0].start, "texts": [], "end": 0.0}

    for seg in segments:
        current_chunk["texts"].append(
            f"[{_format_time(seg.start)}] {seg.text}"
        )
        current_chunk["end"] = seg.end

        if seg.end - current_chunk["start"] >= chunk_seconds:
            chunks.append({
                "start": current_chunk["start"],
                "end": current_chunk["end"],
                "text": "\n".join(current_chunk["texts"]),
            })
            current_chunk = {"start": seg.end, "texts": [], "end": seg.end}

    # 最后一块
    if current_chunk["texts"]:
        chunks.append({
            "start": current_chunk["start"],
            "end": current_chunk["end"],
            "text": "\n".join(current_chunk["texts"]),
        })

    return chunks


def _call_llm(system_prompt: str, user_prompt: str, config: dict) -> str:
    """调用 LLM API"""
    hc = config.get("highlight", {})
    provider = hc.get("llm_provider", "openai")
    model = hc.get("llm_model", "gpt-4.1-mini")
    api_key_env = hc.get("llm_api_key_env", "OPENAI_API_KEY")
    api_key = os.environ.get(api_key_env, "")

    if not api_key:
        raise ValueError(
            f"未找到 API Key。请设置环境变量: export {api_key_env}=your_key"
        )

    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=4096,
        )
        return response.choices[0].message.content

    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text

    else:
        raise ValueError(f"不支持的 LLM 提供商: {provider}")


def _parse_llm_response(response_text: str, chunk_start: float) -> List[Highlight]:
    """解析 LLM 返回的 JSON"""
    # 提取 JSON 部分
    json_match = re.search(r'\[[\s\S]*?\]', response_text)
    if not json_match:
        logger.warning("LLM 返回中未找到有效 JSON")
        return []

    try:
        items = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        logger.warning(f"JSON 解析失败: {e}")
        return []

    highlights = []
    for item in items:
        try:
            scores = item.get("scores", {})
            semantic_score = sum([
                scores.get("emotion_impact", 0),
                scores.get("quote_density", 0),
                scores.get("story_completeness", 0),
                scores.get("interaction_potential", 0),
                scores.get("share_potential", 0),
            ])

            h = Highlight(
                start=_parse_time(item["start_time"]),
                end=_parse_time(item["end_time"]),
                title=item.get("title", "未命名"),
                reason=item.get("reason", ""),
                category=item.get("category", "其他"),
                semantic_score=semantic_score,
                transcript_excerpt="",
            )

            # 有效性检查
            if h.duration < 5 or h.duration > 600:
                logger.debug(f"跳过时长异常的片段: {h.duration:.0f}s")
                continue

            highlights.append(h)

        except (KeyError, ValueError) as e:
            logger.warning(f"解析单个高光失败: {e}")
            continue

    return highlights


def detect_highlights_semantic(transcript_result, config: dict) -> List[Highlight]:
    """
    阶段一：用 LLM 分析转录文本，找出高光片段
    """
    hc = config.get("highlight", {})
    chunk_minutes = hc.get("chunk_minutes", 10)

    chunks = _chunk_segments(transcript_result.segments, chunk_minutes)
    logger.info(f"语义分析: 共 {len(chunks)} 个文本块 (每块~{chunk_minutes}分钟)")

    all_highlights = []

    for i, chunk in enumerate(chunks):
        logger.info(f"分析第 {i+1}/{len(chunks)} 块: "
                     f"{_format_time(chunk['start'])} - {_format_time(chunk['end'])}")

        user_prompt = HIGHLIGHT_USER_PROMPT_TEMPLATE.format(
            start_time=_format_time(chunk["start"]),
            end_time=_format_time(chunk["end"]),
            transcript_chunk=chunk["text"],
        )

        try:
            response = _call_llm(HIGHLIGHT_SYSTEM_PROMPT, user_prompt, config)
            highlights = _parse_llm_response(response, chunk["start"])

            # 填充转录文本摘录
            for h in highlights:
                h.transcript_excerpt = transcript_result.get_text_in_range(
                    h.start, h.end
                )

            all_highlights.extend(highlights)
            logger.info(f"  -> 发现 {len(highlights)} 个高光片段")

        except Exception as e:
            logger.error(f"第 {i+1} 块分析失败: {e}")
            continue

    return all_highlights


# ──────────────────────────────────────────────
# 阶段二：音频能量验证
# ──────────────────────────────────────────────

def compute_audio_energy(audio_path: str, start: float, end: float,
                         sr: int = 16000) -> float:
    """
    计算音频片段的能量得分 (0-50)
    通过 RMS 能量 + 过零率判断是否有高情绪/激动说话
    """
    try:
        import librosa

        # 只加载指定区间
        y, _ = librosa.load(audio_path, sr=sr, offset=start,
                            duration=end - start)

        if len(y) == 0:
            return 25.0  # 中性分数

        # RMS 能量
        rms = np.sqrt(np.mean(y ** 2))

        # 过零率（语速/情绪指标）
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))

        # 音量变化幅度
        rms_frames = librosa.feature.rms(y=y)[0]
        rms_std = np.std(rms_frames) if len(rms_frames) > 1 else 0

        # 归一化并映射到 0-50 分
        # 这些阈值基于直播录音的典型范围，可能需要调校
        rms_score = min(rms / 0.05, 1.0) * 20       # 0-20: 音量
        zcr_score = min(zcr / 0.15, 1.0) * 15       # 0-15: 语速/兴奋度
        dynamic_score = min(rms_std / 0.02, 1.0) * 15  # 0-15: 动态变化

        return rms_score + zcr_score + dynamic_score

    except Exception as e:
        logger.warning(f"音频能量计算失败: {e}")
        return 25.0  # 返回中性分数


# ──────────────────────────────────────────────
# 阶段二补充：弹幕热度信号 (可选，参考 hotclip)
# ──────────────────────────────────────────────

def parse_danmaku_file(danmaku_path: str) -> List[float]:
    """
    解析弹幕文件，返回弹幕时间戳列表（秒）
    支持 B站 XML 格式 (<d p="time,...">text</d>)，这是目前的事实标准，
    可通过录播姬、DanmakuRender 等工具从直播录制中导出
    """
    import xml.etree.ElementTree as ET

    timestamps = []
    try:
        tree = ET.parse(danmaku_path)
        for d in tree.iter("d"):
            p = d.get("p", "")
            first = p.split(",")[0] if p else ""
            try:
                timestamps.append(float(first))
            except ValueError:
                continue
    except (ET.ParseError, OSError) as e:
        logger.warning(f"弹幕文件解析失败: {e}")
        return []

    timestamps.sort()
    logger.info(f"弹幕解析: {len(timestamps)} 条")
    return timestamps


def compute_danmaku_score(danmaku_times: List[float],
                          start: float, end: float) -> float:
    """
    计算片段的弹幕热度得分 (0-50)
    以片段内弹幕密度相对全片平均密度的倍数计分：
    达到平均密度 = 25分，2倍 = 40分，4倍以上 = 50分封顶
    """
    if not danmaku_times:
        return 0.0

    total_span = danmaku_times[-1]
    if total_span <= 0:
        return 0.0

    in_clip = sum(1 for t in danmaku_times if start <= t < end)
    clip_density = in_clip / max(end - start, 1.0)
    avg_density = len(danmaku_times) / total_span

    if avg_density <= 0:
        return 0.0

    ratio = clip_density / avg_density
    return min(10.0 + ratio * 10.0, 50.0)


def validate_with_audio_energy(highlights: List[Highlight],
                                audio_path: str,
                                config: dict,
                                danmaku_times: Optional[List[float]] = None) -> List[Highlight]:
    """
    阶段二：用音频能量 + 弹幕热度验证和调整高光评分
    """
    hc = config.get("highlight", {})
    energy_weight = hc.get("audio_energy_weight", 0.3)
    danmaku_weight = hc.get("danmaku_weight", 0.0)

    has_danmaku = bool(danmaku_times)
    if not has_danmaku:
        danmaku_weight = 0.0

    semantic_weight = max(0.05, 1.0 - energy_weight - danmaku_weight)

    logger.info(f"评分验证: {len(highlights)} 个片段 "
                f"(语义权重={semantic_weight:.2f}, 能量权重={energy_weight:.2f}, "
                f"弹幕权重={danmaku_weight:.2f}{'' if has_danmaku else ' [无弹幕数据]'})")

    for h in highlights:
        h.energy_score = compute_audio_energy(audio_path, h.start, h.end)
        if has_danmaku:
            h.danmaku_score = compute_danmaku_score(danmaku_times, h.start, h.end)

        # 综合评分：归一化到 0-100
        semantic_normalized = (h.semantic_score / 50.0) * 100
        energy_normalized = (h.energy_score / 50.0) * 100
        danmaku_normalized = (h.danmaku_score / 50.0) * 100
        h.total_score = (
            semantic_normalized * semantic_weight +
            energy_normalized * energy_weight +
            danmaku_normalized * danmaku_weight
        )

    # 按总分排序
    highlights.sort(key=lambda h: h.total_score, reverse=True)

    for i, h in enumerate(highlights):
        score_detail = (f"语义={h.semantic_score:.0f}, 能量={h.energy_score:.1f}"
                        + (f", 弹幕={h.danmaku_score:.1f}" if has_danmaku else ""))
        logger.info(f"  #{i+1} [{_format_time(h.start)}-{_format_time(h.end)}] "
                     f"总分={h.total_score:.1f} ({score_detail}) - {h.title}")

    return highlights


# ──────────────────────────────────────────────
# 合并重叠片段
# ──────────────────────────────────────────────

def merge_overlapping(highlights: List[Highlight],
                      overlap_threshold: float = 5.0) -> List[Highlight]:
    """合并时间重叠的高光片段"""
    if len(highlights) <= 1:
        return highlights

    # 按开始时间排序
    sorted_hl = sorted(highlights, key=lambda h: h.start)
    merged = [sorted_hl[0]]

    for current in sorted_hl[1:]:
        last = merged[-1]
        if current.start < last.end + overlap_threshold:
            # 重叠，合并
            merged[-1] = Highlight(
                start=last.start,
                end=max(last.end, current.end),
                title=last.title if last.total_score >= current.total_score else current.title,
                reason=last.reason if last.total_score >= current.total_score else current.reason,
                category=last.category,
                semantic_score=max(last.semantic_score, current.semantic_score),
                energy_score=max(last.energy_score, current.energy_score),
                danmaku_score=max(last.danmaku_score, current.danmaku_score),
                total_score=max(last.total_score, current.total_score),
                transcript_excerpt=last.transcript_excerpt + " " + current.transcript_excerpt,
            )
        else:
            merged.append(current)

    logger.info(f"合并重叠: {len(highlights)} -> {len(merged)} 个片段")
    return merged


# ──────────────────────────────────────────────
# 主检测函数
# ──────────────────────────────────────────────

def detect_highlights(transcript_result, audio_path: str,
                      config: dict,
                      danmaku_file: Optional[str] = None) -> HighlightResult:
    """
    完整高光检测流程
    
    Args:
        transcript_result: TranscriptResult 对象
        audio_path: 音频文件路径
        config: 配置字典
        danmaku_file: 弹幕文件路径（可选，B站 XML 格式）
    
    Returns:
        HighlightResult
    """
    hc = config.get("highlight", {})
    min_score = hc.get("min_total_score", 30)
    min_seconds = hc.get("min_clip_seconds", 15)
    max_seconds = hc.get("max_clip_seconds", 180)

    # 阶段一：语义分析
    highlights = detect_highlights_semantic(transcript_result, config)
    logger.info(f"语义分析发现 {len(highlights)} 个候选片段")

    if not highlights:
        logger.warning("未发现任何高光片段")
        return HighlightResult(
            video_path=transcript_result.video_path,
            highlights=[],
            total_analyzed_minutes=transcript_result.duration_seconds / 60,
        )

    # 可选：解析弹幕文件
    danmaku_times = []
    if danmaku_file and os.path.exists(danmaku_file):
        danmaku_times = parse_danmaku_file(danmaku_file)

    # 阶段二：音频能量 + 弹幕热度验证
    highlights = validate_with_audio_energy(highlights, audio_path, config,
                                            danmaku_times=danmaku_times)

    # 合并重叠
    highlights = merge_overlapping(highlights)

    # 过滤
    filtered = []
    for h in highlights:
        if h.total_score < min_score:
            logger.debug(f"过滤低分片段: {h.title} (分数={h.total_score:.1f})")
            continue
        if h.duration < min_seconds:
            logger.debug(f"过滤过短片段: {h.title} ({h.duration:.0f}s)")
            continue
        if h.duration > max_seconds:
            # 截断到最大时长
            h.end = h.start + max_seconds
            logger.debug(f"截断过长片段: {h.title} -> {max_seconds}s")
        filtered.append(h)

    logger.info(f"最终保留 {len(filtered)}/{len(highlights)} 个高光片段")

    return HighlightResult(
        video_path=transcript_result.video_path,
        highlights=filtered,
        total_analyzed_minutes=transcript_result.duration_seconds / 60,
    )


# ──────────────────────────────────────────────
# CLI 入口
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if len(sys.argv) < 2:
        print("用法: python highlight_detect.py <转录JSON路径> [配置文件]")
        sys.exit(1)

    transcript_json = sys.argv[1]
    config_file = sys.argv[2] if len(sys.argv) > 2 else "config.yaml"

    # 加载转录结果
    from transcribe import TranscriptResult, Segment
    with open(transcript_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    tr = TranscriptResult(
        video_path=data["video_path"],
        engine=data["engine"],
        language=data["language"],
        duration_seconds=data["duration_seconds"],
        full_text=data["full_text"],
        segments=[Segment(**s) for s in data["segments"]],
    )

    with open(config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 提取音频用于能量分析
    from transcribe import extract_audio
    audio_path = extract_audio(tr.video_path)

    try:
        result = detect_highlights(tr, audio_path, cfg)
        out_path = os.path.join(
            cfg.get("paths", {}).get("output_dir", "output"),
            Path(tr.video_path).stem + "_highlights.json"
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        result.save_json(out_path)

        print(f"\n高光检测完成!")
        print(f"  分析时长: {result.total_analyzed_minutes:.0f} 分钟")
        print(f"  高光片段: {len(result.highlights)} 个")
        print(f"  输出: {out_path}")

    finally:
        if os.path.exists(audio_path) and "/tmp" in audio_path:
            os.unlink(audio_path)
