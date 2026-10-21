"""
Xuke Clip Factory — FFmpeg 自动剪辑模块
负责：视频切割、字幕生成与烧录、淡入淡出、画面裁切
"""

import os
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

import yaml

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────

@dataclass
class ClipInfo:
    """一个切片的完整信息"""
    index: int
    video_path: str            # 源视频路径
    start: float               # 起始时间（秒）
    end: float                 # 结束时间（秒）
    title: str
    category: str
    score: float
    transcript_excerpt: str = ""  # 对应的转录文本
    output_path: str = ""         # 输出路径

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class ClipResult:
    """剪辑结果"""
    clips: List[ClipInfo] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)


# ──────────────────────────────────────────────
# ASS 字幕生成
# ──────────────────────────────────────────────

def _seconds_to_ass_time(seconds: float) -> str:
    """转换为 ASS 时间格式 H:MM:SS.CC"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def generate_ass_subtitle(segments: list, clip_start: float, clip_end: float,
                          config: dict) -> str:
    """
    生成 ASS 字幕文件内容
    
    Args:
        segments: 转录 Segment 列表（带 start, end, text）
        clip_start: 切片在原视频中的起始时间
        clip_end: 切片在原视频中的结束时间
        config: 配置
    
    Returns:
        ASS 字幕文件内容字符串
    """
    sc = config.get("subtitle", {})
    font_name = sc.get("font_name", "PingFang SC")
    font_size = sc.get("font_size", 22)
    font_color = sc.get("font_color", "&H00FFFFFF")
    outline_color = sc.get("outline_color", "&H00000000")
    outline_width = sc.get("outline_width", 2)
    margin_bottom = sc.get("margin_bottom", 40)

    # ASS 头部
    header = f"""[Script Info]
Title: Xuke Clip Subtitle
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},{font_color},&H000000FF,{outline_color},&H80000000,1,0,0,0,100,100,0,0,1,{outline_width},0,2,20,20,{margin_bottom},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    # 生成对话行
    lines = []
    for seg in segments:
        # 只处理落在切片范围内的段落
        if seg.end <= clip_start or seg.start >= clip_end:
            continue

        # 调整为相对于切片开始的时间
        rel_start = max(0, seg.start - clip_start)
        rel_end = min(clip_end - clip_start, seg.end - clip_start)

        text = seg.text.strip()
        if not text:
            continue

        # 每行最多显示 16 个字，超过就换行
        if len(text) > 16:
            mid = len(text) // 2
            # 尝试在标点处断行
            for i in range(mid - 3, mid + 3):
                if i < len(text) and text[i] in "，。！？、；：":
                    mid = i + 1
                    break
            text = text[:mid] + "\\N" + text[mid:]

        start_time = _seconds_to_ass_time(rel_start)
        end_time = _seconds_to_ass_time(rel_end)
        lines.append(
            f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{text}"
        )

    return header + "\n".join(lines)


def _save_ass_file(ass_content: str) -> str:
    """将 ASS 内容写入临时文件"""
    tmp = tempfile.NamedTemporaryFile(suffix=".ass", mode="w",
                                      encoding="utf-8", delete=False)
    tmp.write(ass_content)
    tmp.close()
    return tmp.name


# ──────────────────────────────────────────────
# FFmpeg 剪辑核心
# ──────────────────────────────────────────────

def clip_video(video_path: str, start: float, end: float,
               output_path: str, config: dict,
               subtitle_file: Optional[str] = None,
               source_crop: Optional[str] = None,
               overlay_image: Optional[str] = None) -> bool:
    """
    使用 FFmpeg 切割视频
    
    Args:
        video_path: 源视频路径
        start: 开始时间（秒）
        end: 结束时间（秒）
        output_path: 输出路径
        config: 配置
        subtitle_file: ASS 字幕文件路径（可选）
        source_crop: 源视频裁切参数 "w:h:x:y"（可选，用于去除黑边/浏览器等）
        overlay_image: PNG overlay 图片路径（可选，用于叠加头像/关注按钮等）
    
    Returns:
        是否成功
    """
    cc = config.get("clipping", {})
    codec = cc.get("video_codec", "libx264")
    crf = cc.get("video_crf", 20)
    acodec = cc.get("audio_codec", "aac")
    abitrate = cc.get("audio_bitrate", "128k")
    fade_in = cc.get("fade_in", 0.5)
    fade_out = cc.get("fade_out", 0.5)
    padding_before = cc.get("padding_before", 2.0)
    padding_after = cc.get("padding_after", 1.5)

    # 加上前后 padding
    actual_start = max(0, start - padding_before)
    actual_end = end + padding_after
    duration = actual_end - actual_start

    # --- 判断是否需要 filter_complex（有 overlay 时）---
    use_filter_complex = overlay_image and os.path.exists(overlay_image)

    if use_filter_complex:
        # ─── filter_complex 模式：支持 crop + overlay + fade ───
        vfilters = []

        # 1) 源视频裁切（去除黑边/浏览器/Dock）
        if source_crop:
            vfilters.append(f"crop={source_crop}")

        # 2) 淡入淡出
        if fade_in > 0:
            vfilters.append(f"fade=t=in:st=0:d={fade_in}")
        if fade_out > 0:
            fade_start = max(0, duration - fade_out)
            vfilters.append(f"fade=t=out:st={fade_start}:d={fade_out}")

        # 3) 字幕烧录
        if subtitle_file:
            check = subprocess.run(["ffmpeg", "-filters"], capture_output=True, text=True)
            has_ass = any(line.strip().startswith("T.") and "ass" == line.strip().split()[-1]
                          for line in check.stdout.splitlines())
            if has_ass:
                safe_path = subtitle_file.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
                vfilters.append(f"ass={safe_path}")
            else:
                logger.warning("FFmpeg 未编译 libass，跳过字幕烧录")

        # 构建 filter_complex: [0:v]crop→fade→...[base]; [1:v]scale[ovr]; [base][ovr]overlay
        video_chain = ",".join(vfilters) if vfilters else "null"
        filter_complex = (
            f"[0:v]{video_chain}[base];"
            f"[1:v]format=rgba[ovr];"
            f"[base][ovr]overlay=20:(main_h-overlay_h-30)"
        )

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(actual_start),
            "-i", video_path,
            "-i", overlay_image,
            "-t", str(duration),
            "-filter_complex", filter_complex,
            "-map", "[base]",  # 将在下面替换
            "-c:v", codec,
            "-crf", str(crf),
            "-preset", "medium",
            "-c:a", acodec,
            "-b:a", abitrate,
            "-map", "0:a?",
        ]
        # 修正 map：overlay 输出没有标签名，需要用最终输出
        # 重新构建：让 overlay 输出到 [vout]
        filter_complex = (
            f"[0:v]{video_chain}[base];"
            f"[1:v]format=rgba[ovr];"
            f"[base][ovr]overlay=20:(main_h-overlay_h-30)[vout]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(actual_start),
            "-i", video_path,
            "-i", overlay_image,
            "-t", str(duration),
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            "-map", "0:a?",
            "-c:v", codec,
            "-crf", str(crf),
            "-preset", "medium",
            "-c:a", acodec,
            "-b:a", abitrate,
        ]

    else:
        # ─── 简单 -vf 模式（无 overlay）───
        filters = []

        # 1) 源视频裁切
        if source_crop:
            filters.append(f"crop={source_crop}")

        # 2) 淡入淡出
        if fade_in > 0:
            filters.append(f"fade=t=in:st=0:d={fade_in}")
        if fade_out > 0:
            fade_start = max(0, duration - fade_out)
            filters.append(f"fade=t=out:st={fade_start}:d={fade_out}")

        # 3) 字幕烧录
        if subtitle_file:
            check = subprocess.run(["ffmpeg", "-filters"], capture_output=True, text=True)
            has_ass = any(line.strip().startswith("T.") and "ass" == line.strip().split()[-1]
                          for line in check.stdout.splitlines())
            if has_ass:
                safe_path = subtitle_file.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
                filters.append(f"ass={safe_path}")
            else:
                logger.warning("FFmpeg 未编译 libass，跳过字幕烧录")

        vf = ",".join(filters) if filters else None

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(actual_start),
            "-i", video_path,
            "-t", str(duration),
            "-c:v", codec,
            "-crf", str(crf),
            "-preset", "medium",
            "-c:a", acodec,
            "-b:a", abitrate,
        ]

        if vf:
            cmd.extend(["-vf", vf])

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd.append(output_path)

    logger.info(f"FFmpeg 剪辑: {Path(video_path).name} "
                f"[{_format_time(actual_start)}-{_format_time(actual_end)}] "
                f"-> {Path(output_path).name}"
                f"{' [crop=' + source_crop + ']' if source_crop else ''}"
                f"{' [+overlay]' if use_filter_complex else ''}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"FFmpeg 失败:\n{result.stderr[-500:]}")
        return False

    # 验证输出文件
    if not os.path.exists(output_path) or os.path.getsize(output_path) < 1024:
        logger.error(f"输出文件异常: {output_path}")
        return False

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"剪辑成功: {Path(output_path).name} ({file_size_mb:.1f}MB, {duration:.0f}s)")
    return True


def _format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ──────────────────────────────────────────────
# 批量剪辑
# ──────────────────────────────────────────────

def batch_clip(highlights: list, transcript_segments: list,
               video_path: str, config: dict,
               output_dir: Optional[str] = None,
               source_crop: Optional[str] = None,
               overlay_image: Optional[str] = None) -> ClipResult:
    """
    批量剪辑所有高光片段
    
    Args:
        highlights: Highlight 列表
        transcript_segments: Segment 列表（用于生成字幕）
        video_path: 源视频路径
        config: 配置
        output_dir: 输出目录
        source_crop: 源视频裁切参数 "w:h:x:y"（可选）
        overlay_image: PNG overlay 图片路径（可选）
    
    Returns:
        ClipResult
    """
    if output_dir is None:
        output_dir = config.get("paths", {}).get("output_dir", "output")

    video_stem = Path(video_path).stem
    result = ClipResult()

    logger.info(f"开始批量剪辑: {len(highlights)} 个片段"
                f"{' [crop=' + source_crop + ']' if source_crop else ''}"
                f"{' [+overlay]' if overlay_image else ''}")

    for i, hl in enumerate(highlights):
        clip_index = i + 1
        safe_title = "".join(c for c in hl.title if c.isalnum() or c in "_ -")[:20]
        output_filename = f"{video_stem}_clip{clip_index:02d}_{safe_title}.mp4"
        output_path = os.path.join(output_dir, output_filename)

        # 生成字幕
        ass_content = generate_ass_subtitle(
            transcript_segments, hl.start, hl.end, config
        )
        ass_file = _save_ass_file(ass_content)

        try:
            success = clip_video(
                video_path=video_path,
                start=hl.start,
                end=hl.end,
                output_path=output_path,
                config=config,
                subtitle_file=ass_file,
                source_crop=source_crop,
                overlay_image=overlay_image,
            )

            if success:
                clip_info = ClipInfo(
                    index=clip_index,
                    video_path=video_path,
                    start=hl.start,
                    end=hl.end,
                    title=hl.title,
                    category=hl.category,
                    score=hl.total_score,
                    transcript_excerpt=hl.transcript_excerpt,
                    output_path=output_path,
                )
                result.clips.append(clip_info)
            else:
                result.failed.append(f"Clip {clip_index}: {hl.title}")

        finally:
            # 清理临时字幕文件
            if os.path.exists(ass_file):
                os.unlink(ass_file)

    logger.info(f"批量剪辑完成: 成功 {len(result.clips)}, 失败 {len(result.failed)}")
    return result


# ──────────────────────────────────────────────
# 画面裁切（竖屏适配用）
# ──────────────────────────────────────────────

def crop_to_aspect(input_path: str, output_path: str,
                   target_width: int, target_height: int,
                   config: dict) -> bool:
    """
    将视频裁切/缩放到目标尺寸
    用于从 16:9 横屏 -> 9:16 竖屏等场景
    策略: 中心裁切 + 缩放
    """
    cc = config.get("clipping", {})
    codec = cc.get("video_codec", "libx264")
    crf = cc.get("video_crf", 20)

    # 先获取源视频尺寸
    probe_cmd = [
        "ffprobe", "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x",
        input_path
    ]
    probe = subprocess.run(probe_cmd, capture_output=True, text=True)
    if probe.returncode != 0:
        logger.error(f"无法获取视频尺寸: {input_path}")
        return False

    src_w, src_h = map(int, probe.stdout.strip().split("x"))
    src_ratio = src_w / src_h
    tgt_ratio = target_width / target_height

    if abs(src_ratio - tgt_ratio) < 0.01:
        # 宽高比接近，直接缩放
        vf = f"scale={target_width}:{target_height}"
    elif src_ratio > tgt_ratio:
        # 源更宽，需要左右裁切（适合人物居中的直播画面）
        crop_w = int(src_h * tgt_ratio)
        crop_x = (src_w - crop_w) // 2
        vf = f"crop={crop_w}:{src_h}:{crop_x}:0,scale={target_width}:{target_height}"
    else:
        # 源更高，上下裁切
        crop_h = int(src_w / tgt_ratio)
        crop_y = (src_h - crop_h) // 2
        vf = f"crop={src_w}:{crop_h}:0:{crop_y},scale={target_width}:{target_height}"

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", vf,
        "-c:v", codec,
        "-crf", str(crf),
        "-preset", "medium",
        "-c:a", "copy",
        output_path
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"画面裁切失败:\n{result.stderr[-500:]}")
        return False

    logger.info(f"画面裁切完成: {src_w}x{src_h} -> {target_width}x{target_height}")
    return True


# ──────────────────────────────────────────────
# 多片段合并/拼接
# ──────────────────────────────────────────────

def concatenate_clips(clip_paths: List[str], output_path: str,
                      config: dict,
                      transition: str = "fade",
                      transition_duration: float = 0.5) -> bool:
    """
    将多个视频片段按顺序拼接成一个视频
    
    Args:
        clip_paths: 要拼接的视频文件路径列表（按顺序）
        output_path: 输出路径
        config: 配置
        transition: 转场类型 "none" / "fade" / "xfade"
        transition_duration: 转场时长（秒）
    
    Returns:
        是否成功
    """
    if not clip_paths:
        logger.error("拼接列表为空")
        return False

    # 全部转为绝对路径（避免 concat demuxer 路径解析问题）
    clip_paths = [os.path.abspath(p) for p in clip_paths]

    # 过滤不存在的文件
    valid_paths = [p for p in clip_paths if os.path.exists(p)]
    if len(valid_paths) != len(clip_paths):
        missing = set(clip_paths) - set(valid_paths)
        logger.warning(f"跳过 {len(missing)} 个不存在的文件: {missing}")

    if len(valid_paths) == 0:
        logger.error("没有有效的视频文件可拼接")
        return False

    if len(valid_paths) == 1:
        # 只有一个文件，直接复制
        import shutil
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        shutil.copy2(valid_paths[0], output_path)
        logger.info(f"只有一个片段，直接复制: {output_path}")
        return True

    cc = config.get("clipping", {})
    codec = cc.get("video_codec", "libx264")
    crf = cc.get("video_crf", 20)

    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if transition == "none":
        # ─── 无转场：用 concat demuxer（最快）───
        return _concat_no_transition(valid_paths, output_path, codec, crf)

    elif transition == "xfade" and len(valid_paths) <= 10:
        # ─── xfade 交叉转场（适合少量片段）───
        return _concat_xfade(valid_paths, output_path, codec, crf,
                             transition_duration)

    else:
        # ─── fade: 每段淡入淡出后拼接 ───
        return _concat_with_fade(valid_paths, output_path, codec, crf,
                                  transition_duration)


def _concat_no_transition(clip_paths: List[str], output_path: str,
                           codec: str, crf: int) -> bool:
    """无转场直接拼接（concat demuxer）"""
    # 创建 concat 列表文件
    list_file = output_path + ".concat.txt"
    try:
        with open(list_file, "w", encoding="utf-8") as f:
            for path in clip_paths:
                # concat demuxer 需要转义单引号
                safe = path.replace("'", "'\\''")
                f.write(f"file '{safe}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", list_file,
            "-c:v", codec,
            "-crf", str(crf),
            "-preset", "medium",
            "-c:a", "aac",
            "-b:a", "128k",
            output_path
        ]

        logger.info(f"无转场拼接: {len(clip_paths)} 个片段")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"拼接失败:\n{result.stderr[-500:]}")
            return False

        _log_output_info(output_path, clip_paths)
        return True

    finally:
        if os.path.exists(list_file):
            os.unlink(list_file)


def _concat_xfade(clip_paths: List[str], output_path: str,
                   codec: str, crf: int,
                   xfade_duration: float = 0.5) -> bool:
    """使用 xfade 滤镜做交叉转场拼接"""
    n = len(clip_paths)

    # 获取每段时长
    durations = []
    for p in clip_paths:
        cmd = ["ffprobe", "-v", "quiet",
               "-show_entries", "format=duration",
               "-of", "default=noprint_wrappers=1:nokey=1", p]
        r = subprocess.run(cmd, capture_output=True, text=True)
        durations.append(float(r.stdout.strip()) if r.returncode == 0 else 10.0)

    # 构建 filter_complex
    inputs = []
    for i, p in enumerate(clip_paths):
        inputs.extend(["-i", p])

    # xfade 链式串联: [0][1]xfade -> [v01], [v01][2]xfade -> [v012], ...
    filters = []
    offsets = []
    cumulative = 0.0

    for i in range(n - 1):
        if i == 0:
            offset = durations[0] - xfade_duration
        else:
            offset = offsets[-1] + durations[i] - xfade_duration
        offsets.append(offset)

        in1 = f"[v{i}]" if i > 0 else f"[{i}:v]"
        in2 = f"[{i+1}:v]"
        out = f"[v{i+1}]" if i < n - 2 else "[vout]"

        filters.append(
            f"{in1}{in2}xfade=transition=fade:duration={xfade_duration}"
            f":offset={offset:.3f}{out}"
        )

    # 音频也做 crossfade
    audio_filters = []
    for i in range(n - 1):
        in1 = f"[a{i}]" if i > 0 else f"[{i}:a]"
        in2 = f"[{i+1}:a]"
        out = f"[a{i+1}]" if i < n - 2 else "[aout]"
        audio_filters.append(
            f"{in1}{in2}acrossfade=d={xfade_duration}:c1=tri:c2=tri{out}"
        )

    filter_complex = ";".join(filters + audio_filters)

    cmd = ["ffmpeg", "-y"] + inputs + [
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "[aout]",
        "-c:v", codec,
        "-crf", str(crf),
        "-preset", "medium",
        "-c:a", "aac",
        "-b:a", "128k",
        output_path
    ]

    logger.info(f"xfade 交叉转场拼接: {n} 个片段, 转场 {xfade_duration}s")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.warning(f"xfade 拼接失败，回退到无转场模式:\n{result.stderr[-300:]}")
        return _concat_no_transition(clip_paths, output_path, codec, crf)

    _log_output_info(output_path, clip_paths)
    return True


def _concat_with_fade(clip_paths: List[str], output_path: str,
                       codec: str, crf: int,
                       fade_duration: float = 0.5) -> bool:
    """
    每段单独加淡入淡出，然后用 concat demuxer 拼接
    适合大量片段（比 xfade 稳定）
    """
    temp_files = []

    try:
        for i, path in enumerate(clip_paths):
            tmp_out = output_path + f".fade_{i:03d}.mp4"
            temp_files.append(tmp_out)

            # 获取时长
            cmd = ["ffprobe", "-v", "quiet",
                   "-show_entries", "format=duration",
                   "-of", "default=noprint_wrappers=1:nokey=1", path]
            r = subprocess.run(cmd, capture_output=True, text=True)
            dur = float(r.stdout.strip()) if r.returncode == 0 else 10.0

            fade_out_start = max(0, dur - fade_duration)

            vf = (f"fade=t=in:st=0:d={fade_duration},"
                  f"fade=t=out:st={fade_out_start}:d={fade_duration}")
            af = (f"afade=t=in:st=0:d={fade_duration},"
                  f"afade=t=out:st={fade_out_start}:d={fade_duration}")

            cmd = [
                "ffmpeg", "-y", "-i", path,
                "-vf", vf, "-af", af,
                "-c:v", codec, "-crf", str(crf), "-preset", "fast",
                "-c:a", "aac", "-b:a", "128k",
                tmp_out
            ]
            subprocess.run(cmd, capture_output=True, text=True)

        # 拼接所有淡入淡出片段
        return _concat_no_transition(temp_files, output_path, codec, crf)

    finally:
        for tf in temp_files:
            if os.path.exists(tf):
                os.unlink(tf)


def _log_output_info(output_path: str, source_paths: List[str]):
    """记录拼接输出信息"""
    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        cmd = ["ffprobe", "-v", "quiet",
               "-show_entries", "format=duration",
               "-of", "default=noprint_wrappers=1:nokey=1", output_path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        dur = float(r.stdout.strip()) if r.returncode == 0 else 0
        logger.info(f"拼接完成: {len(source_paths)} 段 -> "
                     f"{Path(output_path).name} ({dur:.0f}s, {size_mb:.1f}MB)")


def merge_highlights_compilation(clips: List[ClipInfo], output_path: str,
                                  config: dict,
                                  max_total_minutes: float = 10,
                                  transition: str = "xfade") -> bool:
    """
    将多个高光切片合并成一个精华合集视频
    
    Args:
        clips: ClipInfo 列表（已按分数排序）
        output_path: 输出路径
        config: 配置
        max_total_minutes: 合集最大时长（分钟）
        transition: 转场类型
    
    Returns:
        是否成功
    """
    # 按分数排序，取最好的片段直到填满时长
    sorted_clips = sorted(clips, key=lambda c: c.score, reverse=True)
    selected = []
    total_duration = 0.0

    for clip in sorted_clips:
        if not os.path.exists(clip.output_path):
            continue
        if total_duration + clip.duration > max_total_minutes * 60:
            break
        selected.append(clip)
        total_duration += clip.duration

    if not selected:
        logger.warning("没有可用的切片来生成合集")
        return False

    # 按时间顺序重排（合集中保持原直播的时间线）
    selected.sort(key=lambda c: c.start)
    clip_paths = [c.output_path for c in selected]

    logger.info(f"生成精华合集: {len(selected)} 个片段, "
                f"总时长 {total_duration/60:.1f} 分钟")

    return concatenate_clips(clip_paths, output_path, config,
                             transition=transition)


# ──────────────────────────────────────────────
# CLI 入口
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    print("clip_editor 模块 — 需要通过 pipeline.py 调用或传入高光 JSON")
    print("用法示例: python clip_editor.py <高光JSON> <转录JSON> <源视频> [配置文件]")

    if len(sys.argv) >= 4:
        highlight_json = sys.argv[1]
        transcript_json = sys.argv[2]
        video_file = sys.argv[3]
        config_file = sys.argv[4] if len(sys.argv) > 4 else "config.yaml"

        from highlight_detect import Highlight
        from transcribe import Segment

        with open(highlight_json, "r", encoding="utf-8") as f:
            hl_data = json.load(f)

        with open(transcript_json, "r", encoding="utf-8") as f:
            tr_data = json.load(f)

        with open(config_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        highlights = [Highlight(**h) for h in hl_data["highlights"]]
        segments = [Segment(**s) for s in tr_data["segments"]]

        result = batch_clip(highlights, segments, video_file, cfg)
        print(f"\n剪辑完成: 成功 {len(result.clips)}, 失败 {len(result.failed)}")
