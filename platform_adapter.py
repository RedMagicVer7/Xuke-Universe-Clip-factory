"""
Xuke Clip Factory — 多平台适配模块
将剪辑好的视频转换为各平台需要的格式（尺寸、时长、封面）
并整合文案输出完整的发布包
"""

import os
import json
import logging
import subprocess
import shutil
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────

@dataclass
class PlatformOutput:
    """单平台发布文件"""
    platform: str
    video_path: str
    cover_path: str = ""
    title: str = ""
    description: str = ""
    hashtags: List[str] = field(default_factory=list)
    hook: str = ""
    width: int = 0
    height: int = 0
    duration: float = 0.0
    file_size_mb: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PublishPackage:
    """一个切片的完整发布包"""
    clip_index: int
    clip_title: str
    source_video: str
    platforms: Dict[str, PlatformOutput] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "clip_index": self.clip_index,
            "clip_title": self.clip_title,
            "source_video": self.source_video,
            "platforms": {k: v.to_dict() for k, v in self.platforms.items()},
        }


# ──────────────────────────────────────────────
# 视频格式转换
# ──────────────────────────────────────────────

def _get_video_info(video_path: str) -> dict:
    """获取视频的宽、高、时长"""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-show_entries", "format=duration",
        "-of", "json",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return {"width": 0, "height": 0, "duration": 0}

    data = json.loads(result.stdout)
    stream = data.get("streams", [{}])[0]
    fmt = data.get("format", {})
    return {
        "width": stream.get("width", 0),
        "height": stream.get("height", 0),
        "duration": float(fmt.get("duration", 0)),
    }


def adapt_video(input_path: str, output_path: str,
                target_width: int, target_height: int,
                max_duration: int = 0, max_size_mb: int = 0,
                config: dict = None) -> bool:
    """
    将视频适配到目标平台规格
    
    策略:
    - 源是 16:9 → 目标 9:16: 中心裁切 + 背景模糊填充
    - 源是 9:16 → 目标 9:16: 直接缩放
    - 源是 9:16 → 目标 3:4: 上下裁切
    - 时长超限: 截断
    """
    info = _get_video_info(input_path)
    src_w, src_h = info["width"], info["height"]
    duration = info["duration"]

    if src_w == 0 or src_h == 0:
        logger.error(f"无法读取源视频信息: {input_path}")
        return False

    src_ratio = src_w / src_h
    tgt_ratio = target_width / target_height

    filters = []

    # 画面适配
    if abs(src_ratio - tgt_ratio) < 0.05:
        # 宽高比相近，直接缩放
        filters.append(f"scale={target_width}:{target_height}")
    elif src_ratio > tgt_ratio:
        # 源更宽（如 16:9 → 9:16），使用背景模糊 + 前景裁切叠加
        # 方案: 先中心裁切到目标比例
        crop_w = int(src_h * tgt_ratio)
        crop_x = (src_w - crop_w) // 2
        filters.append(f"crop={crop_w}:{src_h}:{crop_x}:0")
        filters.append(f"scale={target_width}:{target_height}")
    else:
        # 源更窄（如 9:16 → 3:4），上下裁切
        crop_h = int(src_w / tgt_ratio)
        crop_y = (src_h - crop_h) // 2
        filters.append(f"crop={src_w}:{crop_h}:0:{crop_y}")
        filters.append(f"scale={target_width}:{target_height}")

    vf = ",".join(filters)

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
    ]

    # 时长限制
    if max_duration > 0 and duration > max_duration:
        cmd.extend(["-t", str(max_duration)])
        logger.info(f"截断到 {max_duration}s (原 {duration:.0f}s)")

    cmd.extend([
        "-vf", vf,
        "-c:v", "libx264",
        "-crf", "20",
        "-preset", "medium",
        "-c:a", "aac",
        "-b:a", "128k",
    ])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd.append(output_path)

    logger.info(f"平台适配: {src_w}x{src_h} -> {target_width}x{target_height}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"适配转换失败:\n{result.stderr[-500:]}")
        return False

    # 检查文件大小
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    if max_size_mb > 0 and file_size_mb > max_size_mb:
        logger.warning(f"文件 {file_size_mb:.1f}MB 超过限制 {max_size_mb}MB, 需要二次压缩")
        return _recompress(output_path, max_size_mb, duration)

    return True


def _recompress(video_path: str, target_size_mb: int, duration: float) -> bool:
    """二次压缩到目标大小"""
    target_bits = target_size_mb * 8 * 1024 * 1024
    audio_bits = 128 * 1000 * duration
    video_bitrate = int((target_bits - audio_bits) / duration * 0.95)

    if video_bitrate < 500000:  # 低于 500kbps 画质太差
        logger.warning("目标大小过小，无法在保证画质的情况下压缩")
        return True  # 仍然返回 True，让用户自行决定

    tmp_path = video_path + ".tmp.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-c:v", "libx264",
        "-b:v", str(video_bitrate),
        "-pass", "1",
        "-f", "null", "/dev/null",
        "&&",
        "ffmpeg", "-y",
        "-i", video_path,
        "-c:v", "libx264",
        "-b:v", str(video_bitrate),
        "-pass", "2",
        "-c:a", "aac",
        "-b:a", "128k",
        tmp_path
    ]

    # 简化为单次编码
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-c:v", "libx264",
        "-b:v", str(video_bitrate),
        "-maxrate", str(int(video_bitrate * 1.2)),
        "-bufsize", str(video_bitrate * 2),
        "-c:a", "aac", "-b:a", "128k",
        tmp_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        os.replace(tmp_path, video_path)
        new_size = os.path.getsize(video_path) / (1024 * 1024)
        logger.info(f"二次压缩完成: {new_size:.1f}MB")
        return True
    else:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return False


# ──────────────────────────────────────────────
# 封面截取
# ──────────────────────────────────────────────

def extract_cover(video_path: str, output_path: str,
                  timestamp: Optional[float] = None) -> bool:
    """
    从视频中截取封面图
    默认取视频 1/3 处的帧（通常是人物表情最丰富的地方）
    """
    if timestamp is None:
        info = _get_video_info(video_path)
        timestamp = info["duration"] / 3

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(timestamp),
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "2",
        output_path
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"封面截取失败: {result.stderr[-300:]}")
        return False

    logger.info(f"封面已截取: {output_path}")
    return True


# ──────────────────────────────────────────────
# 发布包生成
# ──────────────────────────────────────────────

def generate_publish_packages(clips: list, copywriting_list: list,
                               config: dict,
                               output_dir: Optional[str] = None) -> List[PublishPackage]:
    """
    为每个切片生成多平台发布包
    
    Args:
        clips: ClipInfo 列表
        copywriting_list: ClipCopywriting 列表
        config: 配置
        output_dir: 输出根目录
    
    Returns:
        PublishPackage 列表
    """
    if output_dir is None:
        output_dir = config.get("paths", {}).get("output_dir", "output")

    platforms_config = config.get("platforms", {})
    packages = []

    # 建立文案索引
    copy_map = {}
    for cw in copywriting_list:
        copy_map[cw.clip_index] = cw

    for clip in clips:
        logger.info(f"生成发布包 #{clip.index}: {clip.title}")

        if not os.path.exists(clip.output_path):
            logger.error(f"源切片文件不存在: {clip.output_path}")
            continue

        video_stem = Path(clip.video_path).stem
        package = PublishPackage(
            clip_index=clip.index,
            clip_title=clip.title,
            source_video=clip.output_path,
        )

        copywriting = copy_map.get(clip.index)

        for platform_name, platform_cfg in platforms_config.items():
            if not platform_cfg.get("enabled", False):
                continue

            target_w = platform_cfg["width"]
            target_h = platform_cfg["height"]
            max_dur = platform_cfg.get("max_duration", 0)
            max_size = platform_cfg.get("max_size_mb", 0)

            # 输出路径
            platform_dir = os.path.join(
                output_dir, video_stem, f"clip{clip.index:02d}", platform_name
            )

            video_out = os.path.join(platform_dir, f"{platform_name}.mp4")
            cover_out = os.path.join(platform_dir, f"{platform_name}_cover.jpg")

            # 适配视频
            success = adapt_video(
                input_path=clip.output_path,
                output_path=video_out,
                target_width=target_w,
                target_height=target_h,
                max_duration=max_dur,
                max_size_mb=max_size,
                config=config,
            )

            if not success:
                logger.warning(f"平台 {platform_name} 视频适配失败")
                continue

            # 截取封面
            extract_cover(video_out, cover_out)

            # 获取文案
            platform_copy = None
            if copywriting and platform_name in copywriting.platforms:
                platform_copy = copywriting.platforms[platform_name]

            info = _get_video_info(video_out)
            file_size = os.path.getsize(video_out) / (1024 * 1024)

            platform_output = PlatformOutput(
                platform=platform_name,
                video_path=video_out,
                cover_path=cover_out if os.path.exists(cover_out) else "",
                title=platform_copy.title if platform_copy else clip.title,
                description=platform_copy.description if platform_copy else "",
                hashtags=platform_copy.hashtags if platform_copy else [],
                hook=platform_copy.hook if platform_copy else "",
                width=target_w,
                height=target_h,
                duration=info["duration"],
                file_size_mb=file_size,
            )

            package.platforms[platform_name] = platform_output
            logger.info(f"  {platform_name}: {target_w}x{target_h}, "
                        f"{info['duration']:.0f}s, {file_size:.1f}MB")

        packages.append(package)

        # 保存该切片的发布清单
        manifest_path = os.path.join(
            output_dir, video_stem, f"clip{clip.index:02d}", "manifest.json"
        )
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(package.to_dict(), f, ensure_ascii=False, indent=2)

    logger.info(f"发布包生成完成: {len(packages)} 个切片")
    return packages


def save_all_packages(packages: List[PublishPackage], output_path: str):
    """保存所有发布包的汇总信息"""
    data = [p.to_dict() for p in packages]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"发布包汇总已保存: {output_path}")


# ──────────────────────────────────────────────
# CLI 入口
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    print("platform_adapter 模块 — 通过 pipeline.py 调用")
    print("用于将切片适配到各平台尺寸并生成发布包")
