#!/usr/bin/env python3
"""
Xuke Clip Factory — 主 Pipeline 编排器
一键把直播录像变成多平台短视频发布包

用法:
  python pipeline.py <视频文件>                    # 处理单个视频
  python pipeline.py --scan                        # 扫描目录找新视频
  python pipeline.py --watch                       # 持续监控 + 自动处理
  python pipeline.py <视频> --steps transcribe     # 只运行转录步骤
  python pipeline.py <视频> --steps highlight      # 只运行高光检测
"""

import os
import sys
import json
import time
import logging
import argparse
import glob as glob_mod
from pathlib import Path
from datetime import datetime
from typing import Optional, List

import yaml

# 本地模块
from transcribe import transcribe_video, TranscriptResult, Segment, extract_audio
from highlight_detect import detect_highlights, HighlightResult, Highlight
from clip_editor import batch_clip, ClipResult, ClipInfo
from copywriter import generate_copywriting, save_copywriting, ClipCopywriting
from platform_adapter import generate_publish_packages, save_all_packages

logger = logging.getLogger("pipeline")


# ──────────────────────────────────────────────
# Pipeline 状态管理
# ──────────────────────────────────────────────

class PipelineState:
    """追踪 Pipeline 执行状态"""

    def __init__(self, video_path: str, output_dir: str):
        self.video_path = video_path
        self.video_stem = Path(video_path).stem
        self.output_dir = os.path.join(output_dir, self.video_stem)
        self.state_file = os.path.join(self.output_dir, "_pipeline_state.json")

        self.state = {
            "video_path": video_path,
            "video_stem": self.video_stem,
            "started_at": None,
            "completed_at": None,
            "steps": {
                "transcribe": {"status": "pending", "output": None},
                "highlight": {"status": "pending", "output": None},
                "clip": {"status": "pending", "output": None},
                "copywriting": {"status": "pending", "output": None},
                "publish": {"status": "pending", "output": None},
            },
            "summary": {},
        }

        # 如果已有状态文件，加载
        if os.path.exists(self.state_file):
            with open(self.state_file, "r", encoding="utf-8") as f:
                self.state = json.load(f)

    def update_step(self, step: str, status: str, output: Optional[str] = None):
        self.state["steps"][step]["status"] = status
        if output:
            self.state["steps"][step]["output"] = output
        self._save()

    def is_step_done(self, step: str) -> bool:
        return self.state["steps"][step]["status"] == "done"

    def get_step_output(self, step: str) -> Optional[str]:
        return self.state["steps"][step].get("output")

    def _save(self):
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────
# Pipeline 主流程
# ──────────────────────────────────────────────

def run_pipeline(video_path: str, config: dict,
                 steps: Optional[List[str]] = None,
                 skip_existing: bool = True) -> dict:
    """
    执行完整 Pipeline

    Args:
        video_path: 视频文件路径
        config: 配置字典
        steps: 要执行的步骤列表 (None=全部)
        skip_existing: 是否跳过已完成的步骤

    Returns:
        执行摘要
    """
    all_steps = ["transcribe", "highlight", "clip", "copywriting", "publish"]
    if steps:
        run_steps = [s for s in all_steps if s in steps]
    else:
        run_steps = all_steps

    output_dir = config.get("paths", {}).get("output_dir", "output")
    state = PipelineState(video_path, output_dir)
    state.state["started_at"] = datetime.now().isoformat()

    logger.info("=" * 60)
    logger.info(f"Pipeline 开始: {Path(video_path).name}")
    logger.info(f"步骤: {' -> '.join(run_steps)}")
    logger.info("=" * 60)

    summary = {
        "video": Path(video_path).name,
        "steps_run": [],
        "clips_created": 0,
        "platforms": [],
        "errors": [],
    }

    transcript_result = None
    highlights_result = None
    clip_result = None
    copywriting_list = None
    audio_path = None

    try:
        # ─── Step 1: 转录 ───
        if "transcribe" in run_steps:
            if skip_existing and state.is_step_done("transcribe"):
                logger.info("[跳过] 转录 — 已有结果")
                transcript_result = _load_transcript(state.get_step_output("transcribe"))
            else:
                logger.info("\n" + "─" * 40)
                logger.info("Step 1/5: 转录")
                logger.info("─" * 40)

                state.update_step("transcribe", "running")
                transcript_result = transcribe_video(video_path, config)

                out_path = os.path.join(state.output_dir, "transcript.json")
                os.makedirs(state.output_dir, exist_ok=True)
                transcript_result.save_json(out_path)

                state.update_step("transcribe", "done", out_path)
                summary["steps_run"].append("transcribe")
                logger.info(f"转录完成: {len(transcript_result.segments)} 段, "
                            f"{transcript_result.duration_seconds/3600:.1f}h")

        # ─── Step 2: 高光检测 ───
        if "highlight" in run_steps:
            if transcript_result is None:
                tr_path = state.get_step_output("transcribe")
                if tr_path:
                    transcript_result = _load_transcript(tr_path)
                else:
                    raise RuntimeError("需要先完成转录步骤")

            if skip_existing and state.is_step_done("highlight"):
                logger.info("[跳过] 高光检测 — 已有结果")
                highlights_result = _load_highlights(state.get_step_output("highlight"))
            else:
                logger.info("\n" + "─" * 40)
                logger.info("Step 2/5: 高光检测")
                logger.info("─" * 40)

                state.update_step("highlight", "running")

                # 提取音频用于能量分析
                audio_path = extract_audio(video_path)

                # 可选弹幕热度文件 (B站 XML 格式, 见 config.highlight.danmaku_file)
                danmaku_file = config.get("highlight", {}).get("danmaku_file", "")
                highlights_result = detect_highlights(
                    transcript_result, audio_path, config,
                    danmaku_file=danmaku_file or None,
                )

                out_path = os.path.join(state.output_dir, "highlights.json")
                highlights_result.save_json(out_path)

                state.update_step("highlight", "done", out_path)
                summary["steps_run"].append("highlight")
                logger.info(f"高光检测完成: {len(highlights_result.highlights)} 个片段")

        # ─── Step 3: 剪辑 ───
        if "clip" in run_steps:
            if highlights_result is None:
                hl_path = state.get_step_output("highlight")
                if hl_path:
                    highlights_result = _load_highlights(hl_path)
                else:
                    raise RuntimeError("需要先完成高光检测步骤")

            if transcript_result is None:
                tr_path = state.get_step_output("transcribe")
                if tr_path:
                    transcript_result = _load_transcript(tr_path)

            if skip_existing and state.is_step_done("clip"):
                logger.info("[跳过] 剪辑 — 已有结果")
                clip_result = _load_clips(state.get_step_output("clip"))
            else:
                logger.info("\n" + "─" * 40)
                logger.info("Step 3/5: 自动剪辑")
                logger.info("─" * 40)

                state.update_step("clip", "running")

                segments = transcript_result.segments if transcript_result else []
                clip_result = batch_clip(
                    highlights=highlights_result.highlights,
                    transcript_segments=segments,
                    video_path=video_path,
                    config=config,
                    output_dir=state.output_dir,
                )

                out_path = os.path.join(state.output_dir, "clips.json")
                _save_clips(clip_result, out_path)

                state.update_step("clip", "done", out_path)
                summary["steps_run"].append("clip")
                summary["clips_created"] = len(clip_result.clips)
                logger.info(f"剪辑完成: {len(clip_result.clips)} 个切片")

        # ─── Step 4: 文案 ───
        if "copywriting" in run_steps:
            if clip_result is None:
                cp_path = state.get_step_output("clip")
                if cp_path:
                    clip_result = _load_clips(cp_path)
                else:
                    raise RuntimeError("需要先完成剪辑步骤")

            if skip_existing and state.is_step_done("copywriting"):
                logger.info("[跳过] 文案 — 已有结果")
            else:
                logger.info("\n" + "─" * 40)
                logger.info("Step 4/5: AI 文案生成")
                logger.info("─" * 40)

                state.update_step("copywriting", "running")

                copywriting_list = generate_copywriting(clip_result.clips, config)

                out_path = os.path.join(state.output_dir, "copywriting.json")
                save_copywriting(copywriting_list, out_path)

                state.update_step("copywriting", "done", out_path)
                summary["steps_run"].append("copywriting")
                logger.info(f"文案完成: {len(copywriting_list)} 套")

        # ─── Step 5: 多平台发布包 ───
        if "publish" in run_steps:
            if clip_result is None:
                cp_path = state.get_step_output("clip")
                if cp_path:
                    clip_result = _load_clips(cp_path)

            if copywriting_list is None:
                cw_path = state.get_step_output("copywriting")
                if cw_path and os.path.exists(cw_path):
                    with open(cw_path, "r", encoding="utf-8") as f:
                        cw_data = json.load(f)
                    copywriting_list = _parse_copywriting_json(cw_data)
                else:
                    copywriting_list = []

            if skip_existing and state.is_step_done("publish"):
                logger.info("[跳过] 发布包 — 已有结果")
            else:
                logger.info("\n" + "─" * 40)
                logger.info("Step 5/5: 多平台适配")
                logger.info("─" * 40)

                state.update_step("publish", "running")

                packages = generate_publish_packages(
                    clips=clip_result.clips if clip_result else [],
                    copywriting_list=copywriting_list if copywriting_list else [],
                    config=config,
                    output_dir=state.output_dir,
                )

                out_path = os.path.join(state.output_dir, "publish_packages.json")
                save_all_packages(packages, out_path)

                state.update_step("publish", "done", out_path)
                summary["steps_run"].append("publish")

                enabled_platforms = [
                    k for k, v in config.get("platforms", {}).items()
                    if v.get("enabled", False)
                ]
                summary["platforms"] = enabled_platforms
                logger.info(f"发布包完成: {len(packages)} 个切片 x "
                            f"{len(enabled_platforms)} 个平台")

    except Exception as e:
        logger.error(f"Pipeline 执行失败: {e}", exc_info=True)
        summary["errors"].append(str(e))

    finally:
        # 清理临时音频
        if audio_path and os.path.exists(audio_path) and "/tmp" in audio_path:
            os.unlink(audio_path)

    state.state["completed_at"] = datetime.now().isoformat()
    state.state["summary"] = summary
    state._save()

    # 打印摘要
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline 完成!")
    logger.info(f"  视频: {summary['video']}")
    logger.info(f"  步骤: {', '.join(summary['steps_run'])}")
    logger.info(f"  切片: {summary['clips_created']} 个")
    logger.info(f"  平台: {', '.join(summary.get('platforms', []))}")
    if summary["errors"]:
        logger.warning(f"  错误: {len(summary['errors'])} 个")
    logger.info(f"  输出: {state.output_dir}")
    logger.info("=" * 60)

    return summary


# ──────────────────────────────────────────────
# 数据加载辅助函数
# ──────────────────────────────────────────────

def _load_transcript(json_path: str) -> TranscriptResult:
    """从 JSON 加载转录结果"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return TranscriptResult(
        video_path=data["video_path"],
        engine=data["engine"],
        language=data["language"],
        duration_seconds=data["duration_seconds"],
        full_text=data["full_text"],
        segments=[Segment(**s) for s in data["segments"]],
    )


def _load_highlights(json_path: str) -> HighlightResult:
    """从 JSON 加载高光结果"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return HighlightResult(
        video_path=data["video_path"],
        total_analyzed_minutes=data["total_analyzed_minutes"],
        highlights=[Highlight(**h) for h in data["highlights"]],
    )


def _load_clips(json_path: str) -> ClipResult:
    """从 JSON 加载剪辑结果"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = ClipResult()
    for c in data.get("clips", []):
        result.clips.append(ClipInfo(**c))
    result.failed = data.get("failed", [])
    return result


def _save_clips(clip_result: ClipResult, output_path: str):
    """保存剪辑结果"""
    data = {
        "clips": [
            {
                "index": c.index,
                "video_path": c.video_path,
                "start": c.start,
                "end": c.end,
                "title": c.title,
                "category": c.category,
                "score": c.score,
                "transcript_excerpt": c.transcript_excerpt,
                "output_path": c.output_path,
            }
            for c in clip_result.clips
        ],
        "failed": clip_result.failed,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _parse_copywriting_json(data: list) -> List[ClipCopywriting]:
    """从 JSON 恢复 ClipCopywriting 列表"""
    from copywriter import ClipCopywriting, PlatformCopy
    results = []
    for item in data:
        cw = ClipCopywriting(
            clip_index=item["clip_index"],
            clip_title=item["clip_title"],
            clip_category=item["clip_category"],
            transcript_excerpt=item.get("transcript_excerpt", ""),
        )
        for pk, pv in item.get("platforms", {}).items():
            cw.platforms[pk] = PlatformCopy(**pv)
        results.append(cw)
    return results


# ──────────────────────────────────────────────
# 目录扫描模式
# ──────────────────────────────────────────────

def scan_and_process(config: dict, steps: Optional[List[str]] = None):
    """扫描源目录中的视频文件并处理"""
    source_dir = config.get("paths", {}).get("source_video_dir", ".")
    output_dir = config.get("paths", {}).get("output_dir", "output")

    # 支持的视频格式
    video_exts = {".mp4", ".mkv", ".mov", ".flv", ".avi", ".ts"}

    videos = []
    for f in os.listdir(source_dir):
        if Path(f).suffix.lower() in video_exts:
            videos.append(os.path.join(source_dir, f))

    if not videos:
        logger.info(f"未在 {source_dir} 中找到视频文件")
        return

    logger.info(f"发现 {len(videos)} 个视频文件")

    for i, video in enumerate(sorted(videos)):
        video_stem = Path(video).stem
        state_file = os.path.join(output_dir, video_stem, "_pipeline_state.json")

        # 检查是否已处理完成
        if os.path.exists(state_file):
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            if state.get("completed_at"):
                logger.info(f"[{i+1}/{len(videos)}] 跳过已处理: {Path(video).name}")
                continue

        logger.info(f"\n[{i+1}/{len(videos)}] 处理: {Path(video).name}")
        run_pipeline(video, config, steps=steps)


# ──────────────────────────────────────────────
# CLI 入口
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Xuke Clip Factory — 直播切片自动化 Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python pipeline.py livestream_20240414.mp4
  python pipeline.py livestream.mp4 --steps transcribe highlight
  python pipeline.py --scan
  python pipeline.py --scan --steps transcribe
        """
    )

    parser.add_argument("video", nargs="?", help="视频文件路径")
    parser.add_argument("--config", "-c", default="config.yaml",
                        help="配置文件路径 (默认: config.yaml)")
    parser.add_argument("--scan", action="store_true",
                        help="扫描源目录处理所有新视频")
    parser.add_argument("--steps", nargs="+",
                        choices=["transcribe", "highlight", "clip",
                                 "copywriting", "publish"],
                        help="只运行指定步骤")
    parser.add_argument("--no-skip", action="store_true",
                        help="不跳过已完成的步骤（重新运行）")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="详细日志")

    args = parser.parse_args()

    # 日志设置
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_file = os.path.join(
        os.path.dirname(args.config), "logs",
        f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ]
    )

    # 加载配置
    config_path = args.config
    if not os.path.exists(config_path):
        # 尝试在脚本目录找
        script_dir = Path(__file__).parent
        config_path = str(script_dir / "config.yaml")

    if not os.path.exists(config_path):
        print(f"错误: 找不到配置文件 {args.config}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info(f"配置文件: {config_path}")

    # 执行
    if args.scan:
        scan_and_process(config, steps=args.steps)
    elif args.video:
        if not os.path.exists(args.video):
            print(f"错误: 视频文件不存在 - {args.video}")
            sys.exit(1)
        run_pipeline(
            video_path=args.video,
            config=config,
            steps=args.steps,
            skip_existing=not args.no_skip,
        )
    else:
        parser.print_help()
        print("\n提示: 请指定视频文件或使用 --scan 扫描目录")
        sys.exit(1)


if __name__ == "__main__":
    main()
