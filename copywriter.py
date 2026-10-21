"""
Xuke Clip Factory — AI 文案生成模块
为每个切片生成多平台发布文案（标题、描述、标签）
"""

import os
import json
import logging
import re
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────

@dataclass
class PlatformCopy:
    """单平台文案"""
    platform: str        # shipin_hao / douyin / xiaohongshu / bilibili / x_twitter
    title: str           # 标题
    description: str     # 描述/正文
    hashtags: List[str] = field(default_factory=list)  # 话题标签
    hook: str = ""       # 开头钩子（前3秒文案）

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ClipCopywriting:
    """一个切片的完整文案"""
    clip_index: int
    clip_title: str
    clip_category: str
    transcript_excerpt: str
    platforms: Dict[str, PlatformCopy] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "clip_index": self.clip_index,
            "clip_title": self.clip_title,
            "clip_category": self.clip_category,
            "transcript_excerpt": self.transcript_excerpt[:200],
            "platforms": {k: v.to_dict() for k, v in self.platforms.items()},
        }


# ──────────────────────────────────────────────
# 文案生成 Prompt
# ──────────────────────────────────────────────

COPYWRITING_SYSTEM_PROMPT = """你是徐可的内容运营助手，负责为她的直播切片生成多平台发布文案。

徐可的人设：
- 视频号情感类头部主播，粉丝称她"祖师奶"
- 风格犀利、直白、不装，说话辛辣有力
- 经历丰富：离过婚、带女儿、做过生意
- 核心价值观：女性独立、活出自我、不内耗
- 金句频出，擅长一句话点醒人

文案原则：
1. 保持徐可的语言风格，不要美化成鸡汤
2. 标题必须有冲击力，3秒内抓住注意力
3. 用口语化表达，不要文绉绉
4. 话题标签要蹭热度但不低俗
5. 不同平台的文案风格要有区别"""

COPYWRITING_USER_PROMPT = """以下是一个直播切片的信息：

标题：{title}
分类：{category}
内容摘要：
{transcript}

请为以下平台各生成一套文案。严格按 JSON 格式输出：

```json
{{
  "shipin_hao": {{
    "title": "视频号标题（不超过20字）",
    "description": "视频号描述（2-3句话，可以用emoji）",
    "hashtags": ["#话题1", "#话题2", "#话题3"],
    "hook": "前3秒文案（一句话，让人忍不住看下去）"
  }},
  "douyin": {{
    "title": "抖音标题（不超过20字，可以更夸张）",
    "description": "抖音描述（短小精悍，1-2句）",
    "hashtags": ["#话题1", "#话题2", "#话题3", "#话题4", "#话题5"],
    "hook": "前3秒文案"
  }},
  "xiaohongshu": {{
    "title": "小红书标题（加emoji，15字左右）",
    "description": "小红书正文（分段，可以长一点，3-5句）",
    "hashtags": ["#话题1", "#话题2", "#话题3"],
    "hook": "前3秒文案"
  }}
}}
```

只返回 JSON，不要其他解释。"""


# ──────────────────────────────────────────────
# LLM 调用
# ──────────────────────────────────────────────

def _call_llm(system_prompt: str, user_prompt: str, config: dict) -> str:
    """调用 LLM API"""
    cc = config.get("copywriting", {})
    provider = cc.get("llm_provider", "openai")
    model = cc.get("llm_model", "gpt-4.1-mini")
    api_key_env = cc.get("llm_api_key_env",
                          config.get("highlight", {}).get("llm_api_key_env", "OPENAI_API_KEY"))
    api_key = os.environ.get(api_key_env, "")

    if not api_key:
        raise ValueError(f"未找到 API Key: {api_key_env}")

    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=2048,
        )
        return response.choices[0].message.content

    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text

    else:
        raise ValueError(f"不支持的 LLM 提供商: {provider}")


def _parse_copywriting_response(response_text: str) -> Dict[str, PlatformCopy]:
    """解析 LLM 返回的文案 JSON"""
    # 提取 JSON
    json_match = re.search(r'\{[\s\S]*\}', response_text)
    if not json_match:
        logger.warning("文案 LLM 返回中未找到有效 JSON")
        return {}

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        logger.warning(f"文案 JSON 解析失败: {e}")
        return {}

    platforms = {}
    for platform_key, copy_data in data.items():
        if isinstance(copy_data, dict):
            platforms[platform_key] = PlatformCopy(
                platform=platform_key,
                title=copy_data.get("title", ""),
                description=copy_data.get("description", ""),
                hashtags=copy_data.get("hashtags", []),
                hook=copy_data.get("hook", ""),
            )

    return platforms


# ──────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────

def generate_copywriting(clips: list, config: dict) -> List[ClipCopywriting]:
    """
    为所有切片生成文案
    
    Args:
        clips: ClipInfo 列表（来自 clip_editor）
        config: 配置字典
    
    Returns:
        ClipCopywriting 列表
    """
    results = []

    logger.info(f"开始生成文案: {len(clips)} 个切片")

    for clip in clips:
        logger.info(f"生成文案 #{clip.index}: {clip.title}")

        user_prompt = COPYWRITING_USER_PROMPT.format(
            title=clip.title,
            category=clip.category,
            transcript=clip.transcript_excerpt[:500],
        )

        try:
            response = _call_llm(COPYWRITING_SYSTEM_PROMPT, user_prompt, config)
            platforms = _parse_copywriting_response(response)

            copywriting = ClipCopywriting(
                clip_index=clip.index,
                clip_title=clip.title,
                clip_category=clip.category,
                transcript_excerpt=clip.transcript_excerpt,
                platforms=platforms,
            )
            results.append(copywriting)

            platform_names = ", ".join(platforms.keys())
            logger.info(f"  -> 已生成 {len(platforms)} 个平台文案: {platform_names}")

        except Exception as e:
            logger.error(f"文案生成失败 #{clip.index}: {e}")
            # 生成兜底文案
            results.append(_fallback_copywriting(clip))

    return results


def _fallback_copywriting(clip) -> ClipCopywriting:
    """生成兜底文案（LLM 失败时使用）"""
    excerpt = clip.transcript_excerpt[:50] if clip.transcript_excerpt else clip.title

    default_copy = PlatformCopy(
        platform="default",
        title=clip.title,
        description=excerpt,
        hashtags=["#徐可", "#直播切片", "#情感"],
        hook=clip.title,
    )

    return ClipCopywriting(
        clip_index=clip.index,
        clip_title=clip.title,
        clip_category=clip.category,
        transcript_excerpt=clip.transcript_excerpt,
        platforms={
            "shipin_hao": PlatformCopy(**{**asdict(default_copy), "platform": "shipin_hao"}),
            "douyin": PlatformCopy(**{**asdict(default_copy), "platform": "douyin"}),
            "xiaohongshu": PlatformCopy(**{**asdict(default_copy), "platform": "xiaohongshu"}),
        },
    )


def save_copywriting(copywriting_list: List[ClipCopywriting],
                     output_path: str):
    """保存所有文案到 JSON"""
    data = [c.to_dict() for c in copywriting_list]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"文案已保存: {output_path}")


# ──────────────────────────────────────────────
# CLI 入口
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    print("copywriter 模块 — 通过 pipeline.py 调用")
    print("也可独立运行: python copywriter.py <clip_info.json> [config.yaml]")

    if len(sys.argv) >= 2:
        from clip_editor import ClipInfo

        with open(sys.argv[1], "r", encoding="utf-8") as f:
            clip_data = json.load(f)

        config_file = sys.argv[2] if len(sys.argv) > 2 else "config.yaml"
        with open(config_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        clips = [ClipInfo(**c) for c in clip_data]
        results = generate_copywriting(clips, cfg)

        out_path = os.path.join(
            cfg.get("paths", {}).get("output_dir", "output"),
            "copywriting.json"
        )
        save_copywriting(results, out_path)
        print(f"\n文案生成完成: {len(results)} 个切片, 输出: {out_path}")
