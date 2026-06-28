"""Content analysis using AI."""

import asyncio
import json
import re
from typing import List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, MofNCompleteColumn

from .client import AIClient
from .prompts import CONTENT_ANALYSIS_SYSTEM, CONTENT_ANALYSIS_USER
from .utils import parse_json_response
from ..models import ContentItem

DEFAULT_THROTTLE_SEC = 0.0
RADAR_CATEGORIES = {
    "今日核心热点",
    "AI 与科技动态",
    "地缘政治与国际关系",
    "中国政策与社会治理",
    "财经市场",
    "商业与产业趋势",
    "社会新闻与民生事件",
    "文化生活与大众情绪",
}
CONTENT_VALUES = {"可写公众号", "持续观察", "仅需了解", "忽略"}
DEFAULT_CATEGORY = "今日核心热点"


class ContentAnalyzer:
    """Analyzes content items using AI to determine importance."""

    def __init__(self, ai_client: AIClient):
        self.client = ai_client

    @staticmethod
    def _parse_json_response(response: str) -> Optional[dict]:
        """Try multiple strategies to extract a JSON object from an AI response.

        Returns the parsed dict, or None if all strategies fail.
        """
        return parse_json_response(response)

    def _get_throttle_sec(self) -> float:
        """Return the configured inter-item throttle, clamped to zero or above."""
        config = getattr(self.client, "config", None)
        throttle_sec = getattr(config, "throttle_sec", DEFAULT_THROTTLE_SEC)
        return max(throttle_sec, 0.0)

    def _get_concurrency(self) -> int:
        """Return the configured analysis concurrency, clamped to 1 or above."""
        config = getattr(self.client, "config", None)
        concurrency = getattr(config, "analysis_concurrency", 1)
        return max(concurrency, 1)

    async def analyze_batch(self, items: List[ContentItem]) -> List[ContentItem]:
        throttle_sec = self._get_throttle_sec()
        concurrency = self._get_concurrency()
        semaphore = asyncio.Semaphore(concurrency)

        async def _process(item: ContentItem, index: int, progress_task) -> ContentItem:
            async with semaphore:
                try:
                    await self._analyze_item(item)
                except Exception as e:
                    print(f"Error analyzing item {item.id}: {e}")
                    item.ai_score = 0.0
                    item.ai_reason = "Analysis failed"
                    item.ai_summary = item.title
                if throttle_sec > 0 and index < len(items) - 1:
                    await asyncio.sleep(throttle_sec)
            progress.advance(progress_task)
            return item

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("Analyzing", total=len(items))
            coros = [
                _process(item, i, task) for i, item in enumerate(items)
            ]
            analyzed_items = await asyncio.gather(*coros)

        return analyzed_items

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=2, max=10)
    )
    async def _analyze_item(self, item: ContentItem) -> None:
        """Analyze a single content item.

        Args:
            item: Content item to analyze (modified in-place)
        """
        # Prepare content section
        content_section = ""
        if item.content:
            # Split off comments if present
            content_text = item.content
            if "--- Top Comments ---" in content_text:
                main, comments_part = content_text.split("--- Top Comments ---", 1)
                content_section = f"Content: {main.strip()[:800]}"
            else:
                content_section = f"Content: {content_text[:1000]}"

        # Prepare discussion section (comments, engagement)
        discussion_parts = []
        if item.content and "--- Top Comments ---" in item.content:
            comments_part = item.content.split("--- Top Comments ---", 1)[1]
            discussion_parts.append(f"Community Comments:\n{comments_part[:1500]}")

        meta = item.metadata
        engagement_items = []
        if meta.get("score"):
            engagement_items.append(f"score: {meta['score']}")
        if meta.get("descendants"):
            engagement_items.append(f"{meta['descendants']} comments")
        if meta.get("favorite_count"):
            engagement_items.append(f"{meta['favorite_count']} likes")
        if meta.get("retweet_count"):
            engagement_items.append(f"{meta['retweet_count']} retweets")
        if meta.get("reply_count"):
            engagement_items.append(f"{meta['reply_count']} replies")
        if meta.get("views"):
            engagement_items.append(f"{meta['views']} views")
        if meta.get("bookmarks"):
            engagement_items.append(f"{meta['bookmarks']} bookmarks")
        if meta.get("upvote_ratio"):
            engagement_items.append(f"upvote ratio: {meta['upvote_ratio']:.0%}")
        if engagement_items:
            discussion_parts.append(f"Engagement: {', '.join(engagement_items)}")
        if meta.get("discussion_url"):
            discussion_parts.append(f"Discussion: {meta['discussion_url']}")
        if meta.get("community_note"):
            discussion_parts.append(f"Community Note: {meta['community_note']}")

        discussion_section = "\n".join(discussion_parts) if discussion_parts else ""

        # Generate user prompt
        user_prompt = CONTENT_ANALYSIS_USER.format(
            title=item.title,
            source=f"{item.source_type.value}",
            author=item.author or "Unknown",
            url=str(item.url),
            publish_time=item.published_at.isoformat() if item.published_at else "",
            source_category=item.metadata.get("category", ""),
            language=item.metadata.get("language", ""),
            region=item.metadata.get("region", ""),
            priority=item.metadata.get("priority", ""),
            source_credibility=item.metadata.get("source_credibility", ""),
            noise_level=item.metadata.get("noise_level", ""),
            content_section=content_section,
            discussion_section=discussion_section
        )

        # Get AI completion
        response = await self.client.complete(
            system=CONTENT_ANALYSIS_SYSTEM,
            user=user_prompt,
        )

        # Parse JSON response with robust fallback
        result = self._parse_json_response(response)
        if result is None:
            print(f"Warning: could not parse analysis response for {item.id}, using defaults")
            item.ai_score = 0.0
            item.ai_reason = "Analysis response parse failed"
            item.ai_summary = item.title
            item.ai_tags = []
            return

        # Update item with analysis results. Keep ai_* fields for the existing
        # filtering pipeline, and store radar-specific fields in metadata.
        importance_score = result.get("importance_score", result.get("score", 0))
        try:
            item.ai_score = max(1.0, min(10.0, float(importance_score)))
        except (TypeError, ValueError):
            item.ai_score = 0.0

        item.ai_reason = result.get("reason", "")
        item.ai_summary = result.get("summary", item.title)
        item.ai_tags = result.get("tags", [])

        category = result.get("category") or item.metadata.get("category") or DEFAULT_CATEGORY
        if category not in RADAR_CATEGORIES:
            category = DEFAULT_CATEGORY

        credibility = str(result.get("credibility", "medium")).lower()
        if credibility not in {"high", "medium", "low"}:
            credibility = "medium"

        content_value = result.get("content_value", "仅需了解")
        if content_value not in CONTENT_VALUES:
            content_value = "仅需了解"

        try:
            hotness_score = max(1.0, min(10.0, float(result.get("hotness_score", 5))))
        except (TypeError, ValueError):
            hotness_score = 5.0

        item.metadata.update({
            "radar_title": result.get("title") or item.title,
            "source": result.get("source") or item.source_type.value,
            "publish_time": result.get("publish_time") or (item.published_at.isoformat() if item.published_at else ""),
            "category": category,
            "importance_score": item.ai_score,
            "hotness_score": hotness_score,
            "credibility": credibility,
            "why_it_matters": result.get("why_it_matters", ""),
            "follow_up_needed": bool(result.get("follow_up_needed", False)),
            "content_value": content_value,
        })
