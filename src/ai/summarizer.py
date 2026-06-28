"""Daily summary generation — pure programmatic rendering."""

import re
from typing import List, Dict

from ..models import ContentItem


_CJK = r"[\u4e00-\u9fff\u3400-\u4dbf]"
_ASCII = r"[A-Za-z0-9]"


def _pangu(text: str) -> str:
    """Insert a space between CJK and ASCII letters/digits (Pangu spacing)."""
    text = re.sub(rf"({_CJK})({_ASCII})", r"\1 \2", text)
    text = re.sub(rf"({_ASCII})({_CJK})", r"\1 \2", text)
    return text


LABELS = {
    "en": {
        "header": "Horizon Daily",
        "source": "Source",
        "background": "Background",
        "discussion": "Discussion",
        "references": "References",
        "tags": "Tags",
        "selected_items": "From {total} items, {selected} important content pieces were selected",
        "empty_analyzed": "Analyzed {total} items, but none met the importance threshold.",
        "empty_body": (
            "No significant developments today. This might indicate:\n"
            "- A quiet day in your tracked sources\n"
            "- The AI score threshold is too high\n"
            "- Your information sources need expansion\n\n"
            "Consider:\n"
            "1. Lowering the `ai_score_threshold` in config.json\n"
            "2. Adding more diverse information sources\n"
            "3. Checking if the AI model is working correctly\n"
        ),
    },
    "zh": {
        "header": "Horizon 每日速递",
        "source": "来源",
        "background": "背景",
        "discussion": "社区讨论",
        "references": "参考链接",
        "tags": "标签",
        "selected_items": "从 {total} 条内容中筛选出 {selected} 条重要资讯。",
        "empty_analyzed": "已分析 {total} 条内容，但没有达到重要性阈值的条目。",
        "empty_body": (
            "今日暂无重要动态，可能原因：\n"
            "- 今天关注的信息源较平静\n"
            "- AI 评分阈值设置过高\n"
            "- 信息源种类有待扩充\n\n"
            "建议：\n"
            "1. 在 config.json 中降低 `ai_score_threshold`\n"
            "2. 添加更多多样化的信息源\n"
            "3. 检查 AI 模型是否正常工作\n"
        ),
    },
}

RADAR_CATEGORIES = [
    "今日核心热点",
    "AI 与科技动态",
    "地缘政治与国际关系",
    "中国政策与社会治理",
    "财经市场",
    "商业与产业趋势",
    "社会新闻与民生事件",
    "文化生活与大众情绪",
]


class DailySummarizer:
    """Generates daily Markdown summaries from pre-analyzed content items."""

    def __init__(self):
        pass

    async def generate_summary(
        self,
        items: List[ContentItem],
        date: str,
        total_fetched: int,
        language: str = "en",
    ) -> str:
        """Generate daily summary in Markdown format.

        Items are rendered in score-descending order (already sorted by orchestrator).

        Args:
            items: High-scoring content items (already enriched)
            date: Date string (YYYY-MM-DD)
            total_fetched: Total number of items fetched before filtering
            language: Output language, either "en" or "zh"

        Returns:
            str: Markdown formatted summary
        """
        labels = LABELS.get(language, LABELS["en"])

        if not items:
            return self._generate_empty_summary(date, total_fetched, labels)

        if language != "zh":
            parts = [self._format_item(item, labels, language, i + 1) for i, item in enumerate(items)]
            return (
                f"# {labels['header']} - {date}\n\n"
                f"> {labels['selected_items'].format(total=total_fetched, selected=len(items))}\n\n"
                "---\n\n"
                + "".join(parts)
            )

        return self._generate_personal_radar_summary(items, date, total_fetched)

    def _generate_personal_radar_summary(
        self,
        items: List[ContentItem],
        date: str,
        total_fetched: int,
    ) -> str:
        """Generate the Chinese personal daily information radar."""
        ranked = sorted(
            items,
            key=lambda item: (
                float(item.metadata.get("importance_score") or item.ai_score or 0),
                float(item.metadata.get("hotness_score") or 0),
            ),
            reverse=True,
        )

        lines = [
            f"# 个人每日信息雷达 - {date}",
            "",
            f"> 从 {total_fetched} 条内容中筛选出 {len(items)} 条重点信息。",
            "",
            "## 1. 今日必看（最多 30 条）",
            "",
        ]

        for index, item in enumerate(ranked[:30], start=1):
            lines.extend(self._format_radar_item(item, index, compact=False))

        lines.extend(["", "## 2. 分类简报", ""])
        for category in RADAR_CATEGORIES:
            category_items = [item for item in ranked if self._radar_category(item) == category]
            lines.append(f"### {category}")
            if not category_items:
                lines.extend(["", "- 今日暂无高价值条目。", ""])
                continue
            for item in category_items:
                title = self._radar_title(item)
                score = self._score_text(item)
                lines.append(f"- [{title}]({item.url}) · {score} · {self._content_value(item)}")
            lines.append("")

        follow_items = [
            item for item in ranked
            if item.metadata.get("follow_up_needed") or self._content_value(item) == "持续观察"
        ]
        lines.extend(["## 3. 值得持续跟踪的事件", ""])
        if follow_items:
            for item in follow_items:
                lines.append(f"- [{self._radar_title(item)}]({item.url}): {self._why_it_matters(item)}")
        else:
            lines.append("- 今日暂无必须持续跟踪的事件。")
        lines.append("")

        writing_items = [item for item in ranked if self._content_value(item) == "可写公众号"]
        lines.extend(["## 4. 可转化为公众号选题", ""])
        if writing_items:
            for item in writing_items:
                lines.append(f"- [{self._radar_title(item)}]({item.url}): {self._summary(item)}")
        else:
            lines.append("- 今日暂无特别适合展开成公众号文章的选题。")
        lines.append("")

        tomorrow_items = follow_items[:5] or ranked[:5]
        lines.extend(["## 5. 明天继续观察什么", ""])
        for item in tomorrow_items:
            lines.append(f"- {self._radar_title(item)}")

        return "\n".join(lines).rstrip() + "\n"

    def _format_radar_item(self, item: ContentItem, index: int, compact: bool = False) -> list[str]:
        tags = item.ai_tags or []
        tag_text = ", ".join(tags) if tags else ""
        fields = [
            f"### {index}. [{self._radar_title(item)}]({item.url})",
            "",
            f"- title: {self._radar_title(item)}",
            f"- source: {self._source_label(item)}",
            f"- url: {item.url}",
            f"- publish_time: {self._publish_time(item)}",
            f"- category: {self._radar_category(item)}",
            f"- tags: {tag_text}",
            f"- importance_score: {self._importance_score(item)}",
            f"- hotness_score: {self._hotness_score(item)}",
            f"- credibility: {item.metadata.get('credibility', 'medium')}",
            f"- summary: {self._summary(item)}",
            f"- why_it_matters: {self._why_it_matters(item)}",
            f"- follow_up_needed: {self._follow_up_text(item)}",
            f"- content_value: {self._content_value(item)}",
        ]
        fields.extend(self._related_item_lines(item))
        fields.append("")
        return fields

    def _radar_title(self, item: ContentItem) -> str:
        title = item.metadata.get("radar_title") or item.metadata.get("title_zh") or item.title
        return _pangu(str(title).replace("[", "(").replace("]", ")"))

    def _radar_category(self, item: ContentItem) -> str:
        category = item.metadata.get("category")
        return category if category in RADAR_CATEGORIES else "今日核心热点"

    def _importance_score(self, item: ContentItem) -> str:
        return str(item.metadata.get("importance_score") or item.ai_score or "")

    def _hotness_score(self, item: ContentItem) -> str:
        return str(item.metadata.get("hotness_score") or "")

    def _score_text(self, item: ContentItem) -> str:
        return f"重要性 {self._importance_score(item)}/10，热度 {self._hotness_score(item)}/10"

    def _source_label(self, item: ContentItem) -> str:
        meta = item.metadata
        if meta.get("source"):
            return str(meta["source"])
        if meta.get("feed_name"):
            return str(meta["feed_name"])
        if meta.get("subreddit"):
            return f"reddit/r/{meta['subreddit']}"
        return item.source_type.value

    def _publish_time(self, item: ContentItem) -> str:
        if item.metadata.get("publish_time"):
            return str(item.metadata["publish_time"])
        return item.published_at.isoformat() if item.published_at else ""

    def _summary(self, item: ContentItem) -> str:
        summary = (
            item.metadata.get("detailed_summary_zh")
            or item.metadata.get("detailed_summary")
            or item.ai_summary
            or ""
        )
        return _pangu(str(summary))

    def _why_it_matters(self, item: ContentItem) -> str:
        why = item.metadata.get("why_it_matters") or item.metadata.get("why_it_matters_zh") or item.ai_reason or ""
        return _pangu(str(why))

    def _follow_up_text(self, item: ContentItem) -> str:
        return "true" if item.metadata.get("follow_up_needed") else "false"

    def _content_value(self, item: ContentItem) -> str:
        return str(item.metadata.get("content_value") or "仅需了解")

    def _related_item_lines(self, item: ContentItem) -> list[str]:
        related = item.metadata.get("related_items")
        if not isinstance(related, list) or not related:
            return []

        lines = ["- related_items:"]
        for related_item in related[:5]:
            if not isinstance(related_item, dict):
                continue
            title = _pangu(str(related_item.get("title") or "相关报道"))
            url = str(related_item.get("url") or "")
            source = str(related_item.get("source") or "unknown")
            summary = _pangu(str(related_item.get("summary") or ""))
            if url:
                line = f"  - [{title}]({url}) · {source}"
            else:
                line = f"  - {title} · {source}"
            if summary:
                line += f" · {summary}"
            lines.append(line)
        return lines

    def generate_webhook_overview(
        self,
        items: List[ContentItem],
        date: str,
        total_fetched: int,
        language: str = "en",
    ) -> str:
        """Generate a compact overview for multi-message webhook delivery."""
        labels = LABELS.get(language, LABELS["en"])
        if not items:
            return self._generate_empty_summary(date, total_fetched, labels)

        if language == "zh":
            header = (
                f"# {labels['header']} - {date}\n\n"
                f"> 从 {total_fetched} 条内容中筛选出 {len(items)} 条重要资讯。\n\n"
                "下面会按新闻逐条发送详情，你可以只看感兴趣的标题。\n\n"
            )
        else:
            header = (
                f"# {labels['header']} - {date}\n\n"
                f"> Selected {len(items)} important items from {total_fetched} fetched items.\n\n"
                "Details will be sent item by item so you can read only the topics you care about.\n\n"
            )

        entries = []
        for i, item in enumerate(items, start=1):
            title = str(item.metadata.get(f"title_{language}") or item.title).replace("[", "(").replace("]", ")")
            if language == "zh":
                title = _pangu(title)
            score = item.ai_score or "?"
            entries.append(f"{i}. [{title}]({item.url}) \u2b50\ufe0f {score}/10")

        return header + "\n".join(entries)

    def generate_webhook_item(
        self,
        item: ContentItem,
        language: str,
        index: int,
        total: int,
    ) -> str:
        """Generate one item message for multi-message webhook delivery."""
        labels = LABELS.get(language, LABELS["en"])
        prefix = f"第 {index}/{total} 条\n\n" if language == "zh" else f"Item {index}/{total}\n\n"
        return prefix + self._format_item(item, labels, language, index).rstrip("-\n ")

    def _format_item(self, item: ContentItem, labels: dict, language: str, index: int) -> str:
        """Format a single ContentItem into Markdown."""
        _title = item.metadata.get(f"title_{language}") or item.title
        title = str(_title).replace("[", "(").replace("]", ")")
        url = str(item.url)
        score = item.ai_score or "?"
        meta = item.metadata

        summary = (
            meta.get(f"detailed_summary_{language}")
            or meta.get("detailed_summary")
            or item.ai_summary
            or ""
        )
        background = meta.get(f"background_{language}") or meta.get("background") or ""
        discussion = (
            meta.get(f"community_discussion_{language}")
            or meta.get("community_discussion")
            or ""
        )

        if language == "zh":
            title = _pangu(title)
            summary = _pangu(summary)
            background = _pangu(background)
            discussion = _pangu(discussion)

        # Source line with parts joined by " · ", link appended at end
        source_type = item.source_type.value
        source_parts = [source_type]
        if meta.get("subreddit"):
            source_parts.append(f"r/{meta['subreddit']}")
        if meta.get("feed_name"):
            source_parts.append(meta["feed_name"])
        else:
            source_parts.append(item.author or "unknown")
        if item.published_at:
            if language == "zh":
                source_parts.append(
                    f"{item.published_at.month}月{item.published_at.day}日 "
                    f"{item.published_at:%H:%M}"
                )
            else:
                day = item.published_at.strftime("%d").lstrip("0")
                source_parts.append(item.published_at.strftime(f"%b {day}, %H:%M"))
        source_line = " \u00b7 ".join(source_parts)  # ·

        discussion_url = meta.get("discussion_url")
        if discussion_url:
            discussion_url = str(discussion_url)
            if discussion_url != url:
                source_line += f' · [{labels["discussion"]}]({discussion_url})'

        lines = [
            f'<a id="item-{index}"></a>',
            f"## [{title}]({url}) \u2b50\ufe0f {score}/10",  # ⭐️
            "",
            summary,
            "",
            source_line,
        ]

        if background:
            lines.append("")
            lines.append(f"**{labels['background']}**: {background}")

        sources = meta.get("sources") or []
        if sources:
            items_html = "".join(f'<li><a href="{s["url"]}">{s["title"]}</a></li>\n' for s in sources)
            lines += [
                "",
                f'<details><summary>{labels["references"]}</summary>\n<ul>\n{items_html}\n</ul>\n</details>',
            ]

        if discussion:
            lines.append("")
            lines.append(f"**{labels['discussion']}**: {discussion}")

        if item.ai_tags:
            tags_str = ", ".join([f"`#{t}`" for t in item.ai_tags])
            lines.append("")
            lines.append(f"**{labels['tags']}**: {tags_str}")

        lines.append("")
        lines.append("---")

        return "\n".join(lines) + "\n\n"

    def _generate_empty_summary(self, date: str, total_fetched: int, labels: dict) -> str:
        """Generate summary when no high-scoring items were found."""
        return (
            f"# {labels['header']} - {date}\n\n"
            f"> {labels['empty_analyzed'].format(total=total_fetched)}\n\n"
            + labels["empty_body"]
        )
