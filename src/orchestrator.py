"""Main orchestrator coordinating the entire workflow."""

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
from urllib.parse import urlparse
import httpx
from rich.console import Console

from .models import Config, ContentItem
from .storage.manager import StorageManager
from .services.email import EmailManager
from .services.webhook import WebhookNotifier
from .scrapers.github import GitHubScraper
from .scrapers.hackernews import HackerNewsScraper
from .scrapers.rss import RSSScraper
from .scrapers.reddit import RedditScraper
from .scrapers.telegram import TelegramScraper
from .scrapers.twitter import TwitterScraper
from .scrapers.twitter_playwright import TwitterPlaywrightScraper
from .scrapers.openbb import OpenBBScraper
from .scrapers.ossinsight import OSSInsightScraper
from .ai.client import create_ai_client
from .ai.analyzer import ContentAnalyzer
from .ai.summarizer import DailySummarizer
from .ai.enricher import ContentEnricher
from .ai.tokens import get_usage_snapshot


@dataclass
class BalancedDigestResult:
    """Items and selection statistics from balanced digest filtering."""

    items: List[ContentItem]
    enabled: bool = False
    group_counts: Dict[str, int] = field(default_factory=dict)
    group_limits: Dict[str, Optional[int]] = field(default_factory=dict)
    duplicate_categories: List[str] = field(default_factory=list)


class HorizonOrchestrator:
    """Orchestrates the complete workflow for content aggregation and analysis."""

    def __init__(self, config: Config, storage: StorageManager):
        """Initialize orchestrator.

        Args:
            config: Application configuration
            storage: Storage manager
        """
        self.config = config
        self.storage = storage
        self.console = Console()
        self.email_manager = EmailManager(config.email, console=self.console) if config.email else None
        self.webhook_notifier = (
            WebhookNotifier(config.webhook, console=self.console)
            if config.webhook and config.webhook.enabled
            else None
        )

    async def run(self, force_hours: int = None) -> None:
        """Execute the complete workflow.

        Args:
            force_hours: Optional override for time window in hours
        """
        self.console.print("[bold cyan]🌅 Horizon - Starting aggregation...[/bold cyan]\n")

        # Check email subscriptions if configured
        if (
            self.email_manager
            and self.config.email
            and self.config.email.enabled
            and self.config.email.imap_enabled
        ):
            self.console.print("📧 Checking for new email subscriptions...")
            self.email_manager.check_subscriptions(self.storage)

        try:
            # 1. Determine time window
            since = self._determine_time_window(force_hours)
            self.console.print(f"📅 Fetching content since: {since.strftime('%Y-%m-%d %H:%M:%S')}\n")

            # 2. Fetch content from all sources
            all_items = await self.fetch_all_sources(since)
            self.console.print(f"📥 Fetched {len(all_items)} items from all sources\n")

            if not all_items:
                self.console.print("[yellow]No new content found. Exiting.[/yellow]")
                return

            # 3. Merge cross-source duplicates (same URL from different sources)
            merged_items = self.merge_cross_source_duplicates(all_items)
            if len(merged_items) < len(all_items):
                self.console.print(
                    f"🔗 Merged {len(all_items) - len(merged_items)} cross-source duplicates "
                    f"→ {len(merged_items)} unique items\n"
                )

            candidates = self.apply_candidate_limit(merged_items)

            # 4. Analyze with AI
            analyzed_items = await self._analyze_content(candidates)
            self.console.print(f"🤖 Analyzed {len(analyzed_items)} items with AI\n")

            # 5. Filter by score threshold
            threshold = self.config.filtering.ai_score_threshold
            important_items = [
                item for item in analyzed_items
                if item.ai_score and item.ai_score >= self._effective_score_threshold(item, threshold)
            ]
            important_items.sort(key=lambda x: x.ai_score or 0, reverse=True)

            self.console.print(
                f"⭐️ {len(important_items)} items scored ≥ {threshold}\n"
            )

            balanced_pool = self.include_minimum_category_candidates(
                analyzed_items,
                important_items,
            )
            if len(balanced_pool) > len(important_items):
                self.console.print(
                    f"🧺 Added {len(balanced_pool) - len(important_items)} below-threshold "
                    "items to satisfy category minimums\n"
                )
            important_items = balanced_pool

            # 5.5 Semantic deduplication: drop items covering the same topic
            deduped_items = await self.merge_topic_duplicates(important_items)
            if len(deduped_items) < len(important_items):
                self.console.print(
                    f"🧹 Removed {len(important_items) - len(deduped_items)} topic duplicates "
                    f"→ {len(deduped_items)} unique items\n"
                )
            self.promote_core_hotspots(deduped_items)
            important_items = deduped_items

            # 5.6 Optional second-stage Twitter reply expansion + targeted re-analysis
            await self._expand_twitter_discussion(important_items)

            # 5.7 Apply per-category and global digest limits before enrichment
            balanced_result = self.apply_balanced_digest(important_items)
            important_items = balanced_result.items

            # Show per-sub-source selection breakdown
            selected_counts: Dict[str, int] = defaultdict(int)
            for item in important_items:
                key = f"{item.source_type.value}/{self._sub_source_label(item)}"
                selected_counts[key] += 1
            for source_key, count in sorted(selected_counts.items()):
                self.console.print(f"      • {source_key}: {count}")
            self.console.print("")

            # 6. Search related stories + enrich with background knowledge (2nd AI pass)
            await self._enrich_important_items(important_items)

            # 7. Generate and save daily summaries for each configured language
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            for lang in self.config.ai.languages:
                summarizer = DailySummarizer()
                summary = await summarizer.generate_summary(important_items, today, len(all_items), language=lang)

                # Save to data/summaries/
                summary_path = self.storage.save_daily_summary(today, summary, language=lang)
                self.console.print(f"💾 Saved {lang.upper()} summary to: {summary_path}\n")

                # Copy to docs/ for GitHub Pages
                try:
                    from pathlib import Path

                    post_filename = f"{today}-summary-{lang}.md"
                    posts_dir = Path("docs/_posts")
                    posts_dir.mkdir(parents=True, exist_ok=True)

                    dest_path = posts_dir / post_filename

                    # Add Jekyll front matter
                    front_matter = (
                        "---\n"
                        "layout: default\n"
                        f"title: \"Horizon Summary: {today} ({lang.upper()})\"\n"
                        f"date: {today}\n"
                        f"lang: {lang}\n"
                        "---\n\n"
                    )

                    # Strip leading H1 header to avoid duplication with Jekyll title
                    summary_content = summary
                    first_line = summary_content.strip().split("\n")[0]
                    if first_line.startswith("# "):
                        parts = summary_content.split("\n", 1)
                        if len(parts) > 1:
                            summary_content = parts[1].strip()

                    with open(dest_path, "w", encoding="utf-8") as f:
                        f.write(front_matter + summary_content)

                    self.console.print(f"📄 Copied {lang.upper()} summary to GitHub Pages: {dest_path}\n")
                except Exception as e:
                    self.console.print(f"[yellow]⚠️  Failed to copy {lang.upper()} summary to docs/: {e}[/yellow]\n")

                # Send email if configured
                if self.email_manager and self.config.email and self.config.email.enabled:
                    self.console.print(f"📧 Sending {lang.upper()} email summary...")
                    subscribers = self.storage.load_subscribers()
                    subject = f"Horizon Summary ({lang.upper()}) - {today}"
                    self.email_manager.send_daily_summary(summary, subject, subscribers)

                # Send webhook notification if configured
                if self.webhook_notifier:
                    await self.webhook_notifier.send_daily_summary(
                        summary=summary,
                        important_items=important_items,
                        all_items_count=len(all_items),
                        date=today,
                        lang=lang,
                        summarizer=summarizer,
                    )

            self.console.print("[bold green]✅ Horizon completed successfully![/bold green]")
            usage = get_usage_snapshot()
            if usage.total_tokens > 0:
                self.console.print(
                    f"\n🧮 Token usage this run: "
                    f"{usage.total_tokens} tokens "
                    f"(input: {usage.total_input_tokens}, output: {usage.total_output_tokens})"
                )
                for provider, u in sorted(usage.per_provider.items()):
                    if u.total <= 0:
                        continue
                    self.console.print(
                        f"   • {provider}: {u.total} tokens "
                        f"(in: {u.input_tokens}, out: {u.output_tokens})"
                    )

        except Exception as e:
            self.console.print(f"[bold red]❌ Error: {e}[/bold red]")

            # Send webhook failure notification if configured
            if self.webhook_notifier:
                await self.webhook_notifier.send_failure(
                    date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    error_message=str(e),
                )

            raise

    def _determine_time_window(self, force_hours: int = None) -> datetime:
        if force_hours:
            since = datetime.now(timezone.utc) - timedelta(hours=force_hours)
        else:
            hours = self.config.filtering.time_window_hours
            since = datetime.now(timezone.utc) - timedelta(hours=hours)
        return since

    async def fetch_all_sources(self, since: datetime) -> List[ContentItem]:
        """Fetch content from all configured sources.

        This is a stable stage entry point for integrations such as MCP.

        Args:
            since: Fetch items published after this time

        Returns:
            List[ContentItem]: All fetched items
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            tasks = []

            # GitHub sources
            if self.config.sources.github:
                github_scraper = GitHubScraper(self.config.sources.github, client)
                tasks.append(self._fetch_with_progress("GitHub", github_scraper, since))

            # Hacker News
            if self.config.sources.hackernews.enabled:
                hn_scraper = HackerNewsScraper(self.config.sources.hackernews, client)
                tasks.append(self._fetch_with_progress("Hacker News", hn_scraper, since))

            # RSS feeds
            if self.config.sources.rss:
                rss_scraper = RSSScraper(self.config.sources.rss, client)
                tasks.append(self._fetch_with_progress("RSS Feeds", rss_scraper, since))

            # Reddit
            if self.config.sources.reddit.enabled:
                reddit_scraper = RedditScraper(self.config.sources.reddit, client)
                tasks.append(self._fetch_with_progress("Reddit", reddit_scraper, since))

            # Telegram
            if self.config.sources.telegram.enabled:
                telegram_scraper = TelegramScraper(self.config.sources.telegram, client)
                tasks.append(self._fetch_with_progress("Telegram", telegram_scraper, since))

            # Twitter (Apify or Playwright mode)
            if self.config.sources.twitter and self.config.sources.twitter.enabled:
                tw_cfg = self.config.sources.twitter
                if tw_cfg.mode == "playwright":
                    twitter_scraper = TwitterPlaywrightScraper(tw_cfg)
                else:
                    twitter_scraper = TwitterScraper(tw_cfg, client)
                tasks.append(self._fetch_with_progress("Twitter", twitter_scraper, since))

            # OpenBB (financial news / filings via the OpenBB Platform SDK)
            if self.config.sources.openbb and self.config.sources.openbb.enabled:
                openbb_scraper = OpenBBScraper(self.config.sources.openbb, client)
                tasks.append(self._fetch_with_progress("OpenBB", openbb_scraper, since))

            # OSS Insight trending repos
            if self.config.sources.ossinsight and self.config.sources.ossinsight.enabled:
                oss_scraper = OSSInsightScraper(self.config.sources.ossinsight, client)
                tasks.append(self._fetch_with_progress("OSS Insight", oss_scraper, since))

            # Fetch all concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Flatten results
            all_items = []
            for result in results:
                if isinstance(result, Exception):
                    self.console.print(f"[red]Error fetching source: {result}[/red]")
                elif isinstance(result, list):
                    all_items.extend(result)

            return all_items

    async def _fetch_with_progress(self, name: str, scraper, since: datetime) -> List[ContentItem]:
        """Fetch from a scraper with progress indication.

        Args:
            name: Source name for display
            scraper: Scraper instance
            since: Fetch items after this time

        Returns:
            List[ContentItem]: Fetched items
        """
        self.console.print(f"🔍 Fetching from {name}...")
        items = await scraper.fetch(since)
        self.console.print(f"   Found {len(items)} items from {name}")

        # Show per-sub-source breakdown when there are multiple sub-sources
        sub_counts: Dict[str, int] = defaultdict(int)
        for item in items:
            sub_counts[self._sub_source_label(item)] += 1
        if len(sub_counts) > 1:
            for sub, count in sorted(sub_counts.items()):
                self.console.print(f"      • {sub}: {count}")

        return items

    @staticmethod
    def _sub_source_label(item: ContentItem) -> str:
        """Return a human-readable sub-source label for an item."""
        meta = item.metadata
        if meta.get("subreddit"):
            return f"r/{meta['subreddit']}"
        if meta.get("feed_name"):
            return meta["feed_name"]
        if meta.get("channel"):
            return f"@{meta['channel']}"
        if meta.get("period") and meta.get("repo"):
            return f"ossinsight:{meta.get('primary_language', 'all')}"
        if meta.get("repo"):
            return meta["repo"]
        if meta.get("watchlist"):
            return meta["watchlist"]
        return item.author or "unknown"

    def merge_cross_source_duplicates(self, items: List[ContentItem]) -> List[ContentItem]:
        """Merge items that point to the same URL from different sources.

        This is a stable stage helper for integrations such as MCP.

        Keeps the item with the richest content and combines metadata.

        Args:
            items: Items to deduplicate

        Returns:
            List[ContentItem]: Deduplicated items
        """
        def normalize_url(url: str) -> str:
            parsed = urlparse(str(url))
            # Strip www prefix, trailing slashes, and fragments
            host = parsed.hostname or ""
            if host.startswith("www."):
                host = host[4:]
            path = parsed.path.rstrip("/")
            return f"{host}{path}"

        # Group by normalized URL
        url_groups: Dict[str, List[ContentItem]] = {}
        for item in items:
            key = normalize_url(str(item.url))
            url_groups.setdefault(key, []).append(item)

        merged = []
        for key, group in url_groups.items():
            if len(group) == 1:
                merged.append(group[0])
                continue

            # Pick the item with the richest content as primary
            primary = max(group, key=lambda x: len(x.content or ""))

            # Merge metadata and source info from other items
            all_sources = set()
            for item in group:
                all_sources.add(item.source_type.value)
                # Merge metadata (engagement, discussion, etc.)
                for mk, mv in item.metadata.items():
                    if mk not in primary.metadata or not primary.metadata[mk]:
                        primary.metadata[mk] = mv

                # Append content (e.g., comments from another source)
                if item is not primary and item.content:
                    if primary.content and item.content not in primary.content:
                        primary.content = (primary.content or "") + f"\n\n--- From {item.source_type.value} ---\n" + item.content

            primary.metadata["merged_sources"] = list(all_sources)
            merged.append(primary)

        return merged

    async def merge_topic_duplicates(self, items: List[ContentItem]) -> List[ContentItem]:
        """Merge items covering the same topic using AI semantic deduplication.

        This is a stable stage helper for integrations such as MCP.

        Sends all item titles, tags, and summaries to AI in a single call.
        Items must already be sorted by ai_score descending so that the first
        item in each duplicate group is always the highest-scored one.
        Content (comments) from duplicate items is merged into the primary.

        Falls back to returning items unchanged if the AI call fails.
        """
        if len(items) <= 1:
            return items

        from .ai.prompts import TOPIC_DEDUP_SYSTEM, TOPIC_DEDUP_USER
        from .ai.utils import parse_json_response

        # Build the item list for the prompt
        lines = []
        for i, item in enumerate(items):
            tags = ", ".join(item.ai_tags) if item.ai_tags else "—"
            summary = item.ai_summary or "—"
            lines.append(f"[{i}] {item.title}\n    Tags: {tags}\n    Summary: {summary}")
        items_text = "\n\n".join(lines)

        try:
            ai_client = create_ai_client(self.config.ai)
            response = await ai_client.complete(
                system=TOPIC_DEDUP_SYSTEM,
                user=TOPIC_DEDUP_USER.format(items=items_text),
            )
            result = parse_json_response(response)
            if result is None:
                self.console.print("[yellow]  dedup: could not parse AI response, skipping[/yellow]")
                return items

            duplicate_groups = result.get("duplicates", [])
        except Exception as e:
            self.console.print(f"[yellow]  dedup: AI call failed ({e}), skipping[/yellow]")
            return items

        if not duplicate_groups:
            return items

        # Build a set of indices to drop (all non-primary duplicates)
        drop_indices: set[int] = set()
        for group in duplicate_groups:
            if not isinstance(group, list) or len(group) < 2:
                continue
            primary_idx = group[0]
            if primary_idx < 0 or primary_idx >= len(items):
                continue
            primary = items[primary_idx]
            for dup_idx in group[1:]:
                if not isinstance(dup_idx, int) or dup_idx < 0 or dup_idx >= len(items):
                    continue
                if dup_idx == primary_idx:
                    continue
                dup = items[dup_idx]
                related_items = primary.metadata.setdefault("related_items", [])
                if isinstance(related_items, list):
                    related_items.append(self._related_item_metadata(dup))
                # Merge comments/content from the duplicate into the primary
                if dup.content:
                    if not primary.content or dup.content not in primary.content:
                        label = dup.source_type.value
                        primary.content = (primary.content or "") + f"\n\n--- From {label} ---\n{dup.content}"
                self.console.print(
                    f"   [dim]dedup: keep [{primary_idx}] {primary.title}[/dim]\n"
                    f"   [dim]       drop [{dup_idx}] {dup.title}[/dim]"
                )
                drop_indices.add(dup_idx)

        return [item for i, item in enumerate(items) if i not in drop_indices]

    def _related_item_metadata(self, item: ContentItem) -> dict:
        """Compact metadata for a duplicate item merged into an event cluster."""
        return {
            "title": item.metadata.get("radar_title") or item.title,
            "source": self._sub_source_label(item),
            "url": str(item.url),
            "category": item.metadata.get("category"),
            "summary": item.ai_summary or item.metadata.get("summary") or "",
            "publish_time": item.published_at.isoformat() if item.published_at else "",
        }

    def apply_balanced_digest(
        self,
        items: List[ContentItem],
        *,
        log: bool = True,
    ) -> BalancedDigestResult:
        """Apply configured category quotas and the final item cap.

        Categories are read from ``item.metadata["category"]``. If a category
        appears in more than one configured group, the first group in config
        order wins.
        """
        filtering = self.config.filtering
        groups = filtering.category_groups
        max_items = filtering.max_items

        if not groups and max_items is None:
            return BalancedDigestResult(items=items)

        sorted_items = sorted(
            items,
            key=lambda item: item.ai_score or 0,
            reverse=True,
        )

        category_to_group: Dict[str, str] = {}
        duplicate_categories: List[str] = []
        for group_key, group in groups.items():
            for category in group.categories:
                if category in category_to_group:
                    if category_to_group[category] != group_key:
                        duplicate_categories.append(category)
                    continue
                category_to_group[category] = group_key

        if log:
            for category in sorted(set(duplicate_categories)):
                first_group = category_to_group[category]
                self.console.print(
                    f"[yellow]Warning: category '{category}' is configured in multiple "
                    f"groups; using '{first_group}'.[/yellow]"
                )

        selected: List[tuple[ContentItem, str]] = []
        selected_ids: set[str] = set()
        group_counts: Dict[str, int] = defaultdict(int)
        default_group = filtering.default_group
        item_groups: Dict[str, str] = {}
        group_buckets: Dict[str, List[ContentItem]] = defaultdict(list)

        for item in sorted_items:
            category = item.metadata.get("category")
            group_key = (
                category_to_group.get(category, default_group)
                if isinstance(category, str)
                else default_group
            )
            item_groups[item.id] = group_key
            group_buckets[group_key].append(item)

        for group_key, group in groups.items():
            min_items = min(group.min_items, group.limit)
            if min_items <= 0:
                continue
            for item in group_buckets.get(group_key, []):
                if group_counts[group_key] >= min_items:
                    break
                selected.append((item, group_key))
                selected_ids.add(item.id)
                group_counts[group_key] += 1

        for item in sorted_items:
            if item.id in selected_ids:
                continue
            group_key = item_groups.get(item.id, default_group)

            if group_key in groups:
                limit = groups[group_key].limit
            else:
                limit = filtering.default_group_limit

            if limit is not None and group_counts[group_key] >= limit:
                continue

            selected.append((item, group_key))
            selected_ids.add(item.id)
            group_counts[group_key] += 1

        if max_items is not None:
            selected = selected[:max_items]

        final_counts: Dict[str, int] = defaultdict(int)
        for _, group_key in selected:
            final_counts[group_key] += 1

        group_limits: Dict[str, Optional[int]] = {
            group_key: group.limit for group_key, group in groups.items()
        }
        group_limits.setdefault(default_group, filtering.default_group_limit)

        if log:
            self.console.print(
                f"⚖️ Balanced digest selected {len(selected)}/{len(items)} items"
            )
            for group_key, group in groups.items():
                label = group.name or group_key
                self.console.print(
                    f"      • {label}: {final_counts.get(group_key, 0)}/{group.limit}"
                )
            if (
                final_counts.get(default_group, 0)
                or filtering.default_group_limit is not None
            ):
                limit_label = (
                    str(filtering.default_group_limit)
                    if filtering.default_group_limit is not None
                    else "unlimited"
                )
                self.console.print(
                    f"      • {default_group}: "
                    f"{final_counts.get(default_group, 0)}/{limit_label}"
                )
            self.console.print("")

        return BalancedDigestResult(
            items=[item for item, _ in selected],
            enabled=True,
            group_counts=dict(final_counts),
            group_limits=group_limits,
            duplicate_categories=sorted(set(duplicate_categories)),
        )

    def include_minimum_category_candidates(
        self,
        analyzed_items: List[ContentItem],
        important_items: List[ContentItem],
    ) -> List[ContentItem]:
        """Add the best below-threshold items needed for configured minimums."""
        filtering = self.config.filtering
        groups = filtering.category_groups
        if not groups:
            return important_items

        category_to_group: Dict[str, str] = {}
        for group_key, group in groups.items():
            for category in group.categories:
                category_to_group.setdefault(category, group_key)

        def _group_for(item: ContentItem) -> str:
            category = item.metadata.get("category")
            if isinstance(category, str):
                return category_to_group.get(category, filtering.default_group)
            return filtering.default_group

        selected_ids = {item.id for item in important_items}
        selected = list(important_items)
        group_counts: Dict[str, int] = defaultdict(int)
        buckets: Dict[str, List[ContentItem]] = defaultdict(list)

        for item in important_items:
            group_counts[_group_for(item)] += 1

        for item in sorted(
            analyzed_items,
            key=lambda content_item: content_item.ai_score or 0,
            reverse=True,
        ):
            if item.id in selected_ids:
                continue
            group_key = _group_for(item)
            if group_key in groups:
                buckets[group_key].append(item)

        for group_key, group in groups.items():
            min_items = min(group.min_items, group.limit)
            if min_items <= 0:
                continue
            for item in buckets.get(group_key, []):
                if group_counts[group_key] >= min_items:
                    break
                selected.append(item)
                selected_ids.add(item.id)
                group_counts[group_key] += 1

        return sorted(
            selected,
            key=lambda content_item: content_item.ai_score or 0,
            reverse=True,
        )

    def promote_core_hotspots(self, items: List[ContentItem]) -> None:
        """Ensure the cross-domain core hotspot category has enough items."""
        groups = self.config.filtering.category_groups
        core_category = "今日核心热点"
        core_min_items = 0
        for group in groups.values():
            if core_category in group.categories:
                core_min_items = min(group.min_items, group.limit)
                break
        if core_min_items <= 0:
            return

        core_count = sum(item.metadata.get("category") == core_category for item in items)
        if core_count >= core_min_items:
            return

        def _hotspot_key(item: ContentItem) -> tuple:
            importance = item.metadata.get("importance_score") or item.ai_score or 0
            hotness = item.metadata.get("hotness_score") or 0
            return importance, hotness, item.ai_score or 0

        promoted = 0
        for item in sorted(items, key=_hotspot_key, reverse=True):
            if core_count >= core_min_items:
                break
            if item.metadata.get("category") == core_category:
                continue
            item.metadata.setdefault("original_category", item.metadata.get("category"))
            item.metadata["category"] = core_category
            core_count += 1
            promoted += 1

        if promoted:
            self.console.print(
                f"🔥 Promoted {promoted} top items to 今日核心热点 minimum\n"
            )

    def apply_candidate_limit(self, items: List[ContentItem]) -> List[ContentItem]:
        """Limit pre-analysis candidates to keep large source sets affordable."""
        limit = self.config.filtering.max_candidates
        if limit is None or len(items) <= limit:
            return items

        def _candidate_key(item: ContentItem) -> tuple:
            try:
                priority = int(item.metadata.get("priority") or 3)
            except (TypeError, ValueError):
                priority = 3
            noise_rank = {"low": 2, "medium": 1, "high": 0}.get(
                item.metadata.get("noise_level"),
                1,
            )
            timestamp = item.published_at.timestamp() if item.published_at else 0
            return priority, noise_rank, timestamp

        sorted_items = sorted(items, key=_candidate_key, reverse=True)
        groups = self.config.filtering.category_groups

        if not groups:
            selected = sorted_items[:limit]
        else:
            category_to_group: Dict[str, str] = {}
            for group_key, group in groups.items():
                for category in group.categories:
                    category_to_group.setdefault(category, group_key)

            def _source_group(item: ContentItem) -> Optional[str]:
                category = item.metadata.get("category")
                if isinstance(category, str):
                    return category_to_group.get(category)
                return None

            selected = []
            selected_ids: set[str] = set()
            for group_key, group in groups.items():
                reserve_target = min(group.limit, max(group.min_items * 4, group.min_items))
                if reserve_target <= 0:
                    continue
                group_count = 0
                for item in sorted_items:
                    if item.id in selected_ids or _source_group(item) != group_key:
                        continue
                    selected.append(item)
                    selected_ids.add(item.id)
                    group_count += 1
                    if group_count >= reserve_target or len(selected) >= limit:
                        break
                if len(selected) >= limit:
                    break

            for item in sorted_items:
                if len(selected) >= limit:
                    break
                if item.id in selected_ids:
                    continue
                selected.append(item)
                selected_ids.add(item.id)

        self.console.print(
            f"🎯 Candidate prefilter selected {len(selected)}/{len(items)} items before AI analysis\n"
        )
        return selected

    @staticmethod
    def _effective_score_threshold(item: ContentItem, base_threshold: float) -> float:
        """Raise the bar for low-priority or noisy sources before enrichment."""
        priority = item.metadata.get("priority")
        noise_level = item.metadata.get("noise_level")

        try:
            priority_value = int(priority)
        except (TypeError, ValueError):
            priority_value = 3

        adjustment = 0.0
        if priority_value <= 2:
            adjustment += 0.8
        elif priority_value >= 5:
            adjustment -= 0.4

        if noise_level == "high":
            adjustment += 0.6
        elif noise_level == "low":
            adjustment -= 0.2

        return max(1.0, min(10.0, base_threshold + adjustment))

    async def _expand_twitter_discussion(self, items: List[ContentItem]) -> None:
        """Second-stage: fetch reply text for important Twitter items and re-analyze.

        Only runs when sources.twitter.fetch_reply_text is True.
        Bounded by max_tweets_to_expand to control cost.
        """
        tw_cfg = self.config.sources.twitter
        if not tw_cfg or not tw_cfg.enabled or not tw_cfg.fetch_reply_text:
            return

        from .models import SourceType

        twitter_items = [
            item for item in items
            if item.source_type == SourceType.TWITTER
        ][:tw_cfg.max_tweets_to_expand]

        if not twitter_items:
            return

        self.console.print(
            f"💬 Fetching reply text for {len(twitter_items)} Twitter items..."
        )

        async with httpx.AsyncClient(timeout=30.0) as client:
            if tw_cfg.mode == "playwright":
                self.console.print(
                    "   [yellow]Reply expansion not yet supported in Playwright mode.[/yellow]"
                )
                return
            scraper = TwitterScraper(tw_cfg, client)
            expanded = []
            for item in twitter_items:
                try:
                    reply_lines = await scraper.fetch_replies_for_item(item)
                    if TwitterScraper.append_discussion_content(item, reply_lines):
                        expanded.append(item)
                        self.console.print(
                            f"   💬 {len(reply_lines)} replies added to: {item.title[:60]}"
                        )
                except Exception as exc:
                    self.console.print(
                        f"   [yellow]⚠️  Reply fetch failed for {item.id}: {exc}[/yellow]"
                    )

        if not expanded:
            return

        self.console.print(
            f"   Re-analyzing {len(expanded)} Twitter items with reply context...\n"
        )
        ai_client = create_ai_client(self.config.ai)
        analyzer = ContentAnalyzer(ai_client)
        await analyzer.analyze_batch(expanded)

    async def _enrich_important_items(self, items: List[ContentItem]) -> None:
        """Enrich items with background knowledge (2nd AI pass).

        For each item that passed the score threshold, call AI to generate
        background knowledge based on the item's actual content.

        Args:
            items: Important items to enrich (modified in-place)
        """
        if not items:
            return

        self.console.print("📚 Enriching with background knowledge...")
        ai_client = create_ai_client(self.config.ai)
        enricher = ContentEnricher(ai_client)
        await enricher.enrich_batch(items)
        self.console.print(f"   Enriched {len(items)} items\n")

    async def _analyze_content(self, items: List[ContentItem]) -> List[ContentItem]:
        """Analyze content items with AI.

        Args:
            items: Items to analyze

        Returns:
            List[ContentItem]: Analyzed items
        """
        self.console.print("🤖 Analyzing content with AI...")

        ai_client = create_ai_client(self.config.ai)
        analyzer = ContentAnalyzer(ai_client)

        return await analyzer.analyze_batch(items)

    async def _generate_summary(
        self,
        items: List[ContentItem],
        date: str,
        total_fetched: int,
        language: str = "en",
    ) -> str:
        """Generate daily summary.

        Args:
            items: Important items to include (already enriched with background/related)
            date: Date string
            total_fetched: Total items fetched
            language: Output language ("en" or "zh")

        Returns:
            str: Markdown summary
        """
        self.console.print("📝 Generating daily summary...")

        summarizer = DailySummarizer()

        return await summarizer.generate_summary(items, date, total_fetched, language=language)
