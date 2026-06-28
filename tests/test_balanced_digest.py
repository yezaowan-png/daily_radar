import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from rich.console import Console

from src.models import (
    AIConfig,
    CategoryGroupConfig,
    Config,
    ContentItem,
    FilteringConfig,
    SourceType,
    SourcesConfig,
)
from src.orchestrator import HorizonOrchestrator


def make_item(item_id: str, score: float, category: str | None) -> ContentItem:
    metadata = {"category": category} if category is not None else {}
    return ContentItem(
        id=item_id,
        source_type=SourceType.RSS,
        title=item_id,
        url=f"https://example.com/{item_id}",
        published_at=datetime.now(timezone.utc),
        ai_score=score,
        metadata=metadata,
    )


def make_orchestrator(filtering: FilteringConfig) -> HorizonOrchestrator:
    orchestrator = HorizonOrchestrator.__new__(HorizonOrchestrator)
    orchestrator.config = SimpleNamespace(filtering=filtering)
    orchestrator.console = Console(record=True)
    return orchestrator


def test_unconfigured_balanced_digest_preserves_old_behavior() -> None:
    items = [make_item("lower", 7.0, "ai"), make_item("higher", 9.0, "finance")]
    result = make_orchestrator(FilteringConfig()).apply_balanced_digest(items)

    assert result.enabled is False
    assert result.items is items


def test_category_groups_apply_limits_and_default_group_limit() -> None:
    filtering = FilteringConfig(
        category_groups={
            "ai": CategoryGroupConfig(limit=2, categories=["ai", "ml"]),
            "finance": CategoryGroupConfig(limit=1, categories=["finance"]),
        },
        default_group_limit=1,
    )
    items = [
        make_item("ai-low", 7.0, "ai"),
        make_item("finance-low", 6.0, "finance"),
        make_item("other-high", 9.5, "world"),
        make_item("ai-high", 9.0, "ml"),
        make_item("finance-high", 8.5, "finance"),
        make_item("ai-mid", 8.0, "ai"),
        make_item("other-low", 5.0, None),
    ]

    result = make_orchestrator(filtering).apply_balanced_digest(items)

    assert [item.id for item in result.items] == [
        "other-high",
        "ai-high",
        "finance-high",
        "ai-mid",
    ]
    assert result.group_counts == {"other": 1, "ai": 2, "finance": 1}


def test_max_items_applies_after_group_limits() -> None:
    filtering = FilteringConfig(
        max_items=2,
        category_groups={
            "ai": CategoryGroupConfig(limit=2, categories=["ai"]),
            "finance": CategoryGroupConfig(limit=2, categories=["finance"]),
        },
    )
    items = [
        make_item("finance", 8.0, "finance"),
        make_item("ai-top", 10.0, "ai"),
        make_item("ai-second", 9.0, "ai"),
    ]

    result = make_orchestrator(filtering).apply_balanced_digest(items)

    assert [item.id for item in result.items] == ["ai-top", "ai-second"]
    assert result.group_counts == {"ai": 2}


def test_min_items_preserves_category_coverage_before_global_score_order() -> None:
    filtering = FilteringConfig(
        max_items=4,
        category_groups={
            "ai": CategoryGroupConfig(limit=4, min_items=2, categories=["ai"]),
            "finance": CategoryGroupConfig(limit=4, min_items=2, categories=["finance"]),
        },
    )
    items = [
        make_item("ai-10", 10.0, "ai"),
        make_item("ai-9", 9.0, "ai"),
        make_item("ai-8", 8.0, "ai"),
        make_item("ai-7", 7.0, "ai"),
        make_item("finance-6", 6.0, "finance"),
        make_item("finance-5", 5.0, "finance"),
    ]

    result = make_orchestrator(filtering).apply_balanced_digest(items)

    assert [item.id for item in result.items] == [
        "ai-10",
        "ai-9",
        "finance-6",
        "finance-5",
    ]
    assert result.group_counts == {"ai": 2, "finance": 2}


def test_minimum_category_candidates_can_come_from_below_threshold_pool() -> None:
    filtering = FilteringConfig(
        category_groups={
            "ai": CategoryGroupConfig(limit=4, min_items=2, categories=["ai"]),
            "finance": CategoryGroupConfig(limit=4, min_items=2, categories=["finance"]),
        },
    )
    orchestrator = make_orchestrator(filtering)
    analyzed_items = [
        make_item("ai-9", 9.0, "ai"),
        make_item("ai-8", 8.0, "ai"),
        make_item("finance-5", 5.0, "finance"),
        make_item("finance-4", 4.0, "finance"),
        make_item("finance-3", 3.0, "finance"),
    ]
    important_items = analyzed_items[:2]

    result = orchestrator.include_minimum_category_candidates(
        analyzed_items,
        important_items,
    )

    assert [item.id for item in result] == [
        "ai-9",
        "ai-8",
        "finance-5",
        "finance-4",
    ]


def test_candidate_limit_reserves_source_category_groups() -> None:
    filtering = FilteringConfig(
        max_candidates=6,
        category_groups={
            "ai": CategoryGroupConfig(limit=8, min_items=1, categories=["AI 与科技动态"]),
            "geo": CategoryGroupConfig(limit=8, min_items=1, categories=["地缘政治与国际关系"]),
        },
    )
    orchestrator = make_orchestrator(filtering)
    items = [
        make_item("geo-1", 0, "地缘政治与国际关系"),
        make_item("geo-2", 0, "地缘政治与国际关系"),
        make_item("geo-3", 0, "地缘政治与国际关系"),
        make_item("geo-4", 0, "地缘政治与国际关系"),
        make_item("geo-5", 0, "地缘政治与国际关系"),
        make_item("geo-6", 0, "地缘政治与国际关系"),
        make_item("ai-1", 0, "AI 与科技动态"),
        make_item("ai-2", 0, "AI 与科技动态"),
    ]
    for index, item in enumerate(items):
        item.metadata["priority"] = 5 if item.title.startswith("geo") else 3
        item.metadata["noise_level"] = "low"
        item.published_at = datetime.fromtimestamp(1000 + index, timezone.utc)

    result = orchestrator.apply_candidate_limit(items)

    assert len(result) == 6
    assert sum(item.metadata["category"] == "AI 与科技动态" for item in result) == 2


def test_promote_core_hotspots_fills_core_minimum() -> None:
    filtering = FilteringConfig(
        category_groups={
            "core": CategoryGroupConfig(
                limit=10,
                min_items=2,
                categories=["今日核心热点"],
            ),
            "geo": CategoryGroupConfig(
                limit=10,
                min_items=2,
                categories=["地缘政治与国际关系"],
            ),
        },
    )
    orchestrator = make_orchestrator(filtering)
    items = [
        make_item("core", 9.0, "今日核心热点"),
        make_item("geo-top", 8.5, "地缘政治与国际关系"),
        make_item("geo-low", 7.0, "地缘政治与国际关系"),
    ]
    items[1].metadata["importance_score"] = 9
    items[1].metadata["hotness_score"] = 8

    orchestrator.promote_core_hotspots(items)

    assert [item.metadata["category"] for item in items] == [
        "今日核心热点",
        "今日核心热点",
        "地缘政治与国际关系",
    ]
    assert items[1].metadata["original_category"] == "地缘政治与国际关系"


def test_max_items_works_without_category_groups() -> None:
    filtering = FilteringConfig(max_items=1)
    items = [make_item("lower", 7.0, None), make_item("higher", 9.0, None)]

    result = make_orchestrator(filtering).apply_balanced_digest(items)

    assert [item.id for item in result.items] == ["higher"]


def test_duplicate_category_warns_and_first_group_wins() -> None:
    filtering = FilteringConfig(
        category_groups={
            "first": CategoryGroupConfig(limit=1, categories=["shared"]),
            "second": CategoryGroupConfig(limit=2, categories=["shared"]),
        }
    )
    orchestrator = make_orchestrator(filtering)

    result = orchestrator.apply_balanced_digest(
        [make_item("top", 9.0, "shared"), make_item("second", 8.0, "shared")]
    )

    assert [item.id for item in result.items] == ["top"]
    assert result.duplicate_categories == ["shared"]
    assert "using 'first'" in orchestrator.console.export_text()


def test_topic_dedup_preserves_related_items(monkeypatch) -> None:
    class FakeClient:
        async def complete(self, system, user):  # type: ignore[no-untyped-def]
            return '{"duplicates": [[0, 1]]}'

    filtering = FilteringConfig()
    orchestrator = make_orchestrator(filtering)
    orchestrator.config = SimpleNamespace(
        filtering=filtering,
        ai=AIConfig(
            provider="openai",
            model="test",
            api_key_env="TEST_API_KEY",
            languages=[],
        ),
    )
    items = [
        make_item("primary", 9.0, "地缘政治与国际关系"),
        make_item("duplicate", 8.0, "地缘政治与国际关系"),
        make_item("separate", 7.0, "财经市场"),
    ]
    items[1].ai_summary = "duplicate summary"
    items[1].metadata["feed_name"] = "Duplicate Source"

    monkeypatch.setattr("src.orchestrator.create_ai_client", lambda config: FakeClient())

    result = asyncio.run(orchestrator.merge_topic_duplicates(items))

    assert [item.id for item in result] == ["primary", "separate"]
    assert result[0].metadata["related_items"] == [
        {
            "title": "duplicate",
            "source": "Duplicate Source",
            "url": "https://example.com/duplicate",
            "category": "地缘政治与国际关系",
            "summary": "duplicate summary",
            "publish_time": items[1].published_at.isoformat(),
        }
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_items": 0},
        {"default_group_limit": 0},
        {"category_groups": {"ai": {"limit": 0, "categories": ["ai"]}}},
        {"category_groups": {"ai": {"limit": 1, "categories": []}}},
    ],
)
def test_balanced_digest_config_rejects_non_positive_or_empty_limits(kwargs) -> None:
    with pytest.raises(ValidationError):
        FilteringConfig(**kwargs)


def test_run_applies_balanced_digest_before_enrichment(tmp_path, monkeypatch) -> None:
    config = Config(
        ai=AIConfig(
            provider="openai",
            model="test",
            api_key_env="TEST_API_KEY",
            languages=[],
        ),
        sources=SourcesConfig(),
        filtering=FilteringConfig(
            ai_score_threshold=7.0,
            max_items=1,
            category_groups={
                "ai": CategoryGroupConfig(limit=1, categories=["ai"]),
                "finance": CategoryGroupConfig(limit=1, categories=["finance"]),
            },
        ),
    )
    storage = SimpleNamespace()
    orchestrator = HorizonOrchestrator(config, storage)
    items = [
        make_item("ai", 9.0, "ai"),
        make_item("finance", 8.0, "finance"),
        make_item("below-threshold", 6.0, "ai"),
    ]
    enriched_ids: list[str] = []

    async def fetch_all_sources(since):  # type: ignore[no-untyped-def]
        return items

    async def analyze_content(input_items):  # type: ignore[no-untyped-def]
        return input_items

    async def merge_topic_duplicates(input_items):  # type: ignore[no-untyped-def]
        return input_items

    async def expand_twitter_discussion(input_items):  # type: ignore[no-untyped-def]
        return None

    async def enrich_important_items(input_items):  # type: ignore[no-untyped-def]
        enriched_ids.extend(item.id for item in input_items)

    monkeypatch.setattr(orchestrator, "fetch_all_sources", fetch_all_sources)
    monkeypatch.setattr(orchestrator, "_analyze_content", analyze_content)
    monkeypatch.setattr(orchestrator, "merge_topic_duplicates", merge_topic_duplicates)
    monkeypatch.setattr(orchestrator, "_expand_twitter_discussion", expand_twitter_discussion)
    monkeypatch.setattr(orchestrator, "_enrich_important_items", enrich_important_items)
    monkeypatch.chdir(tmp_path)

    asyncio.run(orchestrator.run())

    assert enriched_ids == ["ai"]
