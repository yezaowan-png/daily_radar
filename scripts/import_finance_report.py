#!/usr/bin/env python3
"""Import a local finance Markdown report into the GitHub Pages docs tree."""

from __future__ import annotations

import argparse
import html
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

try:
    import markdown
except ImportError:  # pragma: no cover - local automation env normally has markdown.
    markdown = None


DEFAULT_SOURCE_DIR = Path(
    "/Users/caleb/projects/claude_prj/claude_agent/finance_news/output"
)
DEFAULT_DOCS_DIR = Path("docs")
REPORT_SUFFIX = "invest_research_report.md"


@dataclass(frozen=True)
class ImportedReport:
    source_path: Path
    markdown_path: Path
    html_path: Path
    target_date: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render yesterday's local invest_research_report.md into docs/finance."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help=f"Finance report output root. Default: {DEFAULT_SOURCE_DIR}",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=DEFAULT_DOCS_DIR,
        help="GitHub Pages docs directory. Default: docs",
    )
    parser.add_argument(
        "--date",
        help="Target report date in YYYY-MM-DD. Default: yesterday in Asia/Shanghai.",
    )
    parser.add_argument(
        "--timezone",
        default="Asia/Shanghai",
        help="Timezone used to compute yesterday. Default: Asia/Shanghai",
    )
    return parser.parse_args()


def default_target_date(timezone: str) -> str:
    now = datetime.now(ZoneInfo(timezone))
    return (now.date() - timedelta(days=1)).isoformat()


def find_report(source_dir: Path, target_date: str) -> Path | None:
    if not source_dir.exists():
        return None

    candidates = [
        path
        for path in source_dir.rglob(f"*_{REPORT_SUFFIX}")
        if target_date in path.name
    ]
    candidates.extend(
        path
        for path in source_dir.rglob(REPORT_SUFFIX)
        if target_date in path.name
    )
    if not candidates:
        return None

    # The finance pipeline can rerun in the same evening; use the latest file.
    return max(candidates, key=lambda path: path.stat().st_mtime)


def extract_title(markdown_text: str, target_date: str) -> str:
    for line in markdown_text.splitlines():
        match = re.match(r"^#\s+(.+?)\s*$", line)
        if match:
            return match.group(1)
    return f"投资研究报告 {target_date}"


def strip_first_h1(markdown_text: str) -> str:
    return re.sub(r"^#\s+.+?\s*\n+", "", markdown_text, count=1)


def fallback_markdown_to_html(markdown_text: str) -> str:
    escaped = html.escape(markdown_text)
    paragraphs = []
    for block in re.split(r"\n{2,}", escaped):
        block = block.strip()
        if not block:
            continue
        paragraphs.append(f"<p>{block.replace(chr(10), '<br>')}</p>")
    return "\n".join(paragraphs)


def render_html(markdown_text: str, title: str, source_path: Path, target_date: str) -> str:
    markdown_body = strip_first_h1(markdown_text)
    if markdown is not None:
        body = markdown.markdown(
            markdown_body,
            extensions=["extra", "tables", "fenced_code", "toc", "sane_lists"],
            output_format="html5",
        )
    else:
        body = fallback_markdown_to_html(markdown_body)

    escaped_title = html.escape(title)
    escaped_source = html.escape(str(source_path))
    escaped_date = html.escape(target_date)
    generated_at = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escaped_title}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #18212f;
      --muted: #667085;
      --line: #d9e0ea;
      --paper: #fffdf8;
      --wash: #f3f6fb;
      --accent: #a44718;
      --accent-soft: #fff1df;
    }}
    body {{
      margin: 0;
      background:
        radial-gradient(circle at top left, rgba(255, 189, 89, .22), transparent 30rem),
        linear-gradient(135deg, #f8fafc 0%, #fff7ed 100%);
      color: var(--ink);
      font-family: "Source Han Serif SC", "Noto Serif CJK SC", "Songti SC", Georgia, serif;
      line-height: 1.78;
    }}
    .shell {{
      max-width: 920px;
      margin: 0 auto;
      padding: 28px 18px 56px;
    }}
    .nav {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-bottom: 18px;
      font-family: "Avenir Next", "Helvetica Neue", sans-serif;
      font-size: 14px;
    }}
    .nav a {{
      color: var(--accent);
      text-decoration: none;
      border-bottom: 1px solid rgba(164, 71, 24, .25);
    }}
    article {{
      background: rgba(255, 253, 248, .9);
      border: 1px solid var(--line);
      box-shadow: 0 24px 70px rgba(24, 33, 47, .08);
      border-radius: 24px;
      padding: clamp(24px, 4vw, 52px);
    }}
    .meta {{
      color: var(--muted);
      font-family: "Avenir Next", "Helvetica Neue", sans-serif;
      font-size: 14px;
      margin-bottom: 24px;
    }}
    h1, h2, h3 {{
      line-height: 1.25;
      letter-spacing: -.02em;
    }}
    h1 {{
      font-size: clamp(30px, 6vw, 52px);
      margin: 0 0 16px;
    }}
    h2 {{
      border-top: 1px solid var(--line);
      margin-top: 42px;
      padding-top: 28px;
    }}
    a {{
      color: var(--accent);
    }}
    blockquote {{
      border-left: 4px solid var(--accent);
      background: var(--accent-soft);
      margin: 24px 0;
      padding: 12px 18px;
      border-radius: 0 14px 14px 0;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      display: block;
      overflow-x: auto;
      margin: 24px 0;
      font-family: "Avenir Next", "Helvetica Neue", sans-serif;
      font-size: 14px;
    }}
    th, td {{
      border: 1px solid var(--line);
      padding: 9px 11px;
      vertical-align: top;
    }}
    th {{
      background: var(--wash);
      text-align: left;
    }}
    code, pre {{
      font-family: "SFMono-Regular", Consolas, monospace;
    }}
    pre {{
      overflow-x: auto;
      background: #101828;
      color: #f8fafc;
      padding: 16px;
      border-radius: 14px;
    }}
    img {{
      max-width: 100%;
    }}
    @media (max-width: 640px) {{
      .shell {{
        padding: 14px 10px 32px;
      }}
      article {{
        border-radius: 18px;
      }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <nav class="nav">
      <a href="../">返回每日信息雷达</a>
      <a href="./">投资研究报告归档</a>
    </nav>
    <article>
      <h1>{escaped_title}</h1>
      <div class="meta">报告日期：{escaped_date} · 来源文件：{escaped_source} · 渲染时间：{generated_at}</div>
      {body}
    </article>
  </main>
</body>
</html>
"""


def render_index(finance_dir: Path) -> None:
    reports = sorted(finance_dir.glob("*.html"), reverse=True)
    report_items = "\n".join(
        f'      <li><a href="{html.escape(path.name)}">{html.escape(path.stem[:10])}</a></li>'
        for path in reports
        if path.name != "index.html"
    )
    if not report_items:
        report_items = "      <li><em>暂无投资研究报告</em></li>"

    index_html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>投资研究报告归档</title>
  <style>
    body {{
      margin: 0;
      background: linear-gradient(135deg, #f8fafc 0%, #fff7ed 100%);
      color: #18212f;
      font-family: "Avenir Next", "Helvetica Neue", sans-serif;
    }}
    main {{
      max-width: 760px;
      margin: 0 auto;
      padding: 42px 18px;
    }}
    section {{
      background: rgba(255, 253, 248, .9);
      border: 1px solid #d9e0ea;
      border-radius: 22px;
      padding: 28px;
      box-shadow: 0 24px 70px rgba(24, 33, 47, .08);
    }}
    h1 {{
      margin-top: 0;
      font-size: clamp(30px, 5vw, 46px);
    }}
    a {{
      color: #a44718;
      text-decoration: none;
      border-bottom: 1px solid rgba(164, 71, 24, .25);
    }}
    li {{
      margin: 12px 0;
    }}
  </style>
</head>
<body>
  <main>
    <section>
      <p><a href="../">返回每日信息雷达</a></p>
      <h1>投资研究报告归档</h1>
      <ul>
{report_items}
      </ul>
    </section>
  </main>
</body>
</html>
"""
    (finance_dir / "index.html").write_text(index_html, encoding="utf-8")


def import_report(source_dir: Path, docs_dir: Path, target_date: str) -> ImportedReport | None:
    source_path = find_report(source_dir, target_date)
    if source_path is None:
        print(f"No finance report found for {target_date}; skipping finance import.")
        return None

    finance_dir = docs_dir / "finance"
    finance_dir.mkdir(parents=True, exist_ok=True)
    output_stem = f"{target_date}-invest-research-report"
    markdown_path = finance_dir / f"{output_stem}.md"
    html_path = finance_dir / f"{output_stem}.html"

    markdown_text = source_path.read_text(encoding="utf-8")
    title = extract_title(markdown_text, target_date)
    html_text = render_html(markdown_text, title, source_path, target_date)

    shutil.copyfile(source_path, markdown_path)
    html_path.write_text(html_text, encoding="utf-8")
    render_index(finance_dir)

    print(f"Imported finance report for {target_date}")
    print(f"Source: {source_path}")
    print(f"Markdown: {markdown_path}")
    print(f"HTML: {html_path}")
    return ImportedReport(source_path, markdown_path, html_path, target_date)


def main() -> int:
    args = parse_args()
    target_date = args.date or default_target_date(args.timezone)
    import_report(args.source_dir, args.docs_dir, target_date)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
