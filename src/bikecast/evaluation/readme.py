"""Copy generated results into README.md between marker comments.

README numbers must come from reports/results.md and reports/rebalancing.md, never typed by hand.
Each block in the README looks like:

    <!-- BEGIN results:station -->
    ...generated content...
    <!-- END results:station -->

`make readme` refreshes every block; tests/test_readme.py fails if the README is out of date.
"""

import re

from bikecast import config

README = config.ROOT / "README.md"
SOURCES = {
    "results": config.REPORTS / "results.md",
    "rebalancing": config.REPORTS / "rebalancing.md",
}
# block name -> (source, heading of the section to copy)
BLOCKS = {
    "results:station": ("results", "Station level (per station-hour)"),
    "results:system": ("results", "System-wide (hourly totals across all stations)"),
    "results:per_station": ("results", "Per-station skill vs historical average"),
    "results:sensitivity": ("results", "Sensitivity: forecast weather vs actual weather"),
    "rebalancing:summary": ("rebalancing", "Summary"),
    "rebalancing:drains": ("rebalancing", "Stations that drain"),
}
BLOCK_RE = re.compile(r"(<!-- BEGIN (?P<name>[\w:]+) -->\n)(.*?)(<!-- END (?P=name) -->)", re.S)


def section(markdown: str, heading: str) -> str:
    """Body of a `## heading` section, up to the next `## ` heading, without the title line."""
    lines = markdown.splitlines()
    try:
        start = lines.index(f"## {heading}") + 1
    except ValueError as err:
        raise ValueError(f"Section '## {heading}' not found") from err
    end = next((i for i in range(start, len(lines)) if lines[i].startswith("## ")), len(lines))
    body = "\n".join(lines[start:end]).strip()
    # Image links in reports/ are relative to reports/; the README lives one level up.
    return body.replace("](figures/", "](reports/figures/")


def render(readme: str, sources: dict[str, str]) -> str:
    def fill(m: re.Match) -> str:
        name = m.group("name")
        if name not in BLOCKS:
            raise ValueError(f"Unknown README block '{name}'")
        source, heading = BLOCKS[name]
        return f"{m.group(1)}{section(sources[source], heading)}\n{m.group(4)}"

    return BLOCK_RE.sub(fill, readme)


def load_sources() -> dict[str, str]:
    return {k: p.read_text() for k, p in SOURCES.items()}


def main() -> None:
    README.write_text(render(README.read_text(), load_sources()))
    print(f"Updated {README}")


if __name__ == "__main__":
    main()
