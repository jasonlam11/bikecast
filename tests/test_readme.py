import pytest

from bikecast.evaluation.readme import BLOCKS, README, load_sources, render, section

MD = "# Title\n\n## A\n\nline a\n![x](figures/a.png)\n\n## B\n\nline b\n"


def test_section_extracts_body_and_fixes_figure_paths():
    assert section(MD, "A") == "line a\n![x](reports/figures/a.png)"
    assert section(MD, "B") == "line b"
    with pytest.raises(ValueError):
        section(MD, "C")


def test_render_replaces_only_marked_blocks(monkeypatch):
    monkeypatch.setitem(BLOCKS, "t:a", ("src", "A"))
    readme = "intro\n<!-- BEGIN t:a -->\nstale\n<!-- END t:a -->\noutro\n"
    out = render(readme, {"src": MD})
    assert out == (
        "intro\n<!-- BEGIN t:a -->\nline a\n![x](reports/figures/a.png)\n<!-- END t:a -->\noutro\n"
    )


def test_readme_matches_generated_reports():
    """Fails if README numbers drift from reports/; run `make readme` to fix."""
    current = README.read_text()
    assert "<!-- BEGIN results:station -->" in current
    assert render(current, load_sources()) == current
