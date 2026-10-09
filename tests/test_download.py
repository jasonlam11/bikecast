import zipfile
from datetime import date
from pathlib import Path

import pytest

from bikecast.data.download import extract_csv, latest_full_month, month_keys, parse_listing

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def keys():
    return parse_listing((FIXTURES / "bucket_listing.xml").read_text())


def test_parse_listing_returns_all_keys(keys):
    assert len(keys) == 8
    assert "index.html" in keys


def test_month_keys_handles_all_three_file_formats(keys):
    out = month_keys(keys, "202401", "202609")
    assert out == {
        "202401": "202401-bluebikes-tripdata.zip",
        "202402": "202402-bluebikes-tripdata.zip",
        "202511": "202511-bluebikes-tripdata.csv.zip",
        "202609": "202609-bluebikes-tripdata.csv",
    }


def test_month_keys_respects_range(keys):
    assert list(month_keys(keys, "202312", "202401")) == ["202312", "202401"]


def test_duplicate_month_raises():
    with pytest.raises(ValueError):
        month_keys(["202401-bluebikes-tripdata.zip", "202401-bluebikes-tripdata.csv"], "2024", "9")


@pytest.mark.parametrize(
    ("today", "expected"),
    [(date(2026, 10, 8), "202609"), (date(2026, 1, 3), "202512")],
)
def test_latest_full_month(today, expected):
    assert latest_full_month(today) == expected


def test_extract_csv_skips_macos_metadata(tmp_path):
    zip_path = tmp_path / "x.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("202401-bluebikes-tripdata.csv", "a,b\n1,2\n")
        zf.writestr("__MACOSX/._202401-bluebikes-tripdata.csv", "junk")
    out = tmp_path / "202401.csv"
    extract_csv(zip_path, out)
    assert out.read_text() == "a,b\n1,2\n"
