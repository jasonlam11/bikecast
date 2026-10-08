from datetime import timedelta

from bikecast import config


def test_test_weeks_are_mondays_in_order():
    assert all(w.weekday() == 0 for w in config.TEST_WEEKS)
    weeks = config.TEST_WEEKS
    assert weeks == sorted(weeks)


def test_test_weeks_come_after_station_ranking_window():
    assert min(config.TEST_WEEKS) >= config.STATION_RANK_END


def test_test_weeks_do_not_overlap():
    for a, b in zip(config.TEST_WEEKS, config.TEST_WEEKS[1:], strict=False):
        assert b - a >= timedelta(days=7)
