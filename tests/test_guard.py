from ui.guard import RunLimiter, check_password


def test_password_check():
    assert check_password("anything", "") and check_password("s3cret", "s3cret")
    assert not check_password("wrong", "s3cret") and not check_password("", "s3cret")


def test_limiter_counts_per_day_and_resets(tmp_path):
    p = tmp_path / "usage.json"
    a = RunLimiter(p, max_per_day=2, today="2026-11-07")
    assert a.allowed() and a.remaining() == 2
    a.record(); a.record()
    assert not a.allowed() and a.remaining() == 0
    assert RunLimiter(p, max_per_day=2, today="2026-11-08").allowed()      # next day starts fresh


def test_limiter_survives_a_corrupt_file(tmp_path):
    p = tmp_path / "usage.json"
    p.write_text("{not json")
    assert RunLimiter(p, max_per_day=1, today="d").allowed()
