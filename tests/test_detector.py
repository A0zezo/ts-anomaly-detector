from detector import detect, make_series, score


def test_detects_most_injected_spikes():
    x, truth = make_series()
    flags = detect(x)
    _, recall, _ = score(flags, truth)
    assert recall >= 0.75


def test_clean_series_has_few_flags():
    rnd_series = [50 + (i % 24) for i in range(200)]
    flags = detect(rnd_series, threshold=4.0)
    assert sum(flags[60:]) <= 5


def test_warmup_not_flagged():
    x, _ = make_series()
    flags = detect(x)
    assert not any(flags[:5])


def test_score_perfect_match():
    assert score([False, True], {1}) == (1.0, 1.0, 1)
