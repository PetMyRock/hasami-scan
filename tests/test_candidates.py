from hasami.core import candidates


def test_nothing_is_recorded_outside_collecting():
    candidates.record(kind="x")  # no collector: no error, nothing kept
    assert not candidates.active()


def test_collecting_gathers_records_and_nests():
    with candidates.collecting() as outer:
        assert candidates.active()
        candidates.record(kind="a")
        with candidates.collecting() as inner:
            candidates.record(kind="b")
        candidates.record(kind="c")
    assert [r["kind"] for r in outer] == ["a", "c"]
    assert [r["kind"] for r in inner] == ["b"]
    assert not candidates.active()
