from traductor_voz.segments import Phrase, merge_segments, reference_window, split_text


def test_merge_joins_cut_sentences():
    phrases = merge_segments(
        [
            (0.0, 2.0, " Hoy fui al mercado"),
            (2.1, 3.5, " y compré pan."),
            (3.6, 5.0, " Después volví."),
            (6.5, 7.0, " ¡Qué día"),
            (7.0, 8.0, ""),
        ]
    )
    assert [p.text for p in phrases] == ["Hoy fui al mercado y compré pan.", "Después volví.", "¡Qué día"]
    assert (phrases[0].start, phrases[0].end) == (0.0, 3.5)


def test_merge_respects_pauses_and_max_length():
    long_pause = merge_segments([(0, 2, "uno"), (3, 4, "dos")])
    assert len(long_pause) == 2
    too_long = merge_segments([(0, 10, "uno"), (10.1, 14, "dos")], max_len=12)
    assert len(too_long) == 2


def phrases(*spans):
    return [Phrase(s, e, f"f{i}") for i, (s, e) in enumerate(spans)]


def test_reference_window_uses_own_phrase_when_long_enough():
    ps = phrases((0, 7), (7.5, 9), (10, 18))
    assert reference_window(ps, 0, 6) == (0, 0)


def test_reference_window_grows_towards_closest_neighbour():
    ps = phrases((0, 3), (3.2, 5), (8, 11))
    assert reference_window(ps, 1, 4) == (0, 1)  # la anterior está más cerca
    assert reference_window(ps, 1, 7) == (0, 2)


def test_reference_window_stops_at_max_and_edges():
    ps = phrases((0, 2), (10, 20))
    assert reference_window(ps, 0, 6, max_sec=15) == (0, 0)
    single = phrases((0, 1))
    assert reference_window(single, 0, 6) == (0, 0)


def test_split_text():
    assert split_text("  Hola   mundo. ") == ["Hola mundo."]
    assert split_text("") == []
    text = "Primera oración bastante larga. " * 12 + "Final, con coma " + "palabra " * 60
    chunks = split_text(text, max_chars=120)
    assert all(len(c) <= 121 for c in chunks)
    assert " ".join(chunks).split() == text.split()
