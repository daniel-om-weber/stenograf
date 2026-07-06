"""Phase 2, Task 5: the live-caption views (LiveView interface + PlainLiveView).

PlainLiveView is exercised through an injected ``echo`` recorder that mirrors
``click.echo``'s ``message``/``nl``/``err`` semantics exactly (message text plus
an optional trailing newline, routed to out or err), so the assertions read the
literal bytes the view would stream. The load-bearing guarantees:

- committed words stream onto one per-channel line, breaking on a channel change
  or a pause (utterance-sized paragraphs), and never re-render;
- the provisional grey tail is dropped (a non-TTY stream can't erase it);
- an out-of-band notice always closes the open caption line first, so a status
  or finalize line never fuses onto a caption.
"""

import threading

from stenograf.asr.base import Word
from stenograf.capture.base import Channel
from stenograf.config import Language, MeetingProfile
from stenograf.live import StreamingUpdate
from stenograf.transcript import Transcript, TranscriptEntry
from stenograf.view import LiveView, PlainLiveView


class EchoRecorder:
    """Stand-in for ``click.echo``: message + optional newline, split out/err."""

    def __init__(self) -> None:
        self.out = ""
        self.err = ""

    def __call__(self, message: object = "", *, nl: bool = True, err: bool = False) -> None:
        text = ("" if message is None else str(message)) + ("\n" if nl else "")
        if err:
            self.err += text
        else:
            self.out += text


def _view() -> tuple[PlainLiveView, EchoRecorder]:
    rec = EchoRecorder()
    return PlainLiveView(echo=rec), rec


def _w(text: str, start: float, end: float) -> Word:
    return Word(text, start, end)


class TestPlainCaptions:
    def test_commit_streams_onto_one_line(self):
        view, rec = _view()
        view.commit(Channel.MIC, [_w("hallo", 0.1, 0.5), _w("welt", 0.5, 0.9)])
        view.commit(Channel.MIC, [_w("wie", 1.0, 1.3)])  # small gap, same channel → continues
        view.status("done")  # a notice closes the still-open line
        assert rec.out == "[0:00] You: hallo welt wie\ndone\n"

    def test_channel_switch_breaks_the_line(self):
        view, rec = _view()
        view.commit(Channel.MIC, [_w("hi", 0.0, 0.4)])
        view.commit(Channel.SYSTEM, [_w("hallo", 2.0, 2.4)])
        view.status("end")
        assert rec.out == "[0:00] You: hi\n[0:02] Remote: hallo\nend\n"

    def test_a_pause_breaks_the_line_for_the_same_channel(self):
        view, rec = _view()
        view.commit(Channel.MIC, [_w("a", 0.0, 0.5)])
        view.commit(Channel.MIC, [_w("b", 3.0, 3.4)])  # gap 2.5 s > _LINE_GAP → new line
        view.status("end")
        assert rec.out == "[0:00] You: a\n[0:03] You: b\nend\n"

    def test_labels_are_channel_coarse(self):
        view, rec = _view()
        view.commit(Channel.MIC, [_w("x", 0.0, 0.1)])
        view.commit(Channel.SYSTEM, [_w("y", 5.0, 5.1)])
        assert "You:" in rec.out and "Remote:" in rec.out

    def test_empty_commit_is_ignored(self):
        view, rec = _view()
        view.commit(Channel.MIC, [])
        assert rec.out == ""

    def test_timestamp_spans_past_a_minute(self):
        view, rec = _view()
        view.commit(Channel.MIC, [_w("spät", 65.0, 65.4)])
        view.status("end")
        assert rec.out == "[1:05] You: spät\nend\n"


class TestPlainInterimAndUpdate:
    def test_interim_is_dropped(self):
        view, rec = _view()
        view.interim(Channel.MIC, "provisional tail")
        assert rec.out == ""

    def test_update_commits_but_never_shows_interim(self):
        view, rec = _view()
        view.update(Channel.MIC, StreamingUpdate((_w("hi", 0.1, 0.4),), "there"))
        view.status("end")
        assert rec.out == "[0:00] You: hi\nend\n"

    def test_update_with_nothing_committed_prints_nothing(self):
        view, rec = _view()
        view.update(Channel.MIC, StreamingUpdate((), "only interim"))
        assert rec.out == ""


class TestPlainNotices:
    def _transcript(self, entries: list[TranscriptEntry]) -> Transcript:
        return Transcript(language=None, profile=MeetingProfile(), entries=entries)

    def test_language_notice(self):
        view, rec = _view()
        view.language(Language.GERMAN)
        assert rec.out == "language: de\n"

    def test_finalizing_and_finalized_close_the_caption_line(self):
        view, rec = _view()
        view.commit(Channel.MIC, [_w("wort", 0.0, 0.4)])
        view.finalizing()
        view.finalized(
            self._transcript(
                [
                    TranscriptEntry("Local-1", "wort", 0.0, 0.4),
                    TranscriptEntry("Remote-1", "ja", 1.0, 1.2),
                ]
            )
        )
        assert rec.out == (
            "[0:00] You: wort\n"
            "finalizing — the on-stop pass replaces the live captions\n"
            "finalized: 2 entries, 2 speakers\n"
        )

    def test_error_goes_to_stderr_and_closes_the_line(self):
        view, rec = _view()
        view.commit(Channel.MIC, [_w("wort", 0.0, 0.4)])
        view.error("live pass stopped early")
        assert rec.out == "[0:00] You: wort\n"  # caption line closed on stdout
        assert "error: live pass stopped early" in rec.err


class TestLiveViewBase:
    def test_base_view_is_a_no_op_context_manager(self):
        # The bare interface doubles as a null view: every event is a safe no-op.
        with LiveView() as view:
            view.update(Channel.MIC, StreamingUpdate((_w("x", 0.0, 0.1),), "y"))
            view.status("s")
            view.language(Language.ENGLISH)
            view.finalizing()
            view.error("e")

    def test_update_matches_the_on_update_signature(self):
        # A view plugs straight into the worker: on_update = view.update.
        seen: list[tuple[Channel, StreamingUpdate]] = []

        class Recording(LiveView):
            def update(self, channel: Channel, update: StreamingUpdate) -> None:
                seen.append((channel, update))

        on_update = Recording().update
        update = StreamingUpdate((_w("hi", 0.0, 0.2),), "")
        on_update(Channel.SYSTEM, update)
        assert seen == [(Channel.SYSTEM, update)]


class TestPlainThreadSafety:
    def test_concurrent_commits_and_notices_never_interleave(self):
        # Commits (worker thread) and notices (main thread) share one lock, so
        # every emitted line is well-formed: a caption, a notice, or empty.
        view, rec = _view()
        barrier = threading.Barrier(2)

        def commits() -> None:
            barrier.wait()
            for i in range(200):
                view.commit(Channel.MIC, [_w("w", float(i), i + 0.4)])

        def notices() -> None:
            barrier.wait()
            for _ in range(200):
                view.status("tick")

        threads = [threading.Thread(target=commits), threading.Thread(target=notices)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        view.status("end")

        assert all(not t.is_alive() for t in threads)
        for line in rec.out.splitlines():
            assert line == "" or line == "tick" or line == "end" or line.startswith("["), line
