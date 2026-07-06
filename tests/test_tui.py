"""Phase 2, Task 6: the Textual live-caption TUI (LiveApp + TextualLiveView).

Textual apps are driven through their async ``run_test`` harness; each test wraps
one such body in ``asyncio.run`` so no ``pytest-asyncio`` plugin is needed. The
load-bearing guarantees:

- minimal-redraw config is actually in force (frame cap pinned, animations off);
- committed captions group into utterance lines and the finalize pass swaps in
  the authoritative diarized transcript;
- Ctrl-C crosses to the capture side via the ``stop`` callback and finalizes
  rather than aborting (it is a captured key event, not a ``KeyboardInterrupt``);
- events cross from a worker thread onto the UI, and the meeting→finalize→exit
  flow runs end-to-end.
"""

import asyncio
import threading

import textual.constants as tconst

from stenograf.asr.base import Word
from stenograf.capture.base import Channel
from stenograf.config import Language, MeetingProfile
from stenograf.live import StreamingUpdate
from stenograf.transcript import Transcript, TranscriptEntry
from stenograf.tui import LiveApp, TextualLiveView


def _run(body) -> None:
    asyncio.run(body())


def _app(**kw) -> LiveApp:
    kw.setdefault("profile", MeetingProfile(local_speakers=1, remote_speakers=1))
    return LiveApp(**kw)


def _w(text: str, start: float, end: float) -> Word:
    return Word(text, start, end)


class TestMinimalRedraw:
    def test_frame_cap_and_animations_are_pinned(self):
        # Importing stenograf.tui sets TEXTUAL_FPS before textual imports, so the
        # frame cap is baked low; animation_level is forced off on mount.
        assert tconst.MAX_FPS == 15

        async def body():
            app = _app()
            async with app.run_test():
                assert app.animation_level == "none"

        _run(body)


class TestCaptions:
    def test_committed_words_group_into_utterance_lines(self):
        async def body():
            app = _app()
            async with app.run_test() as pilot:
                app.push_committed(Channel.MIC, [_w("guten", 0.1, 0.5), _w("Morgen", 0.5, 0.9)])
                app.push_committed(Channel.MIC, [_w("zusammen", 1.0, 1.4)])  # small gap → same line
                await pilot.pause()
                assert app.committed_lines == []  # still open, not yet flushed to the log
                app.push_committed(Channel.MIC, [_w("später", 4.0, 4.4)])  # >1.5 s pause → break
                await pilot.pause()
                assert app.committed_lines == ["You  guten Morgen zusammen"]

        _run(body)

    def test_channel_switch_flushes_the_open_line(self):
        async def body():
            app = _app()
            async with app.run_test() as pilot:
                app.push_committed(Channel.MIC, [_w("hi", 0.0, 0.4)])
                app.push_committed(Channel.SYSTEM, [_w("hallo", 2.0, 2.4)])
                await pilot.pause()
                # The mic line flushed on the switch; the system line is still open.
                assert app.committed_lines == ["You  hi"]

        _run(body)

    def test_interim_tail_shows_open_line_and_grey_tail_then_clears(self):
        async def body():
            app = _app()
            async with app.run_test() as pilot:
                app.push_committed(Channel.MIC, [_w("guten", 0.1, 0.5)])
                app.push_interim(Channel.MIC, "Morgen zusammen")
                await pilot.pause()
                interim = app.query_one("#interim").render()
                shown = interim.plain if hasattr(interim, "plain") else str(interim)
                assert "You" in shown and "guten" in shown and "Morgen zusammen" in shown
                app.push_interim(Channel.MIC, "")  # empty interim drops the grey tail
                await pilot.pause()
                shown2 = app.query_one("#interim").render()
                shown2 = shown2.plain if hasattr(shown2, "plain") else str(shown2)
                assert "Morgen zusammen" not in shown2

        _run(body)


class TestHeader:
    def test_header_tracks_language_profile_and_phase(self):
        async def body():
            app = _app(profile=MeetingProfile(local_speakers=0, remote_speakers=2))
            async with app.run_test():
                assert "● REC" in app.header_text()
                assert "remote 2" in app.header_text()
                assert "  —  " in f"  {app.header_text()}  "  # language unknown
                app.push_language(Language.GERMAN)
                assert " de " in f" {app.header_text()} "
                app.push_finalizing()
                assert "finalizing" in app.header_text()

        _run(body)


class TestFinalizeSwap:
    def test_finalized_replaces_live_captions_with_the_diarized_transcript(self):
        async def body():
            app = _app()
            async with app.run_test() as pilot:
                app.push_committed(Channel.MIC, [_w("wort", 0.0, 0.4)])
                app.push_interim(Channel.SYSTEM, "grey")
                await pilot.pause()
                transcript = Transcript(
                    language=Language.GERMAN,
                    profile=app._profile,
                    entries=[
                        TranscriptEntry("Local-1", "guten Morgen", 0.0, 0.9),
                        TranscriptEntry("Remote-1", "hallo", 2.0, 2.4),
                    ],
                )
                app.push_finalized(transcript)
                await pilot.pause()
                # The live channel-coarse captions are gone; the diarized speakers win.
                assert app.committed_lines == ["Local-1  guten Morgen", "Remote-1  hallo"]
                assert app._phase == "done"
                interim = app.query_one("#interim").render()
                interim = interim.plain if hasattr(interim, "plain") else str(interim)
                assert interim.strip() == ""
                assert " de " in f" {app.header_text()} "

        _run(body)


class TestQuitBinding:
    def test_ctrl_c_while_capturing_stops_but_does_not_exit(self):
        async def body():
            calls = []
            app = _app(stop=lambda: calls.append(1))
            async with app.run_test() as pilot:
                await pilot.press("ctrl+c")
                await pilot.pause()
                assert calls == [1]  # crossed to the capture side
                assert app._phase == "finalizing"
                assert app.is_running  # graceful finalize, not an abort

        _run(body)

    def test_second_ctrl_c_forces_exit(self):
        async def body():
            app = _app(stop=lambda: None)
            async with app.run_test() as pilot:
                await pilot.press("ctrl+c")  # → finalizing
                await pilot.press("ctrl+c")  # impatient → exit
                await pilot.pause()
                assert not app.is_running

        _run(body)

    def test_ctrl_c_without_a_stop_callback_exits(self):
        async def body():
            app = _app()  # no stop wired
            async with app.run_test() as pilot:
                await pilot.press("ctrl+c")
                await pilot.pause()
                assert not app.is_running

        _run(body)

    def test_q_exits_once_finalized(self):
        async def body():
            app = _app(stop=lambda: None)
            async with app.run_test() as pilot:
                app.push_finalized(Transcript(language=None, profile=app._profile, entries=[]))
                await pilot.pause()
                assert app._phase == "done"
                await pilot.press("q")
                await pilot.pause()
                assert not app.is_running

        _run(body)


class TestViewMarshalling:
    def test_events_cross_from_a_worker_thread_onto_the_ui(self):
        async def body():
            view = TextualLiveView(MeetingProfile(local_speakers=1, remote_speakers=0))
            async with view.app.run_test() as pilot:
                done = threading.Event()

                def worker():  # a stand-in for the LiveWorker thread
                    view.commit(Channel.MIC, [_w("hallo", 0.0, 0.4)])
                    view.commit(Channel.MIC, [_w("welt", 3.0, 3.4)])  # pause → flush first line
                    done.set()

                threading.Thread(target=worker, daemon=True).start()
                for _ in range(100):
                    await pilot.pause()
                    if done.is_set() and app_has(view, "You  hallo"):
                        break
                assert "You  hallo" in view.app.committed_lines

        _run(body)

    def test_update_before_mount_is_dropped(self):
        # No running app yet → the view must not raise, just drop the update.
        view = TextualLiveView(MeetingProfile(local_speakers=1, remote_speakers=0))
        view.update(Channel.MIC, StreamingUpdate((_w("x", 0.0, 0.1),), "y"))


class TestServeIntegration:
    def test_meeting_thread_finalizes_and_the_app_waits_to_exit(self):
        async def body():
            view = TextualLiveView(MeetingProfile(local_speakers=1, remote_speakers=1))
            transcript = Transcript(
                language=Language.GERMAN,
                profile=MeetingProfile(local_speakers=1, remote_speakers=1),
                entries=[TranscriptEntry("Local-1", "hallo welt", 0.0, 0.8)],
            )

            def meeting() -> Transcript:
                # Stand-in for recorder.run: stream a caption, then return the
                # authoritative transcript (as the on-stop finalize would).
                view.commit(Channel.MIC, [_w("hallo", 0.0, 0.4)])
                return transcript

            result: dict[str, object] = {}
            view._arm_meeting(meeting, result)  # the same wiring serve() uses
            async with view.app.run_test() as pilot:
                for _ in range(200):
                    await pilot.pause()
                    if view.app._phase == "done":
                        break
                # The finalize swap ran; the app stays up showing the result.
                assert result["transcript"] is transcript
                assert view.app.committed_lines == ["Local-1  hallo welt"]
                assert view.app.is_running
                await pilot.press("q")  # user dismisses
                await pilot.pause()
                assert not view.app.is_running

        _run(body)


def app_has(view: TextualLiveView, line: str) -> bool:
    return line in view.app.committed_lines
