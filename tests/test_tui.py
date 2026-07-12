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
from stenograf.tui import LiveApp, Phase, TextualLiveView


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

    def test_window_sized_batch_flushes_to_the_log_immediately(self):
        # The window pass commits a whole ~30 s window per batch. Past the size
        # cap it must land in the scrolling log at once — not accumulate on the
        # open line inside the height-capped (bottom-clipped) interim area, which
        # froze the UI for minutes during sustained remote speech.
        async def body():
            app = _app()
            async with app.run_test() as pilot:
                words = [_w(f"wort{i}", i * 0.3, i * 0.3 + 0.25) for i in range(60)]
                app.push_committed(Channel.SYSTEM, words)
                await pilot.pause()
                assert len(app.committed_lines) == 1
                assert app.committed_lines[0].startswith("Remote  wort0")
                assert app.committed_lines[0].endswith("wort59")
                # The open line is empty again, so the next window starts fresh
                # instead of continuing a line that is already in the log.
                app.push_committed(Channel.SYSTEM, [_w("weiter", 18.2, 18.5)])
                await pilot.pause()
                assert app._open_words == ["weiter"]

        _run(body)

    def test_small_commits_flush_once_the_open_line_exceeds_the_cap(self):
        # Speculative-mode commits (a few words each) still merge into utterance
        # paragraphs — but the merged line is bounded: crossing the cap moves the
        # whole paragraph into the log in one piece.
        async def body():
            app = _app()
            async with app.run_test() as pilot:
                words = [_w(f"wort{i}", i * 0.3, i * 0.3 + 0.25) for i in range(40)]
                app.push_committed(Channel.MIC, words[:30])  # under the cap → stays open
                await pilot.pause()
                assert app.committed_lines == []
                app.push_committed(Channel.MIC, words[30:])  # crosses the cap → flush
                await pilot.pause()
                assert len(app.committed_lines) == 1
                assert "wort0" in app.committed_lines[0]
                assert app.committed_lines[0].endswith("wort39")

        _run(body)

    def test_idle_open_line_flushes_on_the_tick(self):
        # The last window of a stretch of speech must not sit in the interim area
        # until a future commit displaces it — the 1 Hz tick flushes it once no
        # new commit has arrived for the idle threshold.
        async def body():
            app = _app()
            async with app.run_test() as pilot:
                app.push_committed(Channel.SYSTEM, [_w("hallo", 0.0, 0.4)])
                await pilot.pause()
                app._tick()  # commit is fresh → no flush yet
                assert app.committed_lines == []
                app._last_commit_at -= 6.0  # age the commit past the threshold
                app._tick()
                await pilot.pause()
                assert app.committed_lines == ["Remote  hallo"]
                interim = app.query_one("#interim").render()
                shown = interim.plain if hasattr(interim, "plain") else str(interim)
                assert "hallo" not in shown  # flushed line left the interim area

        _run(body)

    def test_interim_shows_only_the_tail_of_a_long_open_line(self):
        # Defensive: a sub-cap open line can still outgrow the interim area's
        # four rows, which clip at the bottom — render only its freshest tail.
        async def body():
            app = _app()
            async with app.run_test() as pilot:
                words = [_w(f"wort{i}", i * 0.3, i * 0.3 + 0.25) for i in range(33)]
                app.push_committed(Channel.MIC, words)  # ~220 chars: under the flush cap
                await pilot.pause()
                assert app.committed_lines == []  # still open
                interim = app.query_one("#interim").render()
                shown = interim.plain if hasattr(interim, "plain") else str(interim)
                assert "…" in shown and shown.rstrip().endswith("wort32")
                assert "wort0 " not in shown  # the stale head is elided

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
                assert app._phase is Phase.DONE
                interim = app.query_one("#interim").render()
                interim = interim.plain if hasattr(interim, "plain") else str(interim)
                assert interim.strip() == ""
                assert " de " in f" {app.header_text()} "

        _run(body)


class TestQuitBinding:
    def test_ctrl_c_while_capturing_stops_but_does_not_exit(self):
        async def body():
            calls = []
            stopped = threading.Event()

            def stop():  # provider.stop stand-in; runs off the event loop now
                calls.append(1)
                stopped.set()

            app = _app(stop=stop)
            async with app.run_test() as pilot:
                await pilot.press("ctrl+c")
                assert stopped.wait(timeout=5)  # stop is dispatched to a worker thread
                await pilot.pause()
                assert calls == [1]  # crossed to the capture side
                assert app._phase is Phase.FINALIZING
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
                assert app._phase is Phase.DONE
                await pilot.press("q")
                await pilot.pause()
                assert not app.is_running

        _run(body)


class TestPersistAtFinalize:
    def test_finalized_persists_before_the_ui_swap(self):
        # The persist hook runs on the meeting thread at the finalized event —
        # even with no app mounted (the UI hop is dropped, the write is not),
        # so a force-quit "done" screen can never lose the transcript.
        calls = []
        view = TextualLiveView(
            MeetingProfile(local_speakers=1, remote_speakers=0), persist=calls.append
        )
        transcript = Transcript(language=None, profile=view.app._profile, entries=[])
        view.finalized(transcript)
        assert calls == [transcript]

    def test_persist_failure_is_surfaced_not_raised(self):
        # A failed write must not sink the meeting result: the view reports it
        # and the CLI retries after serve() returns.
        errors = []

        def boom(transcript):
            raise OSError("disk full")

        view = TextualLiveView(MeetingProfile(local_speakers=1, remote_speakers=0), persist=boom)
        view.error = errors.append  # capture the surfaced notice
        view.finalized(Transcript(language=None, profile=view.app._profile, entries=[]))
        assert errors and "disk full" in errors[0]


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
                    if view.app._phase is Phase.DONE:
                        break
                # The finalize swap ran; the app stays up showing the result.
                assert result["transcript"] is transcript
                assert view.app.committed_lines == ["Local-1  hallo welt"]
                assert view.app.is_running
                await pilot.press("q")  # user dismisses
                await pilot.pause()
                assert not view.app.is_running

        _run(body)

    def test_force_quit_during_finalize_still_captures_the_transcript(self):
        # 0b: a second Ctrl-C during the on-stop finalize force-exits the UI before
        # the meeting thread produced a transcript. serve() joins that thread before
        # reading the result, so the finalize is never lost to the early UI exit.
        async def body():
            view = TextualLiveView(
                MeetingProfile(local_speakers=1, remote_speakers=1), stop=lambda: None
            )
            transcript = Transcript(
                language=Language.GERMAN,
                profile=view.app._profile,
                entries=[TranscriptEntry("Local-1", "hallo", 0.0, 0.4)],
            )
            release = threading.Event()

            def meeting() -> Transcript:
                release.wait(timeout=5)  # stand-in for a slow on-stop finalize
                return transcript

            result: dict[str, object] = {}
            view._arm_meeting(meeting, result)  # the same wiring serve() uses
            async with view.app.run_test() as pilot:
                await pilot.press("ctrl+c")  # → finalizing
                await pilot.press("ctrl+c")  # impatient → force-exit while finalizing
                await pilot.pause()
                assert not view.app.is_running  # UI exited early...
                assert "transcript" not in result  # ...before the finalize produced one

            # serve()'s join-then-read: let the finalize complete and collect it.
            release.set()
            view._meeting_thread.join(timeout=5)
            assert result["transcript"] is transcript

        _run(body)


def app_has(view: TextualLiveView, line: str) -> bool:
    return line in view.app.committed_lines
