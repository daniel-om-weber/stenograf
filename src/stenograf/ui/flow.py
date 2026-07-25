"""Launcher meeting flow: run a :class:`~stenograf.flow.MeetingRun` behind a pushed screen.

The screen-side half of a meeting. Everything that *is* the meeting — folder
allocation, capture, the load order, persistence, the notes tail — lives in
:mod:`stenograf.flow`, shared with the Qt app; what is left here is the Textual
part: arm the run on a background thread, push the screen, and report the
outcome on whatever screen the user lands back on.

Ordering still matters: the run is armed *before* the push and starts once the
screen mounts, so the user watches the loading status in the header instead of a
frozen launcher, and the transcript is persisted at the ``finalized`` event
(``MeetingRun.persist``, handed to the view here) — so a force-quit on the
"done" screen, or even mid-finalize, never loses the meeting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from stenograf.flow import MeetingRun
from stenograf.transcript import Transcript
from stenograf.ui.meeting import TextualLiveView

if TYPE_CHECKING:
    from stenograf.flow import MeetingRequest
    from stenograf.ui.app import StenografApp


def start_meeting(app: StenografApp, request: MeetingRequest) -> TextualLiveView:
    """Push a meeting screen onto ``app`` and run the meeting behind it.

    Arms the meeting on a background thread (started once the screen mounts),
    pushes the screen, and installs a dismiss callback that reports the outcome
    on whatever screen the user lands back on. Returns the view (tests drive
    it; interactive callers ignore it).
    """
    meeting = MeetingRun(request)
    view = TextualLiveView(
        request.profile,
        language=request.profile.language,
        persist=meeting.persist,
        app=app,
    )

    result: dict[str, object] = {}
    view.arm_meeting(lambda: meeting.run(view), result)

    def finished(transcript: Transcript | None) -> None:
        """Back on the previous screen: say how the meeting ended."""
        out_dir = meeting.out_dir
        if isinstance(transcript, Transcript):
            app.notify(f"Files in {out_dir}", title="Meeting saved", timeout=10)
        elif "error" in result:
            app.notify(str(result["error"]), title="Meeting failed", severity="error", timeout=10)
        elif "transcript" in result:  # ended, but produced nothing to write
            app.notify(
                f"The meeting ended before a transcript was produced; any .partial "
                f"checkpoint is kept in {out_dir}",
                severity="warning",
                timeout=10,
            )
        else:  # force-quit while the finalize still runs on the meeting thread
            app.notify(
                f"Still finalizing in the background — files will appear in {out_dir}",
                severity="warning",
                timeout=10,
            )

    app.track_meeting(view)  # quitting the whole app must not kill this thread
    app.push_screen(view.screen, finished)
    return view
