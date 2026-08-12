"""The shared helper transport (capture/helper.py), against the fake helper.

One :class:`HelperCaptureProvider` serves all three platforms; the only
per-platform variation is the stop gesture — SIGINT for the Swift helper,
stdin-EOF for the Rust one, picked by ``sys.platform``. Every transport test
runs against both gestures via explicit flavors, so the transport stays
covered on Windows too (where the SIGINT flavor cannot run: Windows has no
signal a parent can aim at one child). test_capture_windows.py keeps only
what is genuinely per-platform: the privacy consent store and the
"(loopback)" device suffix.
"""

import io
import os
import signal
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from stenograf.capture.base import CaptureHelperError, CaptureUnavailableError, Channel
from stenograf.capture.helper import (
    HelperCaptureProvider,
    HelperNotFoundError,
    find_helper,
    query_devices,
    read_frame,
)

FAKE = [sys.executable, str(Path(__file__).parent / "fake_stenocap.py")]
_HEADER = struct.Struct("<BdI")


class SigintTransport(HelperCaptureProvider):
    """The Swift helper's stop gesture (the macOS default)."""

    _stop_signal = signal.SIGINT


class StdinEofTransport(HelperCaptureProvider):
    """The Rust helper's stop gesture (the Linux/Windows default)."""

    _stop_signal = None


def test_stop_gesture_matches_the_platform_helper():
    # The Swift helper takes SIGINT; the Rust one stops on stdin EOF.
    expected = signal.SIGINT if sys.platform == "darwin" else None
    assert HelperCaptureProvider._stop_signal == expected


GESTURES = [
    pytest.param(
        SigintTransport,
        id="sigint",
        marks=pytest.mark.skipif(
            sys.platform == "win32",
            reason="the SIGINT stop gesture does not exist on Windows",
        ),
    ),
    pytest.param(StdinEofTransport, id="stdin-eof"),
]


@pytest.fixture(params=GESTURES)
def transport(request):
    """A provider factory for the current stop-gesture flavor.

    The stdin-EOF flavor gets ``--stop-on-stdin`` so the fake helper honors
    the gesture; the SIGINT flavor must not (under pytest the inherited stdin
    may already be closed, which would end the fake immediately)."""
    cls = request.param
    extra = () if cls._stop_signal is not None else ("--stop-on-stdin",)

    def make(*args: str, **kwargs):
        return cls(command=[*FAKE, *args, *extra], **kwargs)

    return make


def make_frame(code: int, timestamp: float, samples: list[int]) -> bytes:
    body = struct.pack(f"<{len(samples)}h", *samples)
    return _HEADER.pack(code, timestamp, len(samples)) + body


class TestReadFrame:
    def test_parses_a_frame(self):
        stream = io.BytesIO(make_frame(1, 1.5, [1, 2, 3]))
        frame = read_frame(stream)
        assert frame.channel is Channel.SYSTEM
        assert frame.timestamp == 1.5
        assert frame.samples.tolist() == [1, 2, 3]
        assert frame.samples.dtype == np.int16

    def test_returns_none_at_clean_eof(self):
        assert read_frame(io.BytesIO(b"")) is None

    def test_returns_none_on_truncated_payload(self):
        # Header promises 3 samples but only 1 is present → treated as EOF.
        truncated = _HEADER.pack(0, 0.0, 3) + struct.pack("<h", 9)
        assert read_frame(io.BytesIO(truncated)) is None

    def test_rejects_a_bad_channel_code(self):
        with pytest.raises(ValueError):
            read_frame(io.BytesIO(_HEADER.pack(7, 0.0, 0)))

    def test_rejects_an_absurd_sample_count(self):
        with pytest.raises(ValueError):
            read_frame(io.BytesIO(_HEADER.pack(0, 0.0, 10_000_000)))


class TestHelperTransport:
    def test_reads_both_channels_until_eof(self, transport):
        provider = transport()
        provider.start({Channel.MIC, Channel.SYSTEM})
        frames = list(provider.frames())
        provider.stop()

        assert {f.channel for f in frames} == {Channel.MIC, Channel.SYSTEM}
        mic = [f for f in frames if f.channel is Channel.MIC]
        assert len(mic) == 3
        assert [f.timestamp for f in mic] == [0.0, 0.1, 0.2]

    def test_only_requested_channel_is_started(self, transport):
        provider = transport()
        provider.start({Channel.MIC})  # in-room: no system tap
        channels = {f.channel for f in provider.frames()}
        provider.stop()
        assert channels == {Channel.MIC}

    def test_stop_terminates_a_running_helper(self, transport):
        provider = transport("--forever")
        provider.start({Channel.MIC})
        first = next(provider.frames())
        assert first.channel is Channel.MIC
        provider.stop()
        assert provider._proc is None  # torn down

    def test_a_stalled_consumer_never_blocks_the_helper(self, transport):
        # The regression behind two production bugs (ebf660a, 7dd1510): the
        # consumer stalls, the 64 KB pipe fills, the helper blocks in write()
        # and Core Audio kills the tap permanently. The drain thread must absorb
        # the stream regardless of the consumer, so a helper with far more
        # output than the pipe holds can exit before frames() is ever read.
        provider = transport("--frames", "200")
        provider.start({Channel.MIC})
        assert provider._proc.wait(timeout=10) == 0  # a blocked write would time out
        frames = list(provider.frames())
        provider.stop()
        assert len(frames) == 200  # buffered while the consumer stalled, none dropped

    def test_stream_desync_raises_in_the_consumer(self, transport):
        # The drain thread hits the malformed header; the error must surface in
        # frames(), not die silently on the drain thread.
        provider = transport("--malformed")
        provider.start({Channel.MIC})
        with pytest.raises(ValueError, match="malformed"):
            list(provider.frames())
        provider.stop()

    def test_on_log_keeps_helper_stderr_off_the_terminal(self, transport, capfd):
        # The GUI path: helper chatter must reach the sink line-by-line and
        # never the real stderr, which a GUI process has no terminal to show
        # (the "device format at start / stopped at Ctrl-C" bug).
        lines = []
        provider = transport("--chatter", on_log=lines.append)
        provider.start({Channel.MIC})
        list(provider.frames())
        provider.stop()  # joins the relay: the final lines are in by now
        assert "fake-stenocap: mic format: 48000.0 Hz, 1 ch" in lines
        assert "fake-stenocap: stopped" in lines
        assert capfd.readouterr().err == ""

    def test_helper_stderr_is_inherited_by_default(self, transport, capfd):
        # The plain CLI keeps today's behaviour: no sink, chatter lands on the
        # terminal's stderr where capture errors have always been visible.
        provider = transport("--chatter")
        provider.start({Channel.MIC})
        list(provider.frames())
        provider.stop()
        assert "fake-stenocap: mic format" in capfd.readouterr().err

    def test_a_raising_sink_does_not_break_capture(self, transport):
        def bad_sink(line: str) -> None:
            raise RuntimeError("sink is broken")

        provider = transport("--chatter", on_log=bad_sink)
        provider.start({Channel.MIC})
        frames = list(provider.frames())
        provider.stop()
        assert len(frames) == 3  # audio unaffected by the sink's failures

    def test_startup_crash_retries_once_then_raises_the_fatal_detail(self, transport):
        # The helper dying before any frame (coreaudiod wedged by a concurrent
        # capture app) used to look like a clean end-of-stream — the meeting
        # finalized an empty transcript that read as success. It must retry
        # once (the wedge is measurably transient), then raise with the
        # helper's own FATAL line so the failure toast says why.
        lines: list[str] = []
        provider = transport("--die-at-start", on_log=lines.append)
        provider.start({Channel.MIC})
        with pytest.raises(CaptureHelperError, match="tap unavailable"):
            list(provider.frames())
        provider.stop()
        assert any("retrying once" in ln for ln in lines)

    def test_startup_crash_recovers_when_the_respawn_succeeds(self, transport, tmp_path):
        # First spawn crashes (marker absent), the respawn streams normally:
        # the consumer sees the frames and no error — the transient-wedge case.
        marker = tmp_path / "died-once"
        lines: list[str] = []
        provider = transport("--die-once", str(marker), on_log=lines.append)
        provider.start({Channel.MIC})
        frames = list(provider.frames())
        provider.stop()
        assert len(frames) == 3
        assert any("retrying once" in ln for ln in lines)

    def test_midstream_crash_raises_without_a_retry(self, transport):
        # After audio has flowed a respawn can't help (the fresh helper would
        # restart the shared clock at t=0); the crash must surface so the
        # session finalizes what it has and reports the error.
        lines: list[str] = []
        provider = transport("--die-after", "2", on_log=lines.append)
        provider.start({Channel.MIC})
        received = []
        with pytest.raises(CaptureHelperError, match="died mid-meeting"):
            for frame in provider.frames():
                received.append(frame)
        provider.stop()
        assert len(received) == 2
        assert not any("retrying" in ln for ln in lines)

    def test_startup_crash_without_a_sink_still_raises(self, transport, capfd):
        # Plain-CLI mode (stderr inherited): no tail is collected, but the
        # crash must still raise — with the exit status, and the FATAL line
        # itself visible on the terminal's stderr as ever.
        provider = transport("--die-at-start")
        provider.start({Channel.MIC})
        with pytest.raises(CaptureHelperError, match="exited with status 1"):
            list(provider.frames())
        provider.stop()
        err = capfd.readouterr().err
        assert "FATAL" in err  # the helper's own line, inherited
        assert "retrying once" in err  # the provider's announcement

    def test_stop_does_not_close_the_pipe_under_a_paused_reader(self, transport):
        # stop() may fire (max_seconds, TUI quit) while the consumer sits between
        # yields; the read that resumes afterwards must end at clean EOF, not
        # raise "read of closed file" because stop() closed the pipe object.
        provider = transport("--forever")
        provider.start({Channel.MIC})
        frames = provider.frames()
        next(frames)  # reader is now paused mid-iteration
        provider.stop()
        for _ in frames:  # drain to EOF — must not raise
            pass


class TestFindHelper:
    def test_env_override_wins(self, monkeypatch, tmp_path):
        target = tmp_path / "stenocap"
        target.write_text("")
        monkeypatch.setenv("STENOGRAF_CAPTURE_HELPER", str(target))
        assert find_helper() == target

    def test_packaged_binary_regains_executable_bit(self, monkeypatch, tmp_path):
        # A wheel-installed binary that lost its executable bit (some install
        # paths drop it) must come back executable, not fail later with EACCES.
        import stenograf.capture.helper as helper

        packaged = tmp_path / "bin" / helper.HELPER_NAME
        packaged.parent.mkdir()
        packaged.write_bytes(b"\x00")
        packaged.chmod(0o644)
        monkeypatch.setattr(helper.resources, "files", lambda package: tmp_path)
        found = find_helper()
        assert found == packaged
        assert os.access(found, os.X_OK)

    def test_raises_when_absent(self, monkeypatch):
        # Point the package-resource and dev-tree lookups at nothing by faking
        # __file__ location is overkill; instead assert the error type when the
        # binary is genuinely missing is exercised via a fresh temp env.
        import stenograf.capture.helper as helper

        monkeypatch.setattr(helper, "HELPER_NAME", "stenocap-does-not-exist")
        with pytest.raises(HelperNotFoundError):
            find_helper()


class TestQueryDevices:
    """The shared, undecorated device preflight (the Linux path uses it as-is)."""

    def _helper(self, monkeypatch, *extra):
        import stenograf.capture.helper as helper

        argv = [*FAKE, *extra]
        monkeypatch.setattr(helper, "find_helper", lambda: argv[0])
        # find_helper returns one path; the fake needs its script argument too,
        # so the run itself is what gets patched for these tests.
        real_run = subprocess.run
        monkeypatch.setattr(
            helper.subprocess,
            "run",
            lambda command, **kw: real_run([*argv, *command[1:]], **kw),
        )

    def test_names_pass_through_undecorated(self, monkeypatch):
        self._helper(monkeypatch)
        devices = query_devices({Channel.MIC, Channel.SYSTEM})
        assert devices == {
            Channel.MIC: "Fake mic device",
            Channel.SYSTEM: "Fake system device",  # no "(loopback)" here
        }

    def test_mic_only_never_asks_for_a_sink(self, monkeypatch):
        # In-room mode: a box with no output device at all must still pass.
        self._helper(monkeypatch)
        assert query_devices({Channel.MIC}) == {Channel.MIC: "Fake mic device"}

    def test_a_hung_helper_is_a_capture_error(self, monkeypatch):
        # A sound server that accepts the connection and never answers leaves
        # the helper blocked in its connect loop; the query's own timeout must
        # surface as a capture error (the CLI preflight and doctor catch those),
        # not a TimeoutExpired traceback.
        import stenograf.capture.helper as helper

        monkeypatch.setattr(helper, "find_helper", lambda: "stenocap")

        def hang(command, **kwargs):
            raise subprocess.TimeoutExpired(command, 15)

        monkeypatch.setattr(helper.subprocess, "run", hang)
        with pytest.raises(CaptureUnavailableError, match="timed out"):
            query_devices({Channel.MIC})


class TestMicDevice:
    """Choosing which microphone the mic channel records from."""

    def _helper(self, monkeypatch, *extra):
        import stenograf.capture.helper as helper

        argv = [*FAKE, *extra]
        monkeypatch.setattr(helper, "find_helper", lambda: argv[0])
        real_run = subprocess.run
        monkeypatch.setattr(
            helper.subprocess,
            "run",
            lambda command, **kw: real_run([*argv, *command[1:]], **kw),
        )

    def test_lists_what_the_machine_can_record_from(self, monkeypatch):
        from stenograf.capture.helper import list_input_devices

        self._helper(monkeypatch)
        listed = list_input_devices()
        assert [d.id for d in listed] == ["fake-mic-builtin", "fake-mic-usb"]
        assert [d.is_default for d in listed] == [True, False]
        assert listed[1].name == "Fake USB Microphone"

    def test_the_preflight_names_the_pinned_device(self, monkeypatch):
        self._helper(monkeypatch)
        devices = query_devices({Channel.MIC}, mic_device="fake-mic-usb")
        assert devices == {Channel.MIC: "Fake USB Microphone"}

    def test_a_pin_that_names_nothing_fails_before_capture(self, monkeypatch):
        # The whole point of the feature: never quietly record the default
        # microphone when the user asked for a specific one.
        self._helper(monkeypatch)
        with pytest.raises(CaptureUnavailableError, match="not available"):
            query_devices({Channel.MIC}, mic_device="no-such-microphone")

    def test_a_name_resolves_the_way_a_hand_edited_file_writes_it(self, monkeypatch):
        # settings.toml holds names as often as ids, spaces and all.
        self._helper(monkeypatch)
        assert query_devices({Channel.MIC}, mic_device="  Fake USB Microphone ") == {
            Channel.MIC: "Fake USB Microphone"
        }

    def test_the_pin_reaches_the_helper_on_every_spawn(self, monkeypatch, tmp_path):
        # Including the respawn after a first-attempt crash: a retry that
        # dropped the flag would record the wrong microphone for the meeting.
        import stenograf.capture.helper as helper

        spawns = []
        real_popen = subprocess.Popen
        monkeypatch.setattr(
            helper.subprocess,
            "Popen",
            lambda argv, **kw: (spawns.append(argv), real_popen(argv, **kw))[1],
        )
        marker = tmp_path / "died-once"
        provider = StdinEofTransport(
            command=[*FAKE, "--stop-on-stdin", "--die-once", str(marker), "--frames", "2"],
            mic_device="fake-mic-usb",
        )
        provider.start({Channel.MIC})
        assert sum(1 for _ in provider.frames()) == 2  # the retry produced the frames
        provider.stop()

        assert len(spawns) == 2, "the first attempt crashed and was retried"
        for argv in spawns:
            assert argv[-2:] == ["--mic-device", "fake-mic-usb"]

    def test_an_unpinned_run_passes_no_device_flag(self, monkeypatch):
        import stenograf.capture.helper as helper

        spawns = []
        real_popen = subprocess.Popen
        monkeypatch.setattr(
            helper.subprocess,
            "Popen",
            lambda argv, **kw: (spawns.append(argv), real_popen(argv, **kw))[1],
        )
        provider = StdinEofTransport(command=[*FAKE, "--stop-on-stdin", "--frames", "1"])
        provider.start({Channel.MIC})
        list(provider.frames())
        provider.stop()
        assert "--mic-device" not in spawns[0]

    def test_matching_follows_the_helpers_rule(self):
        from stenograf.capture.helper import InputDevice, match_input_device

        # The device name is DECOMPOSED, the way macOS reports it, while the
        # settings file holds the precomposed form a keyboard produces. Without
        # the NFC step those are different strings and the selection misses.
        decomposed = "Bu\u0308rgel-Mikrofon"
        precomposed = "B\u00fcrgel-Mikrofon"
        devices = [
            InputDevice(id="usb-1", name="Yeti", is_default=False),
            InputDevice(id="usb-2", name="Yeti", is_default=False),
            InputDevice(id="Yeti", name=decomposed, is_default=True),
        ]
        assert decomposed != precomposed, "the fixture must really be decomposed"

        # An id wins over a name; surrounding space and normal form do not
        # matter; case does; and a name two devices share is refused with the
        # reason the helpers give, not reported as absent.
        assert match_input_device(devices, " Yeti ")[0].id == "Yeti"
        assert match_input_device(devices, precomposed)[0].id == "Yeti"
        assert match_input_device(devices, "yeti") == (None, "not connected")
        assert match_input_device(devices[:2], "Yeti")[1] == (
            "matches 2 connected devices (usb-1, usb-2)"
        )

    def test_a_helper_too_old_for_the_flag_says_so(self, monkeypatch):
        # The stale dev-checkout case: the binary exits 2 with usage and no
        # FATAL line. Pointing the user at their sound settings would be a wild
        # goose chase — the fix is to rebuild the helper.
        import stenograf.capture.helper as helper
        from stenograf.capture.helper import list_input_devices

        monkeypatch.setattr(helper, "find_helper", lambda: "stenocap")

        class Old:
            returncode = 2
            stdout = b""
            stderr = b"stenocap: unknown argument --list-inputs\n"

        monkeypatch.setattr(helper.subprocess, "run", lambda *a, **kw: Old())
        with pytest.raises(CaptureUnavailableError, match="older than this version"):
            list_input_devices()
