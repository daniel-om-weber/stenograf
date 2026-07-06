"""Backend-level guards for the parakeet ASR backend.

The load-thread materialization is a regression guard for a bug that only
reproduces on real hardware (MLX GPU + a worker thread), so it is asserted here
at the mechanism level rather than by running the real model.
"""

import mlx.core as mx
import parakeet_mlx

from stenograf.asr.parakeet import ParakeetMLXBackend


class _FakeModel:
    def __init__(self) -> None:
        self._params = {"weight": object()}

    def parameters(self):
        return self._params


def test_load_materializes_weights_for_cross_thread_use(monkeypatch):
    # The live LiveWorker decodes on a *different* thread than load() ran on.
    # MLX is lazy and its GPU streams are thread-local, so weights left lazy stay
    # bound to the load thread's Stream(gpu, 0) and the worker's first decode dies
    # with "There is no Stream(gpu, 0) in current thread". load() must force the
    # weights concrete on the load thread; if this eval is dropped, the live pass
    # breaks on real hardware while every mocked test stays green — hence this.
    fake = _FakeModel()
    monkeypatch.setattr(parakeet_mlx, "from_pretrained", lambda model_id: fake)
    evaled: list[tuple] = []
    monkeypatch.setattr(mx, "eval", lambda *args: evaled.append(args))

    backend = ParakeetMLXBackend()
    backend.load()

    assert backend._model is fake
    assert evaled == [(fake.parameters(),)]  # weights materialized on the load thread
