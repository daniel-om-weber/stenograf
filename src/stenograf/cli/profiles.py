"""``steno profiles`` — manage saved speaker voiceprints."""

from __future__ import annotations

from pathlib import Path

import click

from stenograf import loaders


@click.group()
def profiles() -> None:
    """Manage saved speaker voiceprints for cross-meeting re-identification.

    Enroll a voice once and every later meeting relabels that speaker
    automatically (``steno start``/``transcribe`` unless ``--no-reid``).
    """


@profiles.command("list")
def profiles_list() -> None:
    """List enrolled speaker profiles."""
    from stenograf import models
    from stenograf.profiles import ProfileStore, default_store_path

    store = ProfileStore.load()
    all_profiles = store.profiles()
    if not all_profiles:
        click.echo(f"no speaker profiles yet ({default_store_path()})")
        click.echo("enroll one with: steno profiles enroll NAME sample.wav")
        return
    active_model = models.SPEAKER_EMBEDDING.name
    click.echo(f"speaker profiles ({default_store_path()}):")
    for p in sorted(all_profiles, key=lambda p: (p.embedding_model, p.name.lower())):
        noun = "sample" if p.samples == 1 else "samples"
        # A profile made under a different embedding model can never match a
        # cluster from the current one — flag it so the count is not misleading.
        tag = "" if p.embedding_model == active_model else "  [inactive: other embedding model]"
        click.echo(f"  {p.name}  ({p.samples} {noun}){tag}")


@profiles.command("enroll")
@click.argument("name")
@click.argument("audio_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--speakers",
    type=click.IntRange(1, 16),
    default=1,
    show_default=True,
    help="How many speakers are in the clip; it is diarized into this many and one "
    "cluster is enrolled.",
)
@click.option(
    "--speaker",
    "cluster",
    default=None,
    metavar="S<n>",
    help="Which diarized cluster to enroll when the clip has several speakers "
    "(re-run without it to see the choices). Ignored for a single-speaker clip.",
)
@click.option(
    "--reinforce",
    is_flag=True,
    help="Fold this sample into an existing profile's voiceprint instead of creating a new one.",
)
def profiles_enroll(
    name: str, audio_file: Path, speakers: int, cluster: str | None, reinforce: bool
) -> None:
    """Enroll speaker NAME from a voice sample in AUDIO_FILE.

    Give a short clip in which NAME is the only speaker (the default), or a
    multi-speaker recording (e.g. a meeting saved with ``--record-audio``) plus
    ``--speakers N`` and ``--speaker S<n>`` to enroll one person from it. The
    voiceprint is computed exactly the way meetings embed their clusters, so
    future meetings relabel this speaker automatically.
    """
    from stenograf import models
    from stenograf.audio import load_audio
    from stenograf.profiles import ProfileStore

    samples = load_audio(audio_file)
    diarizer = loaders.load_diarizer()
    result = diarizer.diarize_with_embeddings(samples, num_speakers=speakers)
    if not result.embeddings:
        raise click.ClickException(
            f"no embeddable speech found in {audio_file.name}; is it silent or too short?"
        )
    embedding = _choose_cluster(result.embeddings, result.turns, cluster)

    model = models.SPEAKER_EMBEDDING.name
    store = ProfileStore.load()
    existing = store.get(name, model)
    if reinforce:
        if existing is None:
            raise click.ClickException(
                f"no profile named {name!r} to reinforce; drop --reinforce to create it."
            )
        updated = store.reinforce(existing, embedding)
        store.save()
        click.echo(f"reinforced {name!r} ({updated.samples} samples)")
        return
    if existing is not None:
        raise click.ClickException(
            f"a profile named {name!r} already exists; use --reinforce to add this sample "
            "to it, or remove it first with `steno profiles remove`."
        )
    store.enroll(name, embedding, model)
    store.save()
    click.echo(f"enrolled {name!r} from {audio_file.name}")


def _choose_cluster(embeddings, turns, cluster: str | None):
    """Pick one cluster's embedding, or raise a helpful error when it is ambiguous."""
    if cluster is not None:
        if cluster not in embeddings:
            available = ", ".join(sorted(embeddings)) or "none"
            raise click.ClickException(
                f"no cluster {cluster!r} in the clip; available: {available}"
            )
        return embeddings[cluster]
    if len(embeddings) == 1:
        return next(iter(embeddings.values()))
    durations: dict[str, float] = {}
    for turn in turns:
        durations[turn.speaker] = durations.get(turn.speaker, 0.0) + (turn.end - turn.start)
    listing = "\n".join(f"  {c}  ({durations.get(c, 0.0):.1f}s speech)" for c in sorted(embeddings))
    raise click.ClickException(
        "the clip has several speakers; re-run with --speaker to pick one:\n" + listing
    )


@profiles.command("rename")
@click.argument("old")
@click.argument("new")
def profiles_rename(old: str, new: str) -> None:
    """Rename speaker profile OLD to NEW."""
    from stenograf import models
    from stenograf.profiles import ProfileStore

    store = ProfileStore.load()
    profile = store.get(old, models.SPEAKER_EMBEDDING.name)
    if profile is None:
        raise click.ClickException(f"no profile named {old!r}")
    try:
        store.rename(profile, new)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    store.save()
    click.echo(f"renamed {old!r} → {new!r}")


@profiles.command("remove")
@click.argument("name")
@click.option("--yes", is_flag=True, help="Skip the confirmation prompt.")
def profiles_remove(name: str, yes: bool) -> None:
    """Delete speaker profile NAME."""
    from stenograf import models
    from stenograf.profiles import ProfileStore

    store = ProfileStore.load()
    profile = store.get(name, models.SPEAKER_EMBEDDING.name)
    if profile is None:
        raise click.ClickException(f"no profile named {name!r}")
    if not yes:
        click.confirm(f"delete speaker profile {name!r}?", abort=True)
    store.remove(profile)
    store.save()
    click.echo(f"removed {name!r}")
