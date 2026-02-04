"""CLI commands for soundpack manager."""

import sqlite3
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from soundpack import __version__
from soundpack.audio import analyze_sample, get_audio_info
from soundpack.config import Config
from soundpack.db import Database
from soundpack.exporter import export_pack, get_folder_for_tags
from soundpack.generator import parse_prompt_simple, parse_prompt_with_ai, select_samples_for_pack, generate_pack_name
from soundpack.tagger import suggest_tags, extract_tags_from_filename

console = Console()


# Global options for config and db paths
pass_config = click.make_pass_decorator(Config)


class Context:
    """CLI context holding config and database."""

    def __init__(self, config_path: Path | None = None, db_path: Path | None = None):
        self.config = Config(config_path)
        if db_path:
            self._db_path = db_path
        else:
            self._db_path = self.config.db_path
        self._db: Database | None = None

    @property
    def db(self) -> Database:
        """Get database connection, creating if needed."""
        if self._db is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._db = Database(self._db_path)
        return self._db

    def close(self):
        """Close database connection."""
        if self._db:
            self._db.close()


pass_context = click.make_pass_decorator(Context)


@click.group()
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    help="Path to config file",
)
@click.option(
    "--db",
    "db_path",
    type=click.Path(path_type=Path),
    help="Path to database file",
)
@click.version_option(version=__version__, prog_name="soundpack")
@click.pass_context
def cli(ctx, config_path: Path | None, db_path: Path | None):
    """Sound Pack Manager - Catalog, tag, and organize audio samples."""
    ctx.ensure_object(dict)
    ctx.obj = Context(config_path, db_path)


@cli.result_callback()
@pass_context
def cleanup(ctx, result, **kwargs):
    """Cleanup after command execution."""
    ctx.close()


# Default max file size for import (50 MB)
DEFAULT_MAX_FILE_SIZE_MB = 50


# Import command
@cli.command("import")
@click.argument("directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--recursive", "-r", is_flag=True, help="Scan subdirectories")
@click.option("--analyze", "-a", is_flag=True, help="Run audio analysis on import")
@click.option("--tag", "-t", is_flag=True, help="Run AI tagging after import")
@click.option(
    "--max-size",
    type=int,
    default=DEFAULT_MAX_FILE_SIZE_MB,
    help=f"Max file size in MB (default: {DEFAULT_MAX_FILE_SIZE_MB})",
)
@pass_context
def import_samples(
    ctx, directory: Path, recursive: bool, analyze: bool, tag: bool, max_size: int
):
    """Import samples from a directory into the library."""
    max_size_bytes = max_size * 1024 * 1024

    # Find all WAV files
    pattern = "**/*.wav" if recursive else "*.wav"
    wav_files = list(directory.glob(pattern))

    # Also check for .WAV (case insensitive)
    pattern_upper = "**/*.WAV" if recursive else "*.WAV"
    wav_files.extend(directory.glob(pattern_upper))

    if not wav_files:
        console.print(f"[yellow]No WAV files found in {directory}[/yellow]")
        return

    console.print(f"Found {len(wav_files)} WAV files")

    imported = 0
    skipped = 0
    too_large = 0

    for wav_path in wav_files:
        try:
            # Check file size first
            file_size = wav_path.stat().st_size
            if file_size > max_size_bytes:
                size_mb = file_size / (1024 * 1024)
                console.print(
                    f"  [yellow]Skipped {wav_path.name} ({size_mb:.1f} MB > {max_size} MB limit)[/yellow]"
                )
                too_large += 1
                continue

            # Check if already in database
            existing = ctx.db.get_sample_by_path(str(wav_path.absolute()))
            if existing:
                skipped += 1
                continue

            # Get basic info
            if analyze:
                info = analyze_sample(wav_path)
                console.print(f"  Analyzing {wav_path.name}...")
            else:
                info = get_audio_info(wav_path)

            # Add to database
            ctx.db.add_sample(
                file_path=str(wav_path.absolute()),
                filename=wav_path.name,
                duration_ms=info.get("duration_ms"),
                sample_rate=info.get("sample_rate"),
                channels=info.get("channels"),
                bit_depth=info.get("bit_depth"),
                bpm=info.get("bpm"),
                bpm_confidence=info.get("bpm_confidence"),
                detected_key=info.get("detected_key"),
                key_confidence=info.get("key_confidence"),
                spectral_centroid=info.get("spectral_centroid"),
                onset_strength=info.get("onset_strength"),
                rms_energy=info.get("rms_energy"),
                is_loop=info.get("is_loop", False),
            )
            imported += 1
            console.print(f"  [green]Imported {wav_path.name}[/green]")

        except Exception as e:
            console.print(f"  [red]Error importing {wav_path.name}: {e}[/red]")

    console.print(f"\n[bold]Imported {imported} samples[/bold] ({skipped} already in library)")
    if too_large:
        console.print(f"[yellow]Skipped {too_large} files exceeding {max_size} MB limit[/yellow]")


# List command
@cli.command("list")
@click.option("--tag", "-t", "tags", multiple=True, help="Filter by tag")
@click.option("--bpm", type=str, help="Filter by BPM range (e.g., 90-120)")
@click.option("--key", type=str, help="Filter by detected key")
@click.option("--untagged", is_flag=True, help="Show only untagged samples")
@pass_context
def list_samples(ctx, tags: tuple, bpm: str | None, key: str | None, untagged: bool):
    """List samples in the library."""
    # Parse BPM range
    bpm_min = bpm_max = None
    if bpm:
        if "-" in bpm:
            bpm_min, bpm_max = map(float, bpm.split("-"))
        else:
            bpm_min = bpm_max = float(bpm)

    samples = ctx.db.list_samples(
        bpm_min=bpm_min,
        bpm_max=bpm_max,
        key=key,
        untagged=untagged,
        tags=list(tags) if tags else None,
    )

    if not samples:
        console.print("[yellow]No samples found[/yellow]")
        return

    table = Table(title="Samples")
    table.add_column("ID", style="cyan")
    table.add_column("Filename")
    table.add_column("Duration")
    table.add_column("BPM")
    table.add_column("Key")

    for sample in samples:
        duration = f"{sample['duration_ms']}ms" if sample["duration_ms"] else "-"
        bpm_str = f"{sample['bpm']:.0f}" if sample["bpm"] else "-"
        key_str = sample["detected_key"] or "-"

        table.add_row(
            str(sample["id"]),
            sample["filename"],
            duration,
            bpm_str,
            key_str,
        )

    console.print(table)


# Info command
@cli.command("info")
@click.argument("sample_id_or_path")
@pass_context
def show_info(ctx, sample_id_or_path: str):
    """Show detailed info for a sample."""
    # Try to parse as ID first
    try:
        sample_id = int(sample_id_or_path)
        sample = ctx.db.get_sample(sample_id)
    except ValueError:
        # It's a path
        sample = ctx.db.get_sample_by_path(sample_id_or_path)

    if not sample:
        console.print(f"[red]Sample not found: {sample_id_or_path}[/red]")
        raise SystemExit(1)

    # Display sample info
    console.print(f"\n[bold]{sample['filename']}[/bold]")
    console.print(f"  Path: {sample['file_path']}")
    console.print(f"  Duration: {sample['duration_ms']}ms")
    console.print(f"  Sample Rate: {sample['sample_rate']} Hz")
    console.print(f"  Channels: {sample['channels']}")

    if sample["bpm"]:
        console.print(f"  BPM: {sample['bpm']:.1f} (confidence: {sample['bpm_confidence']:.2f})")
    if sample["detected_key"]:
        console.print(
            f"  Key: {sample['detected_key']} (confidence: {sample['key_confidence']:.2f})"
        )
    if sample["spectral_centroid"]:
        console.print(f"  Spectral Centroid: {sample['spectral_centroid']:.1f} Hz")

    # Show tags
    tags = ctx.db.get_sample_tags(sample["id"])
    if tags:
        tag_names = [t["name"] for t in tags]
        console.print(f"  Tags: {', '.join(tag_names)}")


# Analyze command
@cli.command("analyze")
@click.option("--all", "all_samples", is_flag=True, help="Analyze all samples")
@click.option("--missing", is_flag=True, help="Only analyze samples without analysis data")
@click.argument("sample_ids", nargs=-1, type=int)
@pass_context
def analyze_samples(ctx, all_samples: bool, missing: bool, sample_ids: tuple):
    """Analyze samples for BPM, key, and spectral features."""
    if all_samples:
        samples = ctx.db.list_samples()
    elif sample_ids:
        samples = [ctx.db.get_sample(sid) for sid in sample_ids]
        samples = [s for s in samples if s]  # Filter out None
    else:
        console.print("[yellow]Specify --all or sample IDs to analyze[/yellow]")
        return

    if missing:
        samples = [s for s in samples if s.get("bpm") is None]

    console.print(f"Analyzing {len(samples)} samples...")

    for sample in samples:
        try:
            console.print(f"  Analyzing {sample['filename']}...")
            analysis = analyze_sample(sample["file_path"])

            ctx.db.update_sample(
                sample["id"],
                bpm=analysis.get("bpm"),
                bpm_confidence=analysis.get("bpm_confidence"),
                detected_key=analysis.get("detected_key"),
                key_confidence=analysis.get("key_confidence"),
                spectral_centroid=analysis.get("spectral_centroid"),
                onset_strength=analysis.get("onset_strength"),
                rms_energy=analysis.get("rms_energy"),
                is_loop=analysis.get("is_loop", False),
            )
        except Exception as e:
            console.print(f"  [red]Error analyzing {sample['filename']}: {e}[/red]")

    console.print("[bold green]Analysis complete[/bold green]")


# Tag command
@cli.command("tag")
@click.argument("sample_id", type=int)
@click.argument("tag_names", nargs=-1)
@pass_context
def tag_sample(ctx, sample_id: int, tag_names: tuple):
    """Tag a sample with one or more tags."""
    sample = ctx.db.get_sample(sample_id)
    if not sample:
        console.print(f"[red]Sample not found: {sample_id}[/red]")
        raise SystemExit(1)

    for tag_name in tag_names:
        # Get or create tag
        tag = ctx.db.get_tag_by_name(tag_name)
        if not tag:
            # Create tag with default category
            tag_id = ctx.db.add_tag(tag_name, category="instrument")
            console.print(f"  [yellow]Created new tag: {tag_name}[/yellow]")
        else:
            tag_id = tag["id"]

        ctx.db.tag_sample(sample_id, tag_id)
        console.print(f"  Tagged with [cyan]{tag_name}[/cyan]")

    console.print(f"[green]Tagged sample {sample_id}[/green]")


# Tags command group
@cli.group("tags", invoke_without_command=True)
@click.option("--category", "-c", help="Filter by category")
@pass_context
@click.pass_context
def tags_group(click_ctx, ctx, category: str | None):
    """Manage tag vocabulary."""
    # If invoked without subcommand, list tags
    if click_ctx.invoked_subcommand is None:
        tags = ctx.db.list_tags(category=category)
        if not tags:
            console.print("[yellow]No tags found[/yellow]")
            return

        table = Table(title="Tags")
        table.add_column("Name", style="cyan")
        table.add_column("Category")

        for tag in tags:
            table.add_row(tag["name"], tag["category"])

        console.print(table)


@tags_group.command("list")
@click.option("--category", "-c", help="Filter by category")
@pass_context
def list_tags(ctx, category: str | None):
    """List all tags."""
    tags = ctx.db.list_tags(category=category)

    if not tags:
        console.print("[yellow]No tags found[/yellow]")
        return

    table = Table(title="Tags")
    table.add_column("Name", style="cyan")
    table.add_column("Category")

    for tag in tags:
        table.add_row(tag["name"], tag["category"])

    console.print(table)




@tags_group.command("add")
@click.argument("name")
@click.option("--category", "-c", required=True, help="Tag category")
@pass_context
def add_tag(ctx, name: str, category: str):
    """Add a new tag to the vocabulary."""
    try:
        ctx.db.add_tag(name, category)
        console.print(f"[green]Added tag: {name} ({category})[/green]")
    except sqlite3.IntegrityError:
        console.print(f"[yellow]Tag already exists: {name}[/yellow]")


# Config command group
@cli.group("config")
def config_group():
    """Manage configuration."""
    pass


@config_group.command("show")
@pass_context
def config_show(ctx):
    """Show current configuration."""
    import toml

    config = ctx.config.all()
    console.print(toml.dumps(config))


@config_group.command("set")
@click.argument("key")
@click.argument("value")
@pass_context
def config_set(ctx, key: str, value: str):
    """Set a configuration value (format: section.key value)."""
    if "." not in key:
        console.print("[red]Key must be in format: section.key[/red]")
        raise SystemExit(1)

    section, setting = key.split(".", 1)

    # Try to parse value as int/float/bool
    parsed_value: str | int | float | bool = value
    if value.lower() in ("true", "false"):
        parsed_value = value.lower() == "true"
    else:
        try:
            parsed_value = int(value)
        except ValueError:
            try:
                parsed_value = float(value)
            except ValueError:
                pass

    ctx.config.set(section, setting, parsed_value)
    ctx.config.save()
    console.print(f"[green]Set {key} = {parsed_value}[/green]")


# Remove command
@cli.command("remove")
@click.argument("sample_id_or_path")
@click.option("--delete-file", is_flag=True, help="Also delete the audio file")
@pass_context
def remove_sample(ctx, sample_id_or_path: str, delete_file: bool):
    """Remove a sample from the library."""
    # Try to parse as ID first
    try:
        sample_id = int(sample_id_or_path)
        sample = ctx.db.get_sample(sample_id)
    except ValueError:
        sample = ctx.db.get_sample_by_path(sample_id_or_path)
        sample_id = sample["id"] if sample else None

    if not sample:
        console.print(f"[red]Sample not found: {sample_id_or_path}[/red]")
        raise SystemExit(1)

    ctx.db.remove_sample(sample_id)
    console.print(f"[green]Removed {sample['filename']} from library[/green]")

    if delete_file:
        file_path = Path(sample["file_path"])
        if file_path.exists():
            file_path.unlink()
            console.print(f"[yellow]Deleted file: {file_path}[/yellow]")


# Search command
@cli.command("search")
@click.argument("query", required=False)
@click.option("--tag", "-t", "tags", multiple=True, help="Filter by tag")
@click.option("--bpm", type=str, help="Filter by BPM range (e.g., 90-120)")
@click.option("--key", type=str, help="Filter by detected key")
@pass_context
def search_samples(ctx, query: str | None, tags: tuple, bpm: str | None, key: str | None):
    """Search samples by natural language query or filters."""
    # Parse BPM range
    bpm_min = bpm_max = None
    if bpm:
        if "-" in bpm:
            bpm_min, bpm_max = map(float, bpm.split("-"))
        else:
            bpm_min = bpm_max = float(bpm)

    # Parse natural language query
    search_tags = list(tags) if tags else []
    if query:
        parsed = parse_prompt_simple(query)
        search_tags.extend(parsed.tags)
        if parsed.bpm_min and not bpm_min:
            bpm_min = parsed.bpm_min
            bpm_max = parsed.bpm_max
        if parsed.key and not key:
            key = parsed.key

    # Get all samples
    samples = ctx.db.list_samples(
        bpm_min=bpm_min,
        bpm_max=bpm_max,
        key=key,
        tags=search_tags if search_tags else None,
    )

    if not samples:
        console.print("[yellow]No samples found matching your search[/yellow]")
        return

    table = Table(title=f"Search Results ({len(samples)} samples)")
    table.add_column("ID", style="cyan")
    table.add_column("Filename")
    table.add_column("BPM")
    table.add_column("Key")
    table.add_column("Tags")

    for sample in samples[:50]:  # Limit display
        sample_tags = ctx.db.get_sample_tags(sample["id"])
        tag_str = ", ".join(t["name"] for t in sample_tags[:5])
        if len(sample_tags) > 5:
            tag_str += f" +{len(sample_tags) - 5}"

        bpm_str = f"{sample['bpm']:.0f}" if sample["bpm"] else "-"
        key_str = sample["detected_key"] or "-"

        table.add_row(
            str(sample["id"]),
            sample["filename"],
            bpm_str,
            key_str,
            tag_str,
        )

    console.print(table)
    if len(samples) > 50:
        console.print(f"[dim]Showing 50 of {len(samples)} results[/dim]")


# Generate command
@cli.command("generate")
@click.argument("prompt")
@click.option("--name", "-n", help="Pack name (auto-generated if omitted)")
@click.option("--max", "max_samples", type=int, default=64, help="Maximum samples")
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output directory")
@click.option("--dry-run", is_flag=True, help="Show what would be included without exporting")
@pass_context
def generate_pack(
    ctx,
    prompt: str,
    name: str | None,
    max_samples: int,
    output: Path | None,
    dry_run: bool,
):
    """Generate a pack from a natural language prompt."""
    # Parse prompt - use AI if API key is configured
    api_key = ctx.config.api_key
    if api_key:
        console.print("[dim]Using AI to interpret prompt...[/dim]")
        parsed = parse_prompt_with_ai(prompt, api_key=api_key)
    else:
        parsed = parse_prompt_simple(prompt)
    console.print(f"Parsed tags: [cyan]{', '.join(parsed.tags)}[/cyan]")

    if parsed.bpm_min:
        console.print(f"BPM range: {parsed.bpm_min:.0f}-{parsed.bpm_max:.0f}")
    if parsed.key:
        console.print(f"Key: {parsed.key}")

    # Get all samples and their tags
    samples = ctx.db.list_samples()
    tag_mapping = {}
    for sample in samples:
        sample_tags = ctx.db.get_sample_tags(sample["id"])
        tag_mapping[sample["id"]] = [t["name"] for t in sample_tags]

    # Select samples
    selected = select_samples_for_pack(samples, tag_mapping, parsed, max_samples=max_samples)

    if not selected:
        console.print("[yellow]No samples match your criteria[/yellow]")
        return

    console.print(f"\n[bold]Selected {len(selected)} samples[/bold]")

    # Show selected samples
    table = Table(title="Pack Contents")
    table.add_column("Folder", style="cyan")
    table.add_column("Filename")
    table.add_column("Tags")

    for sample in selected[:20]:
        sample_tags = tag_mapping.get(sample["id"], [])
        folder = get_folder_for_tags(sample_tags)
        table.add_row(folder, sample["filename"], ", ".join(sample_tags[:5]))

    console.print(table)
    if len(selected) > 20:
        console.print(f"[dim]... and {len(selected) - 20} more[/dim]")

    if dry_run:
        return

    # Generate pack name if not provided
    pack_name = name or generate_pack_name(parsed)

    # Determine output directory
    output_dir = output or (ctx.config.output_dir / pack_name)

    # Build tag mapping for selected samples only
    selected_tag_mapping = {s["id"]: tag_mapping.get(s["id"], []) for s in selected}

    # Export with tag mapping for subfolder organization
    result = export_pack(selected, output_dir, pack_name, tag_mapping=selected_tag_mapping)

    console.print(f"\n[bold green]Pack exported![/bold green]")
    console.print(f"  Location: {result['output_dir']}")
    console.print(f"  Samples: {result['exported_count']}")
    if result["skipped_count"]:
        console.print(f"  Skipped: {result['skipped_count']}")

    # Show warnings about percussion folders
    for warning in result.get("warnings", []):
        console.print(f"  [yellow]⚠ {warning}[/yellow]")


# Export command (export existing search as pack)
@cli.command("export")
@click.option("--tag", "-t", "tags", multiple=True, required=True, help="Filter by tag")
@click.option("--bpm", type=str, help="Filter by BPM range")
@click.option("--key", type=str, help="Filter by key")
@click.option("--name", "-n", required=True, help="Pack name")
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output directory")
@click.option("--max", "max_samples", type=int, default=255, help="Maximum samples")
@pass_context
def export_samples(
    ctx,
    tags: tuple,
    bpm: str | None,
    key: str | None,
    name: str,
    output: Path | None,
    max_samples: int,
):
    """Export filtered samples as a pack."""
    # Parse BPM
    bpm_min = bpm_max = None
    if bpm:
        if "-" in bpm:
            bpm_min, bpm_max = map(float, bpm.split("-"))
        else:
            bpm_min = bpm_max = float(bpm)

    # Get matching samples
    samples = ctx.db.list_samples(
        bpm_min=bpm_min,
        bpm_max=bpm_max,
        key=key,
        tags=list(tags),
    )

    if not samples:
        console.print("[yellow]No samples match your filters[/yellow]")
        return

    # Limit to max
    samples = samples[:max_samples]

    # Build tag mapping for subfolder organization
    tag_mapping = {}
    for sample in samples:
        sample_tags = ctx.db.get_sample_tags(sample["id"])
        tag_mapping[sample["id"]] = [t["name"] for t in sample_tags]

    # Determine output directory
    output_dir = output or (ctx.config.output_dir / name)

    # Export with tag mapping
    result = export_pack(samples, output_dir, name, tag_mapping=tag_mapping)

    console.print(f"[bold green]Pack exported![/bold green]")
    console.print(f"  Location: {result['output_dir']}")
    console.print(f"  Samples: {result['exported_count']}")

    # Show warnings about percussion folders
    for warning in result.get("warnings", []):
        console.print(f"  [yellow]⚠ {warning}[/yellow]")


# Auto-tag command
@cli.command("autotag")
@click.option("--all", "all_samples", is_flag=True, help="Tag all samples")
@click.option("--untagged", is_flag=True, help="Only tag samples without tags")
@click.option("--ai", is_flag=True, help="Use AI for tagging (requires API key)")
@click.option("--force", is_flag=True, help="Re-tag even if already AI-tagged")
@click.argument("sample_ids", nargs=-1, type=int)
@pass_context
def autotag_samples(
    ctx, all_samples: bool, untagged: bool, ai: bool, force: bool, sample_ids: tuple
):
    """Automatically tag samples based on filename and optionally AI."""
    if all_samples:
        samples = ctx.db.list_samples()
    elif sample_ids:
        samples = [ctx.db.get_sample(sid) for sid in sample_ids]
        samples = [s for s in samples if s]
    else:
        console.print("[yellow]Specify --all or sample IDs to tag[/yellow]")
        return

    if untagged:
        samples = [s for s in samples if not ctx.db.get_sample_tags(s["id"])]

    # Skip already AI-tagged samples unless --force is used
    skipped_ai = 0
    if ai and not force:
        filtered = []
        for s in samples:
            if ctx.db.has_ai_tags(s["id"]):
                skipped_ai += 1
            else:
                filtered.append(s)
        samples = filtered

    api_key = ctx.config.api_key if ai else None

    if skipped_ai > 0:
        console.print(f"[dim]Skipping {skipped_ai} already AI-tagged samples (use --force to re-tag)[/dim]")

    console.print(f"Auto-tagging {len(samples)} samples...")

    # Pattern cache for AI results - samples with similar names share tags
    # e.g., kick_01.wav through kick_50.wav will only need one API call
    pattern_cache: dict[str, list[str]] = {} if ai else None
    cache_hits = 0

    for sample in samples:
        try:
            # Get suggested tags (with caching for AI)
            tags = suggest_tags(sample, api_key=api_key, use_ai=ai, pattern_cache=pattern_cache)

            # Track cache usage
            if pattern_cache is not None:
                from soundpack.tagger import get_filename_pattern
                pattern = get_filename_pattern(sample.get("filename", ""))
                if pattern in pattern_cache and len(pattern_cache) > 0:
                    cache_hits += 1

            if not tags:
                continue

            # Add tags to sample
            for tag_name in tags:
                tag = ctx.db.get_tag_by_name(tag_name)
                if not tag:
                    # Determine category based on tag vocabulary
                    from soundpack.tagger import TAG_VOCABULARY

                    category = "instrument"  # default
                    for cat, cat_tags in TAG_VOCABULARY.items():
                        if tag_name in cat_tags:
                            category = cat
                            break
                    tag_id = ctx.db.add_tag(tag_name, category)
                else:
                    tag_id = tag["id"]

                ctx.db.tag_sample(
                    sample["id"],
                    tag_id,
                    confidence=0.8 if ai else 0.9,
                    source="ai" if ai else "filename",
                )

            console.print(f"  {sample['filename']}: {', '.join(tags)}")

        except Exception as e:
            console.print(f"  [red]Error tagging {sample['filename']}: {e}[/red]")

    # Show cache stats
    if pattern_cache is not None and cache_hits > 0:
        api_calls = len(samples) - cache_hits
        console.print(f"[dim]Pattern cache: {cache_hits} samples used cached tags, {api_calls} API calls made[/dim]")

    console.print("[bold green]Auto-tagging complete[/bold green]")


# Stats command
@cli.command("stats")
@pass_context
def show_stats(ctx):
    """Show library statistics and overview."""
    stats = ctx.db.get_stats()

    # Header
    console.print("\n[bold]Library Statistics[/bold]\n")

    # Main counts table
    table = Table(show_header=False, box=None)
    table.add_column("Label", style="dim")
    table.add_column("Value", style="bold")

    table.add_row("Total Samples", str(stats["total_samples"]))
    table.add_row("Analyzed", f"{stats['analyzed_samples']} ({_pct(stats['analyzed_samples'], stats['total_samples'])})")
    table.add_row("Tagged", f"{stats['tagged_samples']} ({_pct(stats['tagged_samples'], stats['total_samples'])})")
    table.add_row("AI Tagged", f"{stats['ai_tagged_samples']} ({_pct(stats['ai_tagged_samples'], stats['total_samples'])})")

    # Format duration
    total_seconds = stats["total_duration_ms"] // 1000
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        duration_str = f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        duration_str = f"{minutes}m {seconds}s"
    else:
        duration_str = f"{seconds}s"
    table.add_row("Total Duration", duration_str)

    table.add_row("Packs Created", str(stats["packs_created"]))

    console.print(table)

    # Top tags
    if stats["top_tags"] and any(t["count"] > 0 for t in stats["top_tags"]):
        console.print("\n[bold]Top Tags[/bold]")
        tag_table = Table()
        tag_table.add_column("Tag", style="cyan")
        tag_table.add_column("Category")
        tag_table.add_column("Samples", justify="right")

        for tag in stats["top_tags"]:
            if tag["count"] > 0:
                tag_table.add_row(tag["name"], tag["category"], str(tag["count"]))

        console.print(tag_table)


def _pct(part: int, total: int) -> str:
    """Calculate percentage string."""
    if total == 0:
        return "0%"
    return f"{100 * part // total}%"


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
