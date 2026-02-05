"""FastAPI server for spectral map visualization."""

import json
import logging
import sys
from pathlib import Path
from typing import Any

# IMMEDIATE print at module load time
print("[SERVER.PY] Module loading...", file=sys.stderr, flush=True)

from fastapi import FastAPI, HTTPException, Query

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("[SERVER.PY] Module loaded successfully", file=sys.stderr, flush=True)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from soundpack.db import Database
from soundpack.config import Config

# Will be set when server starts
db: Database | None = None
config: Config | None = None

app = FastAPI(title="Soundpack Spectral Map", version="0.1.0")

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


def get_db() -> Database:
    """Get database instance."""
    if db is None:
        raise HTTPException(status_code=500, detail="Database not initialized")
    return db


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main visualization page."""
    html_path = Path(__file__).parent / "static" / "index.html"
    # Force print to stderr with flush (bypasses buffering)
    print(f"[INDEX] __file__ = {__file__}", file=sys.stderr, flush=True)
    print(f"[INDEX] html_path = {html_path}", file=sys.stderr, flush=True)
    print(f"[INDEX] html_path.exists() = {html_path.exists()}", file=sys.stderr, flush=True)
    if html_path.exists():
        content = html_path.read_text()
        print(f"[INDEX] HTML length = {len(content)} chars", file=sys.stderr, flush=True)
        # Check if our debug log is in the file
        has_debug = '=== Spectral Map JS Loading ===' in content
        print(f"[INDEX] Contains debug log: {has_debug}", file=sys.stderr, flush=True)
        return HTMLResponse(content=content)
    print("[INDEX] ERROR: Static files not found!", file=sys.stderr, flush=True)
    return HTMLResponse(content="<h1>Spectral Map</h1><p>Static files not found</p>")


@app.get("/api/map")
async def get_map_data(
    min_x: float = Query(0.0, ge=0.0, le=1.0),
    max_x: float = Query(1.0, ge=0.0, le=1.0),
    min_y: float = Query(0.0, ge=0.0, le=1.0),
    max_y: float = Query(1.0, ge=0.0, le=1.0),
    tags: str | None = None,
) -> dict[str, Any]:
    """Get map data for visualization.

    Returns samples with their 2D positions and metadata.
    Supports filtering by bounding box and tags.
    """
    logger.info("GET /api/map called")
    database = get_db()
    samples = database.get_samples_with_map_data()
    logger.info(f"Found {len(samples)} samples with map data")

    # Filter by bounding box
    filtered = []
    for sample in samples:
        x, y = sample["map_x"], sample["map_y"]
        if min_x <= x <= max_x and min_y <= y <= max_y:
            # Get tags for this sample
            sample_tags = database.get_sample_tags(sample["id"])
            tag_names = [t["name"] for t in sample_tags]

            # Filter by tags if specified
            if tags:
                required_tags = [t.strip() for t in tags.split(",")]
                if not any(t in tag_names for t in required_tags):
                    continue

            filtered.append({
                "id": sample["id"],
                "filename": sample["filename"],
                "x": x,
                "y": y,
                "tags": tag_names,
                "bpm": sample.get("bpm"),
                "key": sample.get("detected_key"),
                "duration_ms": sample.get("duration_ms"),
                "spectral_centroid": sample.get("spectral_centroid"),
                "rms_energy": sample.get("rms_energy"),
            })

    return {
        "samples": filtered,
        "total": len(filtered),
        "bounds": {"min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y},
    }


@app.get("/api/sample/{sample_id}")
async def get_sample(sample_id: int) -> dict[str, Any]:
    """Get detailed info for a single sample."""
    database = get_db()
    sample = database.get_sample(sample_id)

    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")

    tags = database.get_sample_tags(sample_id)

    return {
        "id": sample["id"],
        "filename": sample["filename"],
        "file_path": sample["file_path"],
        "duration_ms": sample.get("duration_ms"),
        "sample_rate": sample.get("sample_rate"),
        "channels": sample.get("channels"),
        "bpm": sample.get("bpm"),
        "bpm_confidence": sample.get("bpm_confidence"),
        "detected_key": sample.get("detected_key"),
        "key_confidence": sample.get("key_confidence"),
        "spectral_centroid": sample.get("spectral_centroid"),
        "onset_strength": sample.get("onset_strength"),
        "rms_energy": sample.get("rms_energy"),
        "map_x": sample.get("map_x"),
        "map_y": sample.get("map_y"),
        "tags": [{"name": t["name"], "category": t["category"]} for t in tags],
    }


@app.get("/api/audio/{sample_id}")
async def get_audio(sample_id: int):
    """Stream audio file for a sample."""
    database = get_db()
    sample = database.get_sample(sample_id)

    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")

    file_path = Path(sample["file_path"])
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(
        file_path,
        media_type="audio/wav",
        filename=sample["filename"],
    )


@app.get("/api/neighbors/{sample_id}")
async def get_neighbors(sample_id: int, limit: int = Query(10, ge=1, le=50)) -> dict[str, Any]:
    """Get nearest neighbors for a sample."""
    import numpy as np
    from soundpack.map import find_neighbors

    database = get_db()
    sample = database.get_sample(sample_id)

    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")

    if sample.get("map_x") is None:
        raise HTTPException(status_code=400, detail="Sample has no map position")

    all_samples = database.get_samples_with_map_data()
    positions = np.array([[s["map_x"], s["map_y"]] for s in all_samples])

    sample_idx = next(
        (i for i, s in enumerate(all_samples) if s["id"] == sample_id), None
    )

    if sample_idx is None:
        raise HTTPException(status_code=400, detail="Sample not in map data")

    neighbors = find_neighbors(sample_idx, positions, k=limit)

    result = []
    for idx, distance in neighbors:
        neighbor = all_samples[idx]
        tags = database.get_sample_tags(neighbor["id"])
        result.append({
            "id": neighbor["id"],
            "filename": neighbor["filename"],
            "distance": distance,
            "x": neighbor["map_x"],
            "y": neighbor["map_y"],
            "tags": [t["name"] for t in tags],
        })

    return {"sample_id": sample_id, "neighbors": result}


@app.get("/api/tags")
async def get_tags() -> dict[str, Any]:
    """Get all tags grouped by category."""
    database = get_db()
    tags = database.list_tags()

    by_category: dict[str, list[str]] = {}
    for tag in tags:
        category = tag["category"]
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(tag["name"])

    return {"tags": by_category}


@app.get("/api/stats")
async def get_stats() -> dict[str, Any]:
    """Get library and map statistics."""
    database = get_db()
    lib_stats = database.get_stats()
    map_stats = database.get_map_stats()

    return {
        "library": {
            "total_samples": lib_stats["total_samples"],
            "analyzed_samples": lib_stats["analyzed_samples"],
            "tagged_samples": lib_stats["tagged_samples"],
        },
        "map": {
            "samples_with_features": map_stats["samples_with_features"],
            "samples_with_positions": map_stats["samples_with_positions"],
        },
    }


@app.post("/api/suggest")
async def suggest_samples(
    pack_ids: list[int],
    categories: list[str] | None = None,
    limit: int = Query(10, ge=1, le=50),
) -> dict[str, Any]:
    """Suggest samples that complement the current pack selection.

    Args:
        pack_ids: List of sample IDs currently in the pack.
        categories: Optional list of categories to focus on (e.g., ["snare", "hihat"]).
        limit: Maximum number of suggestions to return.

    Returns:
        Dict with suggestions and pack analysis.
    """
    from soundpack.map import suggest_complements, analyze_pack_balance

    database = get_db()

    # Load pack samples with their tags
    pack_samples = []
    for sample_id in pack_ids:
        sample = database.get_sample(sample_id)
        if sample:
            tags = database.get_sample_tags(sample_id)
            sample_dict = dict(sample)
            sample_dict["tags"] = [t["name"] for t in tags]
            pack_samples.append(sample_dict)

    if not pack_samples:
        return {
            "suggestions": [],
            "analysis": {"empty": True},
            "message": "No valid samples in pack",
        }

    # Load all samples with tags
    all_samples = database.get_samples_with_map_data()
    for sample in all_samples:
        tags = database.get_sample_tags(sample["id"])
        sample["tags"] = [t["name"] for t in tags]

    # Get suggestions
    suggestions = suggest_complements(
        pack_samples,
        all_samples,
        target_categories=categories,
        limit=limit,
    )

    # Format suggestions for response
    formatted_suggestions = []
    for s in suggestions:
        formatted_suggestions.append({
            "id": s["id"],
            "filename": s["filename"],
            "x": s["map_x"],
            "y": s["map_y"],
            "tags": s.get("tags", []),
            "bpm": s.get("bpm"),
            "score": s["suggestion_score"],
            "reason": s["suggestion_reason"],
        })

    # Analyze pack balance
    analysis = analyze_pack_balance(pack_samples)

    return {
        "suggestions": formatted_suggestions,
        "analysis": analysis,
    }


@app.post("/api/pack/analyze")
async def analyze_pack(pack_ids: list[int]) -> dict[str, Any]:
    """Analyze the balance and composition of a pack.

    Args:
        pack_ids: List of sample IDs in the pack.

    Returns:
        Dict with pack analysis including category distribution,
        spectral balance, and suggestions.
    """
    from soundpack.map import analyze_pack_balance

    database = get_db()

    # Load pack samples with their tags
    pack_samples = []
    for sample_id in pack_ids:
        sample = database.get_sample(sample_id)
        if sample:
            tags = database.get_sample_tags(sample_id)
            sample_dict = dict(sample)
            sample_dict["tags"] = [t["name"] for t in tags]
            pack_samples.append(sample_dict)

    if not pack_samples:
        return {"empty": True, "message": "No valid samples in pack"}

    analysis = analyze_pack_balance(pack_samples)

    return analysis


def create_app(db_path: Path | None = None, config_path: Path | None = None) -> FastAPI:
    """Create and configure the FastAPI app.

    Args:
        db_path: Path to database file.
        config_path: Path to config file.

    Returns:
        Configured FastAPI app.
    """
    global db, config

    config = Config(config_path)
    db_file = db_path or config.db_path
    db = Database(db_file)

    return app


def run_server(
    host: str = "127.0.0.1",
    port: int = 8080,
    db_path: Path | None = None,
    config_path: Path | None = None,
    open_browser: bool = True,
) -> None:
    """Run the web server.

    Args:
        host: Host to bind to.
        port: Port to bind to.
        db_path: Path to database file.
        config_path: Path to config file.
        open_browser: Whether to open browser automatically.
    """
    import uvicorn
    import webbrowser
    import threading

    create_app(db_path, config_path)

    if open_browser:
        def open_browser_delayed():
            import time
            time.sleep(1)
            webbrowser.open(f"http://{host}:{port}")

        threading.Thread(target=open_browser_delayed, daemon=True).start()

    uvicorn.run(app, host=host, port=port, log_level="info")
