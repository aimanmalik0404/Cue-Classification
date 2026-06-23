"""utils/artifact_loader.py — Download & verify hot-reload artifacts.

Used by the cue-service hot-reload path (C5) to pull a candidate config /
model bundle from GCS (gs://), HTTPS, or a local path, verify it against an
optional manifest (sha256), and hand the local path back to the caller.

Self-contained — Cue-Classification is a separate repo and cannot import the
monorepo's shared/ helpers, so this duplicates the minimal logic it needs.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

logger = logging.getLogger("CueService.artifact_loader")

DEFAULT_DEST_DIR = "/app/heuristic_pipeline/hotload"


def _parse_gs_uri(uri: str) -> tuple[str, str]:
    """Split gs://bucket/path/to/object into (bucket, object_name)."""
    rest = uri[len("gs://"):]
    if "/" not in rest:
        raise ValueError(f"Malformed gs:// uri (no object path): {uri}")
    bucket, _, object_name = rest.partition("/")
    if not bucket or not object_name:
        raise ValueError(f"Malformed gs:// uri: {uri}")
    return bucket, object_name


def download_artifact_to_local(uri: str, dest_dir: str = DEFAULT_DEST_DIR) -> str:
    """Download ``uri`` into ``dest_dir`` and return the local file path.

    Supports:
      - gs://bucket/object   → google.cloud.storage
      - https:// / http://   → urllib
      - local path           → passthrough (returned unchanged if it exists)
    """
    if not uri:
        raise ValueError("download_artifact_to_local: empty uri")

    # Local passthrough — already on disk, nothing to download.
    if not (uri.startswith("gs://") or uri.startswith("http://") or uri.startswith("https://")):
        local = Path(uri)
        if not local.exists():
            raise FileNotFoundError(f"Local artifact not found: {uri}")
        logger.info(f"Artifact is a local path, using directly: {uri}")
        return str(local)

    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    filename = uri.rstrip("/").split("/")[-1] or "artifact"
    dest_path = str(Path(dest_dir) / filename)

    if uri.startswith("gs://"):
        from google.cloud import storage  # imported lazily — optional dependency

        bucket_name, object_name = _parse_gs_uri(uri)
        logger.info(f"Downloading gs://{bucket_name}/{object_name} → {dest_path}")
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(object_name)
        blob.download_to_filename(dest_path)
    else:
        logger.info(f"Downloading {uri} → {dest_path}")
        req = urllib.request.Request(uri)
        with urllib.request.urlopen(req, timeout=30) as resp, open(dest_path, "wb") as out:
            out.write(resp.read())

    logger.info(f"Downloaded artifact to {dest_path}")
    return dest_path


def _sha256_of_file(local_path: str) -> str:
    h = hashlib.sha256()
    with open(local_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_artifact(local_path: str, manifest: Optional[dict]) -> bool:
    """Verify a downloaded artifact against an optional manifest.

    If the manifest carries a ``sha256`` digest, assert it matches the file's
    digest (raises ValueError on mismatch). When no manifest / no sha256 is
    present, verification is a no-op and returns True.
    """
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Artifact to verify not found: {local_path}")
    if not manifest:
        return True
    expected = manifest.get("sha256")
    if not expected:
        return True
    actual = _sha256_of_file(local_path)
    if actual.lower() != str(expected).lower():
        raise ValueError(
            f"Artifact sha256 mismatch for {local_path}: "
            f"expected {expected}, got {actual}"
        )
    logger.info(f"Artifact sha256 verified for {local_path}")
    return True


def download_manifest(url: Optional[str]) -> Optional[dict]:
    """Download and parse a manifest JSON from gs:// / https:// / local path.

    Returns the parsed dict, or None when ``url`` is falsy or the download /
    parse fails (manifest is optional — callers degrade gracefully).
    """
    if not url:
        return None
    try:
        local = download_artifact_to_local(url)
        with open(local, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not download/parse manifest {url}: {e}")
        return None
