"""
00_datasets_downloader.py – Download and prepare th1nhng0/vietnamese-legal-documents
using the NeMo Curator DocumentDownloadExtractStage 4-step pipeline pattern.

Pipeline steps
--------------
1. URLGenerator      – lists all content-config parquet shard URLs from HuggingFace
2. DocumentDownloader – fetches each shard via huggingface_hub (resume-safe)
3. DocumentIterator   – reads rows from one content-config parquet shard
4. DocumentExtractor  – joins pre-loaded metadata → final record with "text" field

Output
------
<datasets_dir>/vietnamese-legal-documents/
    raw/
        metadata/   downloaded metadata parquet shards (cached)
        content/    downloaded content parquet shards (cached)
    data/
        part-*.parquet   NeMo-ready parquet files with columns:
                         text, id, document_number, title, url,
                         legal_type, legal_sectors, issuing_authority,
                         issuance_date, signers

Usage
-----
    python 00_datasets_downloader.py
    python 00_datasets_downloader.py --datasets_dir /my/data --url_limit 5
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download
from omegaconf import DictConfig, OmegaConf

from nemo_curator.pipeline import Pipeline
# Import directly from base submodules to avoid the top-level __init__.py,
# which transitively imports CommonCrawlDownloadExtractStage → warcio (optional dep).
from nemo_curator.stages.text.download.base.download import DocumentDownloader
from nemo_curator.stages.text.download.base.extract import DocumentExtractor
from nemo_curator.stages.text.download.base.iterator import DocumentIterator
from nemo_curator.stages.text.download.base.stage import DocumentDownloadExtractStage
from nemo_curator.stages.text.download.base.url_generation import URLGenerator
from nemo_curator.stages.text.io.writer import ParquetWriter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("viet-legal-downloader")

# ─── Config ───────────────────────────────────────────────────────────────────
_CFG: DictConfig = OmegaConf.load(Path(__file__).parent / "config.yaml")
_DS_CFG = _CFG.datasets["vietnamese-legal-documents"]

HF_REPO_ID: str      = _DS_CFG.hf_name
JOIN_KEY: str        = _DS_CFG.join_key        # "id"
TEXT_FIELD: str      = _DS_CFG.text_field      # "content" → renamed to "text"
METADATA_FIELDS: list[str] = OmegaConf.to_container(_DS_CFG.metadata_fields, resolve=True)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – URLGenerator
# ─────────────────────────────────────────────────────────────────────────────
class VietLegalURLGenerator(URLGenerator):
    """List all content-config parquet shard URLs from the HuggingFace repo."""

    def __init__(self, repo_id: str = HF_REPO_ID) -> None:
        self.repo_id = repo_id

    def generate_urls(self) -> list[str]:
        fs = HfFileSystem()
        paths = fs.glob(f"datasets/{self.repo_id}/content/**/*.parquet")
        if not paths:
            raise RuntimeError(
                f"No parquet files found under 'content' config in {self.repo_id}."
            )
        # FS path: datasets/{repo_id}/{rel}
        # URL:     https://huggingface.co/datasets/{repo_id}/resolve/main/{rel}
        urls = [
            f"https://huggingface.co/datasets/{self.repo_id}/resolve/main/"
            + p[len(f"datasets/{self.repo_id}/"):]
            for p in paths
        ]
        logger.info("Found %d content shards in %s", len(urls), self.repo_id)
        return urls


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – DocumentDownloader
# ─────────────────────────────────────────────────────────────────────────────
class VietLegalDownloader(DocumentDownloader):
    """Fetch a parquet shard from HuggingFace Hub to local storage.

    We override the public `download()` method rather than `_download_to_path`
    because the base-class template writes to a `.tmp` file then renames,
    but `hf_hub_download` is already atomic.  Overriding `download()` also
    avoids relying on `self.download_dir`, which does not survive Ray actor
    serialisation.  The two abstract methods are implemented as required stubs.
    """

    def __init__(self, download_dir: str, repo_id: str = HF_REPO_ID) -> None:
        super().__init__(download_dir=download_dir, verbose=True)
        self._local_dir = download_dir  # survives Ray serialisation
        self.repo_id = repo_id

    # Required abstract stubs – not called because download() is overridden.
    def _get_output_filename(self, url: str) -> str:
        marker = "/resolve/main/"
        return url.split(marker, 1)[1] if marker in url else url.split("/")[-1]

    def _download_to_path(self, url: str, path: str) -> tuple[bool, str | None]:
        return False, "unreachable: download() is overridden"

    def download(self, url: str) -> str | None:
        rel_path = self._get_output_filename(url)
        output_file = Path(self._local_dir) / rel_path

        if output_file.exists():
            logger.debug("Already cached: %s", output_file)
            return str(output_file)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            fetched = hf_hub_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                filename=rel_path,
                local_dir=self._local_dir,
            )
            logger.info("Downloaded → %s", fetched)
            return fetched
        except Exception as exc:
            logger.error("Failed to download %s: %s", url, exc)
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – DocumentIterator
# ─────────────────────────────────────────────────────────────────────────────
class VietLegalIterator(DocumentIterator):
    """Stream rows from one content-config parquet shard."""

    def iterate(self, file_path: str) -> Iterator[dict[str, Any]]:
        table = pq.read_table(file_path, columns=[JOIN_KEY, TEXT_FIELD])
        for batch in table.to_batches(max_chunksize=1000):
            df = batch.to_pandas()
            for row in df.itertuples(index=False):
                yield {JOIN_KEY: getattr(row, JOIN_KEY), TEXT_FIELD: getattr(row, TEXT_FIELD)}

    def output_columns(self) -> list[str]:
        return [JOIN_KEY, TEXT_FIELD]


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 – DocumentExtractor
# ─────────────────────────────────────────────────────────────────────────────
class VietLegalExtractor(DocumentExtractor):
    """Join pre-loaded metadata into each content record; rename field → "text"."""

    def __init__(self, metadata_lookup: dict[Any, dict]) -> None:
        self._lookup = metadata_lookup

    def extract(self, record: dict[str, str]) -> dict[str, Any] | None:
        doc_id = record.get(JOIN_KEY)
        content = record.get(TEXT_FIELD, "")
        if doc_id is None or not content or not content.strip():
            return None

        meta = self._lookup.get(doc_id, {})
        return {
            "text": content.strip(),
            JOIN_KEY: doc_id,
            **{field: meta.get(field, "") for field in METADATA_FIELDS},
        }

    def input_columns(self) -> list[str]:
        return [JOIN_KEY, TEXT_FIELD]

    def output_columns(self) -> list[str]:
        return ["text", JOIN_KEY] + METADATA_FIELDS


# ─────────────────────────────────────────────────────────────────────────────
# Metadata pre-loader
#   Downloads the metadata config shards once and builds a {id → row} lookup.
#   Must run before the pipeline so every Ray worker's extractor has the data.
# ─────────────────────────────────────────────────────────────────────────────
def load_metadata_lookup(repo_id: str, raw_dir: Path) -> dict[Any, dict]:
    """Return {id: {field: value}} from all metadata-config parquet shards."""
    fs = HfFileSystem()
    meta_paths = fs.glob(f"datasets/{repo_id}/metadata/**/*.parquet")
    if not meta_paths:
        raise RuntimeError(f"No parquet files found under 'metadata' config in {repo_id}.")

    lookup: dict[Any, dict] = {}
    columns = [JOIN_KEY] + METADATA_FIELDS

    for hf_path in meta_paths:
        rel = hf_path[len(f"datasets/{repo_id}/"):]
        local_path = raw_dir / rel

        if not local_path.exists():
            local_path.parent.mkdir(parents=True, exist_ok=True)
            fetched = hf_hub_download(
                repo_id=repo_id, repo_type="dataset", filename=rel, local_dir=str(raw_dir)
            )
            logger.info("Downloaded metadata shard → %s", fetched)
        else:
            logger.debug("Metadata shard cached: %s", local_path)

        for row in pq.read_table(str(local_path), columns=columns).to_pandas().itertuples(index=False):
            lookup[getattr(row, JOIN_KEY)] = {col: getattr(row, col, "") for col in METADATA_FIELDS}

    logger.info("Loaded metadata for %d documents", len(lookup))
    return lookup


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline builder
# ─────────────────────────────────────────────────────────────────────────────
def build_pipeline(datasets_dir: str, url_limit: int | None, record_limit: int | None) -> Pipeline:
    raw_dir    = Path(datasets_dir) / "vietnamese-legal-documents" / "raw"
    output_dir = Path(datasets_dir) / "vietnamese-legal-documents" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("─── Phase 1: loading metadata ───")
    metadata_lookup = load_metadata_lookup(HF_REPO_ID, raw_dir)

    logger.info("─── Phase 2: building pipeline ───")
    return Pipeline(
        name="viet_legal_download_pipeline",
        description="Download th1nhng0/vietnamese-legal-documents, join content + metadata.",
        stages=[
            DocumentDownloadExtractStage(
                url_generator=VietLegalURLGenerator(repo_id=HF_REPO_ID),
                downloader=VietLegalDownloader(download_dir=str(raw_dir), repo_id=HF_REPO_ID),
                iterator=VietLegalIterator(),
                extractor=VietLegalExtractor(metadata_lookup=metadata_lookup),
                url_limit=url_limit,
                record_limit=record_limit,
                add_filename_column="source_shard",
            ),
            ParquetWriter(path=str(output_dir)),
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download and prepare vietnamese-legal-documents")
    p.add_argument("--datasets_dir", default=_CFG.datasets_dir,
                   help="Root datasets directory (default: from config.yaml)")
    p.add_argument("--url_limit",    type=int, default=None,
                   help="Limit number of content shards (useful for testing)")
    p.add_argument("--record_limit", type=int, default=None,
                   help="Limit total records extracted across all shards")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("datasets_dir : %s", args.datasets_dir)
    logger.info("url_limit    : %s", args.url_limit)
    logger.info("record_limit : %s", args.record_limit)

    pipeline = build_pipeline(args.datasets_dir, args.url_limit, args.record_limit)

    logger.info("─── Phase 3: running pipeline ───")
    results = pipeline.run()

    written = list((Path(args.datasets_dir) / "vietnamese-legal-documents" / "data").glob("*.parquet"))
    logger.info("Done. %d parquet shards written.", len(written))
    if results:
        logger.info("Pipeline results: %s", results)


if __name__ == "__main__":
    main()
