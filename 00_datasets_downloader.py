"""
00_datasets_downloader.py — HuggingFace → NeMo Curator preprocessed parquet.

Uses the DocumentDownloadExtractStage pattern (URLGenerator → Downloader →
Iterator → Extractor) and ParquetWriter, matching NeMo Curator’s download
pipelines.

Output
------
<pipeline_data_dir>/<dataset_key>/
    raw/              — cached HF shards (metadata + content)
    preprocessed/     — NeMo-ready part-*.parquet (text + id + metadata columns)

Usage
-----
    python 00_datasets_downloader.py
    python 00_datasets_downloader.py --dataset vietnamese-legal-documents
    python 00_datasets_downloader.py --pipeline_data_dir /my/root --url_limit 5
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.download.base.download import DocumentDownloader
from nemo_curator.stages.text.download.base.extract import DocumentExtractor
from nemo_curator.stages.text.download.base.iterator import DocumentIterator
from nemo_curator.stages.text.download.base.stage import DocumentDownloadExtractStage
from nemo_curator.stages.text.download.base.url_generation import URLGenerator
from nemo_curator.stages.text.io.writer import ParquetWriter

_CFG: DictConfig = OmegaConf.load(Path(__file__).parent / "config.yaml")


def _layout_subdir(name: str, default: str) -> str:
    v = OmegaConf.select(_CFG, f"curator_layout.{name}")
    return str(v) if v is not None else default


def _raw_dir(pipeline_root: str | Path, dataset_key: str) -> Path:
    return Path(pipeline_root) / dataset_key / _layout_subdir("raw", "raw")


def _preprocessed_dir(pipeline_root: str | Path, dataset_key: str) -> Path:
    return Path(pipeline_root) / dataset_key / _layout_subdir("preprocessed", "preprocessed")


def _canonicalize_parquet_shards(directory: Path) -> int:
    files = sorted(
        p for p in directory.glob("*.parquet")
        if p.is_file() and not p.name.startswith(".")
    )
    if not files:
        return 0
    n = len(files)
    targets = [directory / f"part-{i:05d}-of-{n:05d}.parquet" for i in range(n)]
    if all(f.name == t.name for f, t in zip(files, targets, strict=True)):
        return 0
    tmp = [directory / f".layout_tmp_{i:05d}.parquet" for i in range(n)]
    for f, t in zip(files, tmp, strict=True):
        f.rename(t)
    for t, final in zip(tmp, targets, strict=True):
        t.rename(final)
    return n


class DatasetURLGenerator(URLGenerator):
    """List content-config parquet URLs for a HuggingFace dataset repo."""

    def __init__(self, repo_id: str) -> None:
        self.repo_id = repo_id

    def generate_urls(self) -> list[str]:
        fs = HfFileSystem()
        paths = fs.glob(f"datasets/{self.repo_id}/content/**/*.parquet")
        if not paths:
            raise RuntimeError(
                f"No parquet files under 'content' config in {self.repo_id}."
            )
        base = f"datasets/{self.repo_id}/"
        urls = [
            f"https://huggingface.co/datasets/{self.repo_id}/resolve/main/"
            + p[len(base) :]
            for p in paths
        ]
        logger.info("Found {} content shards in {}", len(urls), self.repo_id)
        return urls


class HfHubParquetDownloader(DocumentDownloader):
    """Download dataset parquet shards via huggingface_hub (resume-safe)."""

    def __init__(self, download_dir: str, repo_id: str) -> None:
        super().__init__(download_dir=download_dir, verbose=True)
        self._local_dir = download_dir
        self.repo_id = repo_id

    def _get_output_filename(self, url: str) -> str:
        marker = "/resolve/main/"
        return url.split(marker, 1)[1] if marker in url else url.split("/")[-1]

    def _download_to_path(self, url: str, path: str) -> tuple[bool, str | None]:
        return False, "unreachable: download() overridden"

    def download(self, url: str) -> str | None:
        rel_path = self._get_output_filename(url)
        output_file = Path(self._local_dir) / rel_path
        if output_file.exists():
            logger.debug("Cached: {}", output_file)
            return str(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            fetched = hf_hub_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                filename=rel_path,
                local_dir=self._local_dir,
            )
            logger.info("Downloaded → {}", fetched)
            return fetched
        except Exception as exc:
            logger.error("Failed {}: {}", url, exc)
            return None


class ParquetColumnIterator(DocumentIterator):
    """Stream join_key + text_field rows from one parquet shard."""

    def __init__(self, join_key: str, text_field: str) -> None:
        self.join_key = join_key
        self.text_field = text_field

    def iterate(self, file_path: str) -> Iterator[dict[str, Any]]:
        table = pq.read_table(file_path, columns=[self.join_key, self.text_field])
        for batch in table.to_batches(max_chunksize=1000):
            df = batch.to_pandas()
            for row in df.itertuples(index=False):
                yield {
                    self.join_key: getattr(row, self.join_key),
                    self.text_field: getattr(row, self.text_field),
                }

    def output_columns(self) -> list[str]:
        return [self.join_key, self.text_field]


class MetadataJoinExtractor(DocumentExtractor):
    """Join pre-loaded metadata; rename text column to ``text``."""

    def __init__(
        self,
        metadata_lookup: dict[Any, dict],
        join_key: str,
        text_field: str,
        metadata_fields: list[str],
    ) -> None:
        self._lookup = metadata_lookup
        self.join_key = join_key
        self.text_field = text_field
        self.metadata_fields = metadata_fields

    def extract(self, record: dict[str, str]) -> dict[str, Any] | None:
        doc_id = record.get(self.join_key)
        content = record.get(self.text_field, "")
        if doc_id is None or not content or not content.strip():
            return None
        meta = self._lookup.get(doc_id, {})
        return {
            "text": content.strip(),
            self.join_key: doc_id,
            **{f: meta.get(f, "") for f in self.metadata_fields},
        }

    def input_columns(self) -> list[str]:
        return [self.join_key, self.text_field]

    def output_columns(self) -> list[str]:
        return ["text", self.join_key] + self.metadata_fields


def load_metadata_lookup(
    repo_id: str,
    raw_base: Path,
    join_key: str,
    metadata_fields: list[str],
) -> dict[Any, dict]:
    """Build {join_key: {field: value}} from all metadata-config parquet shards."""
    fs = HfFileSystem()
    meta_paths = fs.glob(f"datasets/{repo_id}/metadata/**/*.parquet")
    if not meta_paths:
        raise RuntimeError(f"No parquet under 'metadata' config in {repo_id}.")

    lookup: dict[Any, dict] = {}
    columns = [join_key] + metadata_fields
    prefix = f"datasets/{repo_id}/"

    for hf_path in meta_paths:
        rel = hf_path[len(prefix) :]
        local_path = raw_base / rel
        if not local_path.exists():
            local_path.parent.mkdir(parents=True, exist_ok=True)
            fetched = hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=rel,
                local_dir=str(raw_base),
            )
            logger.info("Metadata shard → {}", fetched)
        for row in pq.read_table(str(local_path), columns=columns).to_pandas().itertuples(
            index=False
        ):
            lookup[getattr(row, join_key)] = {
                col: getattr(row, col, "") for col in metadata_fields
            }

    logger.info("Loaded metadata for {} documents", len(lookup))
    return lookup


def build_pipeline(
    pipeline_root: str,
    dataset_key: str,
    ds_cfg: DictConfig,
    url_limit: int | None,
    record_limit: int | None,
) -> tuple[Pipeline, Path]:
    repo_id = ds_cfg.hf_name
    join_key = ds_cfg.join_key
    text_field = ds_cfg.text_field
    metadata_fields: list[str] = OmegaConf.to_container(ds_cfg.metadata_fields, resolve=True)

    rdir = _raw_dir(pipeline_root, dataset_key)
    out = _preprocessed_dir(pipeline_root, dataset_key)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Loading metadata lookup…")
    lookup = load_metadata_lookup(repo_id, rdir, join_key, metadata_fields)

    pipeline = Pipeline(
        name=f"download_{dataset_key}",
        description=f"HF → preprocessed parquet ({repo_id})",
        stages=[
            DocumentDownloadExtractStage(
                url_generator=DatasetURLGenerator(repo_id=repo_id),
                downloader=HfHubParquetDownloader(download_dir=str(rdir), repo_id=repo_id),
                iterator=ParquetColumnIterator(join_key=join_key, text_field=text_field),
                extractor=MetadataJoinExtractor(
                    metadata_lookup=lookup,
                    join_key=join_key,
                    text_field=text_field,
                    metadata_fields=metadata_fields,
                ),
                url_limit=url_limit,
                record_limit=record_limit,
                add_filename_column="source_shard",
            ),
            ParquetWriter(path=str(out)),
        ],
    )
    return pipeline, out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download HF dataset → NeMo preprocessed parquet")
    p.add_argument("--dataset", default="vietnamese-legal-documents", help="Key under config.datasets")
    p.add_argument(
        "--pipeline_data_dir",
        default=_CFG.pipeline_data_dir,
        help="Root containing <dataset_key>/preprocessed (default: config.yaml)",
    )
    p.add_argument("--url_limit", type=int, default=None, help="Max content shards (testing)")
    p.add_argument("--record_limit", type=int, default=None, help="Max total records")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    key = args.dataset
    if key not in _CFG.datasets:
        logger.error("Unknown dataset '{}'; add it to config.yaml", key)
        raise SystemExit(1)

    ds_cfg = _CFG.datasets[key]
    if ds_cfg.get("format") != "parquet":
        logger.error("Dataset '{}' must have format: parquet for this downloader", key)
        raise SystemExit(1)

    logger.info("pipeline_data_dir : {}", args.pipeline_data_dir)
    logger.info("dataset           : {}", key)
    logger.info("url_limit         : {}", args.url_limit)
    logger.info("record_limit      : {}", args.record_limit)

    pipeline, preprocessed_out = build_pipeline(
        args.pipeline_data_dir, key, ds_cfg, args.url_limit, args.record_limit
    )

    logger.info("Running pipeline…")
    results = pipeline.run()

    n = _canonicalize_parquet_shards(preprocessed_out)
    if n:
        logger.info("Canonicalized {} preprocessed shard(s) → part-*-of-*.parquet", n)

    shards = list(preprocessed_out.glob("part-*.parquet"))
    logger.info("Done. {} preprocessed parquet shard(s) in {}", len(shards), preprocessed_out)
    if results:
        logger.debug("Pipeline results: {}", results)


if __name__ == "__main__":
    main()
