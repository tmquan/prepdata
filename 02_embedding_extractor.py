"""
02_embedding_extractor.py — NeMo Curator embedding extraction (multi-GPU).

Pipeline (ParquetReader → EmbeddingCreatorStage → ParquetWriter).

Input
-----
<pipeline_data_dir>/<dataset>/preprocessed/part-*.parquet
    columns: text, …

Output
------
<pipeline_data_dir>/<dataset>/embedding/
    part-*.parquet  — embeddings column (float32 list, length = embedding_dim)
    metadata.json

Resume-safe: skips datasets whose embedding/ already contains parquet shards.

Usage
-----
    python 02_embedding_extractor.py
    python 02_embedding_extractor.py --dataset vietnamese-legal-documents
    python 02_embedding_extractor.py --file_limit 3
    python 02_embedding_extractor.py --dry_run
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.embedders import EmbeddingCreatorStage
from nemo_curator.stages.text.io.reader import ParquetReader
from nemo_curator.stages.text.io.writer import ParquetWriter

_CFG: DictConfig = OmegaConf.load(Path(__file__).parent / "config.yaml")


def _layout_subdir(name: str, default: str) -> str:
    v = OmegaConf.select(_CFG, f"curator_layout.{name}")
    return str(v) if v is not None else default


def _preprocessed_dir(pipeline_root: str | Path, dataset_key: str) -> Path:
    return Path(pipeline_root) / dataset_key / _layout_subdir("preprocessed", "preprocessed")


def _embedding_dir(pipeline_root: str | Path, dataset_key: str) -> Path:
    return Path(pipeline_root) / dataset_key / _layout_subdir("embedding", "embedding")


def _list_parquet_shards(shard_dir: Path) -> list[Path]:
    part = sorted(
        p for p in shard_dir.glob("part-*.parquet")
        if p.is_file() and not p.name.startswith(".")
    )
    if part:
        return part
    return sorted(
        p for p in shard_dir.glob("*.parquet")
        if p.is_file() and not p.name.startswith(".")
    )


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


_LIBRARY_PATCHES: list[tuple[str, str, str]] = [
    (
        "nemo_curator.stages.text.embedders.base",
        "AutoModel.from_pretrained(self.model_identifier, local_files_only=True)",
        "AutoModel.from_pretrained(self.model_identifier, local_files_only=True, trust_remote_code=True)",
    ),
    (
        "nemo_curator.stages.text.models.tokenizer",
        "AutoConfig.from_pretrained(\n            self.model_identifier, cache_dir=self.cache_dir, local_files_only=local_files_only\n        )",
        "AutoConfig.from_pretrained(\n            self.model_identifier, cache_dir=self.cache_dir, local_files_only=local_files_only, trust_remote_code=True\n        )",
    ),
    (
        "nemo_curator.stages.text.models.tokenizer",
        "AutoTokenizer.from_pretrained(\n            self.model_identifier,\n            padding_side=self.padding_side,\n            cache_dir=self.cache_dir,\n            local_files_only=local_files_only,\n        )",
        "AutoTokenizer.from_pretrained(\n            self.model_identifier,\n            padding_side=self.padding_side,\n            cache_dir=self.cache_dir,\n            local_files_only=local_files_only,\n            trust_remote_code=True,\n        )",
    ),
]


def _patch_nemo_curator_library() -> None:
    for module_path, old_text, new_text in _LIBRARY_PATCHES:
        spec = importlib.util.find_spec(module_path)
        if spec is None or spec.origin is None:
            logger.warning("Cannot locate module {} — skip patch", module_path)
            continue
        src_file = Path(spec.origin)
        source = src_file.read_text(encoding="utf-8")
        if new_text in source:
            continue
        if old_text not in source:
            logger.warning("Patch target missing in {} — Curator version may differ", src_file.name)
            continue
        src_file.write_text(source.replace(old_text, new_text, 1), encoding="utf-8")
        logger.info("Patched {} (trust_remote_code=True)", src_file.name)


def _ensure_model_cached(model_id: str) -> None:
    logger.info("Caching model: {}", model_id)
    snapshot_download(repo_id=model_id)
    logger.info("Model cache ready.")


def _build_executor(num_gpus: int):
    import ray
    from nemo_curator.backends.experimental.ray_data.executor import RayDataExecutor

    if not ray.is_initialized():
        ray.init(num_gpus=num_gpus, log_to_driver=True, ignore_reinit_error=True)
        logger.info("Ray initialised ({} GPU(s))", num_gpus)
    ctx = ray.data.DataContext.get_current()
    ctx.issue_detectors_config.high_memory_detector_config.detection_time_interval_s = -1
    return RayDataExecutor()


def _build_pipeline(
    input_files: list[str],
    output_dir: str,
    model_id: str,
    max_seq_length: int,
    batch_size: int,
    files_per_partition: int,
) -> Pipeline:
    return Pipeline(
        name="embedding_generation_pipeline",
        stages=[
            ParquetReader(
                file_paths=input_files,
                files_per_partition=files_per_partition,
                fields=["text"],
                _generate_ids=False,
            ),
            EmbeddingCreatorStage(
                model_identifier=model_id,
                text_field="text",
                max_seq_length=max_seq_length,
                max_chars=None,
                embedding_pooling="mean_pooling",
                model_inference_batch_size=batch_size,
            ),
            ParquetWriter(path=output_dir, fields=["embeddings"]),
        ],
    )


def _run_dataset(
    dataset_key: str,
    input_files: list[str],
    output_dir: Path,
    executor,
    model_id: str,
    max_seq_length: int,
    batch_size: int,
    files_per_partition: int,
    num_gpus: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("  shards in  : {}  |  out: {}", len(input_files), output_dir)

    t0 = time.perf_counter()
    pipeline = _build_pipeline(
        input_files=input_files,
        output_dir=str(output_dir),
        model_id=model_id,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        files_per_partition=files_per_partition,
    )
    output_tasks = pipeline.run(executor)
    elapsed = time.perf_counter() - t0

    num_docs = sum(task._stage_perf[-1].num_items_processed for task in output_tasks)
    thr = num_docs / elapsed if elapsed > 0 else 0
    logger.success("  {:,} docs in {:.1f}s ({:.1f} docs/s)", num_docs, elapsed, thr)

    n = _canonicalize_parquet_shards(output_dir)
    if n:
        logger.info("  Canonicalized {} embedding shard(s) → part-*-of-*.parquet", n)

    return {
        "dataset_key": dataset_key,
        "model_id": model_id,
        "embedding_dim": _CFG.embedding_dim,
        "max_seq_length": max_seq_length,
        "batch_size": batch_size,
        "num_gpus": num_gpus,
        "num_input_files": len(input_files),
        "num_documents": num_docs,
        "time_taken_s": round(elapsed, 2),
        "throughput_docs_per_sec": round(thr, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NeMo Curator embedding extraction")
    p.add_argument("--dataset", default=None, help="config.datasets key (default: all)")
    p.add_argument(
        "--pipeline_data_dir",
        default=_CFG.pipeline_data_dir,
        help="Root with <dataset>/preprocessed and <dataset>/embedding",
    )
    p.add_argument("--num_gpus", type=int, default=_CFG.num_gpus)
    p.add_argument("--batch_size", type=int, default=_CFG.batch_size)
    p.add_argument("--max_seq_length", type=int, default=_CFG.max_seq_length)
    p.add_argument("--files_per_partition", type=int, default=_CFG.files_per_partition)
    p.add_argument("--file_limit", type=int, default=None)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    keys = [args.dataset] if args.dataset else list(_CFG.datasets.keys())

    logger.info("NeMo Curator — embedding extraction")
    logger.info("  model            : {}", _CFG.model_id)
    logger.info("  pipeline_data_dir: {}", args.pipeline_data_dir)
    logger.info("  datasets         : {}", keys)

    work_items = []
    for key in keys:
        if key not in _CFG.datasets:
            logger.warning("Unknown dataset '{}' — skip", key)
            continue
        pre = _preprocessed_dir(args.pipeline_data_dir, key)
        out = _embedding_dir(args.pipeline_data_dir, key)
        inputs = _list_parquet_shards(pre)
        if not inputs:
            logger.warning("No preprocessed parquet in {} — run 00_datasets_downloader.py", pre)
            continue
        if args.file_limit:
            inputs = inputs[: args.file_limit]
        rows = sum(pq.ParquetFile(str(f)).metadata.num_rows for f in inputs)
        work_items.append({"key": key, "input_files": inputs, "output_dir": out, "total_rows": rows})
        logger.info("  {} : {} shards, {:,} rows → {}", key, len(inputs), rows, out)

    if not work_items:
        logger.warning("No work items.")
        sys.exit(0)

    if args.dry_run:
        total = sum(w["total_rows"] for w in work_items)
        est_gb = total * _CFG.embedding_dim * 4 / 1e9
        logger.info("DRY RUN — {:,} rows, ~{:.1f} GB embeddings", total, est_gb)
        sys.exit(0)

    _patch_nemo_curator_library()
    _ensure_model_cached(_CFG.model_id)
    executor = _build_executor(args.num_gpus)

    t_wall = time.perf_counter()
    errors: list[tuple[str, BaseException]] = []

    for item in work_items:
        key, out = item["key"], item["output_dir"]
        logger.info("— {} ({:,} rows) —", key, item["total_rows"])
        existing = list(out.glob("*.parquet")) if out.exists() else []
        if existing:
            logger.info("  embedding/ already has {} shard(s) — skip", len(existing))
            continue
        try:
            meta = _run_dataset(
                dataset_key=key,
                input_files=[str(f) for f in item["input_files"]],
                output_dir=out,
                executor=executor,
                model_id=_CFG.model_id,
                max_seq_length=args.max_seq_length,
                batch_size=args.batch_size,
                files_per_partition=args.files_per_partition,
                num_gpus=args.num_gpus,
            )
            mp = out / "metadata.json"
            mp.write_text(json.dumps(meta, indent=2))
            logger.info("  metadata → {}", mp)
        except Exception as exc:
            logger.error("  FAILED {}: {}", key, exc)
            errors.append((key, exc))

    logger.info("Finished in {:.1f}s", time.perf_counter() - t_wall)
    if errors:
        for k, e in errors:
            logger.error("  FAILED {}: {}", k, e)
        sys.exit(1)


if __name__ == "__main__":
    main()
