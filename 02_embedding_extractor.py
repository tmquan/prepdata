"""
02_embedding_extractor.py – Extract embeddings from NeMo-ready parquet datasets
using the NeMo Curator Pipeline with RayDataExecutor for multi-GPU processing.

Pipeline (PDF slide 14)
-----------------------
ParquetReader  →  EmbeddingCreatorStage  →  ParquetWriter

Input
-----
<datasets_dir>/<dataset>/data/part-*.parquet   (from 00_datasets_downloader.py)
    columns: text, id, …metadata…

Output
------
<embeddings_dir>/<dataset>/
    part-*.parquet   columns: embeddings  (float32 list, length = embedding_dim)
    metadata.json

Resume-safe: existing output directories that already contain .parquet files
are skipped automatically.

Usage
-----
    python 02_embedding_extractor.py
    python 02_embedding_extractor.py --dataset vietnamese-legal-documents
    python 02_embedding_extractor.py --file_limit 3          # quick test
    python 02_embedding_extractor.py --dry_run               # show plan only
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

# ─── Config ───────────────────────────────────────────────────────────────────
_CFG: DictConfig = OmegaConf.load(Path(__file__).parent / "config.yaml")


# ─────────────────────────────────────────────────────────────────────────────
# NeMo Curator library patch
#
# nvidia/llama-embed-nemotron-8b ships custom Python code, so every
# AutoModel / AutoTokenizer / AutoConfig.from_pretrained call inside NeMo
# Curator must pass trust_remote_code=True.
#
# A driver-process monkey-patch is NOT enough because Ray workers are separate
# subprocesses that re-import the library fresh.  We patch the source files on
# disk so every worker picks up the fix automatically at import time.
#
# Each entry: (module_dotpath, old_snippet, new_snippet)
# The function is idempotent – it skips files that are already patched.
# ─────────────────────────────────────────────────────────────────────────────
_LIBRARY_PATCHES: list[tuple[str, str, str]] = [
    # ── embedders/base.py ── AutoModel.from_pretrained ───────────────────────
    (
        "nemo_curator.stages.text.embedders.base",
        "AutoModel.from_pretrained(self.model_identifier, local_files_only=True)",
        "AutoModel.from_pretrained(self.model_identifier, local_files_only=True, trust_remote_code=True)",
    ),
    # ── models/tokenizer.py ── AutoConfig.from_pretrained ────────────────────
    (
        "nemo_curator.stages.text.models.tokenizer",
        "AutoConfig.from_pretrained(\n            self.model_identifier, cache_dir=self.cache_dir, local_files_only=local_files_only\n        )",
        "AutoConfig.from_pretrained(\n            self.model_identifier, cache_dir=self.cache_dir, local_files_only=local_files_only, trust_remote_code=True\n        )",
    ),
    # ── models/tokenizer.py ── AutoTokenizer.from_pretrained ─────────────────
    (
        "nemo_curator.stages.text.models.tokenizer",
        "AutoTokenizer.from_pretrained(\n            self.model_identifier,\n            padding_side=self.padding_side,\n            cache_dir=self.cache_dir,\n            local_files_only=local_files_only,\n        )",
        "AutoTokenizer.from_pretrained(\n            self.model_identifier,\n            padding_side=self.padding_side,\n            cache_dir=self.cache_dir,\n            local_files_only=local_files_only,\n            trust_remote_code=True,\n        )",
    ),
]


def _patch_nemo_curator_library() -> None:
    """Edit NeMo Curator source files on disk to add trust_remote_code=True.

    Idempotent: files already containing the patched snippet are left unchanged.

    NOTE: we do NOT touch sys.modules after patching.  The driver process has
    already imported the library and only uses the stage objects for building /
    pickling the pipeline — it never calls setup() or from_pretrained().
    Ray workers are separate subprocesses that import the library fresh from
    the (now-patched) .py files, so they pick up the fix automatically.
    Reloading modules in the driver would create a new lru_cache wrapper for
    TokenizerStage.load_cfg that differs from the one already held by the
    stage instance, causing Ray's pickler to raise an identity-mismatch error.
    """
    for module_path, old_text, new_text in _LIBRARY_PATCHES:
        spec = importlib.util.find_spec(module_path)
        if spec is None or spec.origin is None:
            logger.warning("Cannot locate module {} – skipping patch", module_path)
            continue

        src_file = Path(spec.origin)
        source = src_file.read_text(encoding="utf-8")

        if new_text in source:
            logger.debug("Already patched: {}", src_file.name)
            continue

        if old_text not in source:
            logger.warning(
                "Patch target not found in {} – NeMo Curator version may differ",
                src_file.name,
            )
            continue

        src_file.write_text(source.replace(old_text, new_text, 1), encoding="utf-8")
        logger.info("Patched {}: trust_remote_code=True added", src_file.name)


def _ensure_model_cached(model_id: str) -> None:
    """Download model to HF cache so NeMo's local_files_only=True succeeds."""
    logger.info("Ensuring model is cached: {}", model_id)
    snapshot_download(repo_id=model_id)
    logger.info("Model cache ready.")


# ─────────────────────────────────────────────────────────────────────────────
# Executor
# ─────────────────────────────────────────────────────────────────────────────
def _build_executor(num_gpus: int):
    """Initialise Ray and return a RayDataExecutor for multi-GPU inference."""
    import ray
    from nemo_curator.backends.experimental.ray_data.executor import RayDataExecutor

    if not ray.is_initialized():
        ray.init(
            num_gpus=num_gpus,
            log_to_driver=True,
            ignore_reinit_error=True,
        )
        logger.info("Ray initialised with {} GPU(s)", num_gpus)

    # Suppress noisy memory detector warnings (advisory only)
    ctx = ray.data.DataContext.get_current()
    ctx.issue_detectors_config.high_memory_detector_config.detection_time_interval_s = -1

    return RayDataExecutor()


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline builder
# ─────────────────────────────────────────────────────────────────────────────
def _build_pipeline(
    input_files: list[str],
    output_dir: str,
    model_id: str,
    max_seq_length: int,
    batch_size: int,
    files_per_partition: int,
) -> Pipeline:
    """Construct the NeMo Curator embedding pipeline (PDF slide 14).

    Stages:
      1. ParquetReader         – stream text shards from disk
      2. EmbeddingCreatorStage – tokenise + embed with the LLM model
      3. ParquetWriter         – write embeddings to output_dir
    """
    return Pipeline(
        name="embedding_generation_pipeline",
        stages=[
            # ── Step 1 ────────────────────────────────────────────────────────
            ParquetReader(
                file_paths=input_files,
                files_per_partition=files_per_partition,
                fields=["text"],
                _generate_ids=False,
            ),
            # ── Step 2 ────────────────────────────────────────────────────────
            EmbeddingCreatorStage(
                model_identifier=model_id,
                text_field="text",
                max_seq_length=max_seq_length,
                max_chars=None,
                embedding_pooling="mean_pooling",
                model_inference_batch_size=batch_size,
            ),
            # ── Step 3 ────────────────────────────────────────────────────────
            ParquetWriter(
                path=output_dir,
                fields=["embeddings"],
            ),
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset runner
# ─────────────────────────────────────────────────────────────────────────────
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
    """Run the embedding pipeline for one dataset; return result metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("  Input shards   : {}", len(input_files))
    logger.info("  Output dir     : {}", output_dir)
    logger.info("  Model          : {}", model_id)
    logger.info("  max_seq_length : {}  batch_size : {}", max_seq_length, batch_size)

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

    num_docs = sum(
        task._stage_perf[-1].num_items_processed for task in output_tasks
    )
    throughput = num_docs / elapsed if elapsed > 0 else 0

    logger.success(
        "Done – {:,} docs in {:.1f}s ({:.1f} docs/s)",
        num_docs, elapsed, throughput,
    )

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
        "throughput_docs_per_sec": round(throughput, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract embeddings from NeMo-ready parquet datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset",       default=None,                    help="Single dataset key from config.yaml (default: all)")
    p.add_argument("--datasets_dir",  default=_CFG.datasets_dir,      help="Root datasets directory")
    p.add_argument("--embeddings_dir",default=_CFG.embeddings_dir,    help="Root embeddings output directory")
    p.add_argument("--num_gpus",      type=int, default=_CFG.num_gpus,help="Number of GPUs for Ray (default: from config.yaml)")
    p.add_argument("--batch_size",    type=int, default=_CFG.batch_size)
    p.add_argument("--max_seq_length",type=int, default=_CFG.max_seq_length)
    p.add_argument("--files_per_partition", type=int, default=_CFG.files_per_partition)
    p.add_argument("--file_limit",    type=int, default=None,          help="Process only the first N shards (for testing)")
    p.add_argument("--dry_run",       action="store_true",             help="Show plan without processing")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    dataset_keys = [args.dataset] if args.dataset else list(_CFG.datasets.keys())

    logger.info("=" * 68)
    logger.info("NeMo Curator Embedding Extractor")
    logger.info("=" * 68)
    logger.info("  model          : {}", _CFG.model_id)
    logger.info("  embedding_dim  : {}", _CFG.embedding_dim)
    logger.info("  num_gpus       : {}", args.num_gpus)
    logger.info("  batch_size     : {}", args.batch_size)
    logger.info("  max_seq_length : {}", args.max_seq_length)
    logger.info("  datasets_dir   : {}", args.datasets_dir)
    logger.info("  embeddings_dir : {}", args.embeddings_dir)
    logger.info("  datasets       : {}", dataset_keys)

    # ── Collect work ──────────────────────────────────────────────────────────
    work_items = []
    for key in dataset_keys:
        if key not in _CFG.datasets:
            logger.warning("Dataset '{}' not found in config.yaml – skipping", key)
            continue

        input_dir  = Path(args.datasets_dir)  / key / "data"
        output_dir = Path(args.embeddings_dir) / key

        input_files = sorted(input_dir.glob("*.parquet"))
        if not input_files:
            logger.warning("No parquet files in {} – run 00_datasets_downloader.py first", input_dir)
            continue
        if args.file_limit:
            input_files = input_files[: args.file_limit]

        total_rows = sum(pq.ParquetFile(str(f)).metadata.num_rows for f in input_files)
        work_items.append(dict(
            key=key, input_files=input_files,
            output_dir=output_dir, total_rows=total_rows,
        ))
        logger.info("  {} : {} shards, {:,} rows", key, len(input_files), total_rows)

    if not work_items:
        logger.warning("No work items found. Exiting.")
        sys.exit(0)

    # ── Dry-run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        total = sum(w["total_rows"] for w in work_items)
        est_gb = total * _CFG.embedding_dim * 4 / 1e9
        logger.info("DRY RUN – {:,} total rows, ~{:.1f} GB estimated output", total, est_gb)
        sys.exit(0)

    # ── Pre-flight: patch library files + cache model ────────────────────────
    _patch_nemo_curator_library()
    _ensure_model_cached(_CFG.model_id)

    # ── Build executor once (Ray init is expensive) ───────────────────────────
    executor = _build_executor(args.num_gpus)

    # ── Process each dataset ──────────────────────────────────────────────────
    t_wall = time.perf_counter()
    errors = []

    for item in work_items:
        key        = item["key"]
        output_dir = item["output_dir"]

        logger.info("-" * 68)
        logger.info("Dataset: {}  ({:,} rows)", key, item["total_rows"])

        # Resume safety: skip if embeddings already exist
        existing = list(output_dir.glob("*.parquet")) if output_dir.exists() else []
        if existing:
            logger.info("  Embeddings already exist ({} shards) – skipping", len(existing))
            continue

        try:
            meta = _run_dataset(
                dataset_key=key,
                input_files=[str(f) for f in item["input_files"]],
                output_dir=output_dir,
                executor=executor,
                model_id=_CFG.model_id,
                max_seq_length=args.max_seq_length,
                batch_size=args.batch_size,
                files_per_partition=args.files_per_partition,
                num_gpus=args.num_gpus,
            )
            # Write metadata.json alongside embeddings
            meta_path = output_dir / "metadata.json"
            meta_path.write_text(json.dumps(meta, indent=2))
            logger.info("  Metadata → {}", meta_path)

        except Exception as exc:
            logger.error("  FAILED {}: {}", key, exc)
            errors.append((key, exc))

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_wall
    logger.info("=" * 68)
    logger.info("Finished in {:.1f}s ({:.1f} min)", elapsed, elapsed / 60)
    if errors:
        for key, exc in errors:
            logger.error("  FAILED {}: {}", key, exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
