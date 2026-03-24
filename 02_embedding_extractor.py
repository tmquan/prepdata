"""
02_embedding_extractor.py — NeMo Curator embedding extraction (multi-GPU).

Pipeline (ParquetReader → EmbeddingCreatorStage → ParquetWriter).

Input
-----
<pipeline_data_dir>/<dataset>/preprocessed/part-*.parquet
    columns: text, …

Output
------
If ``embeddings_curator_dir`` is set in config.yaml:
    <embeddings_curator_dir>/<dataset>/part-*.parquet
else:
    <pipeline_data_dir>/<dataset>/<embedding>/
    part-*.parquet  — embeddings column (float32 list, length = embedding_dim)
    metadata.json

Resume-safe: skips datasets whose embedding/ already contains parquet shards.

Usage
-----
    python 02_embedding_extractor.py --list
    python 02_embedding_extractor.py --list --list_format tsv
    python 02_embedding_extractor.py --datasets vietnamese-legal-documents
    python 02_embedding_extractor.py --datasets @selected.txt
    python 02_embedding_extractor.py --dataset vietnamese-legal-documents
    python 02_embedding_extractor.py --file_limit 3
    python 02_embedding_extractor.py --dry_run

``--list`` scans ``pipeline_data_dir`` (datasets root) and ``config.yaml`` together:
every name that appears as a top-level directory **or** as a ``datasets:`` key, with
status ``ready`` / ``missing_preprocessed`` / ``disk_only_ready``.

Embedding runs on any folder that has preprocessed shards (including disk-only names
not listed in config). ``--datasets`` uses exact directory/config keys (see ``--list``).
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


def _resolved_embeddings_curator_root(cli_override: str | None) -> Path | None:
    """Central embedding parquet root: CLI wins, else ``config.yaml`` ``embeddings_curator_dir``."""
    if cli_override is not None and str(cli_override).strip():
        return Path(str(cli_override).strip())
    raw = OmegaConf.select(_CFG, "embeddings_curator_dir")
    if raw is not None and str(raw).strip():
        return Path(str(raw).strip())
    return None


def _embedding_dir(
    pipeline_root: str | Path,
    dataset_key: str,
    embeddings_curator_root: Path | None,
) -> Path:
    if embeddings_curator_root is not None:
        return embeddings_curator_root / dataset_key
    return Path(pipeline_root) / dataset_key / _layout_subdir("embedding", "embedding")


def _list_parquet_shards(shard_dir: Path) -> list[Path]:
    """``part-*.parquet`` and ``part_*_of_*.parquet`` (underscore style)."""
    seen: set[str] = set()
    out: list[Path] = []
    for pattern in ("part-*.parquet", "part_*_of_*.parquet"):
        for p in shard_dir.glob(pattern):
            if not p.is_file() or p.name.startswith(".") or p.name in seen:
                continue
            seen.add(p.name)
            out.append(p)
    if out:
        return sorted(out, key=lambda x: x.name)
    return sorted(
        p for p in shard_dir.glob("*.parquet")
        if p.is_file() and not p.name.startswith(".")
    )


def _dataset_filter_matches(filter_s: str | None, key: str) -> bool:
    if filter_s is None:
        return True
    return key == filter_s or key.startswith(filter_s + "/")


def _parse_datasets_keys(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    keys: set[str] = set()
    for raw in values:
        s = raw.strip()
        if not s:
            continue
        if s.startswith("@") and len(s) > 1:
            path = Path(s[1:]).expanduser()
            if not path.is_file():
                raise SystemExit(f"--datasets: file not found: {path}")
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    keys.add(line)
        else:
            keys.add(s)
    return keys if keys else None


def _top_level_dir_names(pipeline_root: Path) -> list[str]:
    """Names of immediate subdirectories under ``pipeline_data_dir`` (datasets on disk)."""
    if not pipeline_root.is_dir():
        return []
    return sorted(
        p.name
        for p in pipeline_root.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )


def _all_dataset_keys(pipeline_root: Path) -> list[str]:
    """Union of ``config.datasets`` keys and top-level names under ``pipeline_data_dir``."""
    names = set(_CFG.datasets.keys()) | set(_top_level_dir_names(pipeline_root))
    return sorted(names)


def _describe_dataset(
    key: str,
    pipeline_root: Path,
    embeddings_curator_root: Path | None,
) -> dict:
    """One row: preprocessed shards, output path, config membership, coarse status."""
    in_config = key in _CFG.datasets
    hf_name = ""
    if in_config:
        hf = _CFG.datasets[key].get("hf_name", "")
        hf_name = str(hf) if hf is not None else ""

    pre = _preprocessed_dir(pipeline_root, key)
    inputs = _list_parquet_shards(pre) if pre.is_dir() else []
    rows = (
        sum(pq.ParquetFile(str(f)).metadata.num_rows for f in inputs) if inputs else 0
    )
    out = _embedding_dir(pipeline_root, key, embeddings_curator_root)

    if inputs:
        status = "ready" if in_config else "disk_only_ready"
    else:
        status = "missing_preprocessed"

    return {
        "key": key,
        "in_config": in_config,
        "status": status,
        "input_files": inputs,
        "total_rows": rows,
        "preprocessed_dir": pre,
        "output_dir": out,
        "hf_name": hf_name,
    }


def _list_datasets_dir_snapshot(
    pipeline_root: Path,
    embeddings_curator_root: Path | None,
    dataset_filter: str | None,
) -> list[dict]:
    """All known dataset keys (config ∪ disk) with preprocessed / status."""
    rows: list[dict] = []
    for key in _all_dataset_keys(pipeline_root):
        if not _dataset_filter_matches(dataset_filter, key):
            continue
        rows.append(_describe_dataset(key, pipeline_root, embeddings_curator_root))
    return rows


def _collect_embedding_candidates(
    pipeline_root: str | Path,
    embeddings_curator_root: Path | None,
    dataset_filter: str | None,
) -> list[dict]:
    """Datasets with preprocessed parquet (config and/or disk-only under ``pipeline_data_dir``)."""
    root = Path(pipeline_root)
    items: list[dict] = []
    for row in _list_datasets_dir_snapshot(root, embeddings_curator_root, dataset_filter):
        if not row["input_files"]:
            continue
        items.append(
            {
                "key": row["key"],
                "input_files": row["input_files"],
                "output_dir": row["output_dir"],
                "total_rows": row["total_rows"],
                "preprocessed_dir": row["preprocessed_dir"],
                "hf_name": row["hf_name"],
                "in_config": row["in_config"],
                "status": row["status"],
            }
        )
    return items


def _apply_datasets_key_filter(
    candidates: list[dict], wanted: set[str], pipeline_root: Path
) -> list[dict]:
    have = {c["key"] for c in candidates}
    missing = wanted - have
    if missing:
        for m in sorted(missing):
            ds_dir = pipeline_root / m
            pre = _preprocessed_dir(pipeline_root, m)
            if not ds_dir.is_dir():
                logger.error(
                    "No dataset directory '{}' under {} (see --list)",
                    m,
                    pipeline_root,
                )
            elif not pre.is_dir() or not _list_parquet_shards(pre):
                logger.error(
                    "No preprocessed parquet for '{}' under {} — run 00_datasets_downloader.py",
                    m,
                    pre,
                )
            else:
                logger.error("Dataset '{}' could not be selected — run with --list", m)
        logger.info("Run with --list to see datasets under pipeline_data_dir and config.")
        return []
    return [c for c in candidates if c["key"] in wanted]


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
    p = argparse.ArgumentParser(
        description="NeMo Curator embedding extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --list\n"
            "  %(prog)s --list --list_format tsv > ready.tsv\n"
            "  %(prog)s --datasets vietnamese-legal-documents\n"
            "  %(prog)s --datasets @selected.txt --dry_run\n"
        ),
    )
    p.add_argument(
        "--list",
        action="store_true",
        help=(
            "List dataset names from pipeline_data_dir ∪ config.yaml with status "
            "(ready / missing_preprocessed / disk_only_ready) and exit"
        ),
    )
    p.add_argument(
        "--list_format",
        choices=("plain", "tsv", "json"),
        default="plain",
        help="Output style for --list (default: plain)",
    )
    p.add_argument(
        "--dataset",
        default=None,
        metavar="PREFIX",
        help="Only datasets whose key equals PREFIX or starts with PREFIX/ (single filter)",
    )
    p.add_argument(
        "--datasets",
        nargs="*",
        metavar="KEY",
        default=None,
        help=(
            "Exact dataset name(s): must match a top-level dir under pipeline_data_dir "
            "(space-separated). Use @path for a key file (one per line, # ok). "
            "Do not combine with --dataset."
        ),
    )
    p.add_argument(
        "--pipeline_data_dir",
        default=_CFG.pipeline_data_dir,
        help="Root with <dataset>/preprocessed; embeddings go to embeddings_curator_dir if set",
    )
    p.add_argument(
        "--embeddings_curator_dir",
        default=None,
        help="Override config.yaml embeddings_curator_dir for output parquet root",
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

    if args.dataset and bool(args.datasets):
        logger.error("Use either --dataset or --datasets, not both.")
        sys.exit(2)

    datasets_keys = _parse_datasets_keys(args.datasets)

    pipeline_root = Path(args.pipeline_data_dir)
    curator_root = _resolved_embeddings_curator_root(args.embeddings_curator_dir)

    if args.list:
        rows = _list_datasets_dir_snapshot(pipeline_root, curator_root, None)
        if args.list_format == "json":
            payload = [
                {
                    "key": r["key"],
                    "in_config": r["in_config"],
                    "status": r["status"],
                    "rows": r["total_rows"],
                    "shards": len(r["input_files"]),
                    "hf_name": r["hf_name"],
                    "preprocessed": str(r["preprocessed_dir"]),
                    "embeddings_out": str(r["output_dir"]),
                }
                for r in rows
            ]
            print(json.dumps(payload, indent=2))
        elif args.list_format == "tsv":
            print("key\tin_config\tstatus\trows\tshards\thf_name\tpreprocessed\tembeddings_out")
            for r in rows:
                print(
                    f"{r['key']}\t{r['in_config']}\t{r['status']}\t{r['total_rows']}\t"
                    f"{len(r['input_files'])}\t{r['hf_name']}\t{r['preprocessed_dir']}\t{r['output_dir']}"
                )
        else:
            wk = max((len(r["key"]) for r in rows), default=0)
            wk = min(wk, 48)
            print(
                f"{'key':<{wk}}  {'cfg':>3}  {'status':<20}  {'rows':>12}  {'sh':>4}  embeddings_out"
            )
            print("-" * (wk + 3 + 20 + 12 + 4 + 45))
            for r in rows:
                k = r["key"] if len(r["key"]) <= wk else r["key"][: wk - 3] + "..."
                cfg = "yes" if r["in_config"] else "no"
                out_s = str(r["output_dir"])
                if len(out_s) > 56:
                    out_s = "..." + out_s[-53:]
                print(
                    f"{k:<{wk}}  {cfg:>3}  {r['status']:<20}  {r['total_rows']:12,}  "
                    f"{len(r['input_files']):4d}  {out_s}"
                )
        n_ready = sum(1 for r in rows if r["status"] in ("ready", "disk_only_ready"))
        n_wait = sum(1 for r in rows if r["status"] == "missing_preprocessed")
        print(
            f"\nTotal: {len(rows)} name(s) — {n_ready} embed-ready, {n_wait} missing preprocessed",
            file=sys.stderr,
        )
        print(f"pipeline_data_dir: {pipeline_root}", file=sys.stderr)
        sys.exit(0)

    logger.info("NeMo Curator — embedding extraction")
    logger.info("  model            : {}", _CFG.model_id)
    logger.info("  pipeline_data_dir: {}", args.pipeline_data_dir)
    if curator_root is not None:
        logger.info("  embeddings_curator_dir: {} (output)", curator_root)
    else:
        logger.info("  embeddings_curator_dir: (unset — output under each dataset dir)")

    name_filter = None if datasets_keys is not None else args.dataset
    work_items = _collect_embedding_candidates(pipeline_root, curator_root, name_filter)
    if datasets_keys is not None:
        work_items = _apply_datasets_key_filter(work_items, datasets_keys, pipeline_root)
        if not work_items:
            sys.exit(1)

    logger.info("  datasets         : {}", [w["key"] for w in work_items])
    disk_only = [w["key"] for w in work_items if not w.get("in_config", True)]
    if disk_only:
        logger.info("  disk-only (no config.yaml entry): {}", disk_only)

    for w in work_items:
        if args.file_limit:
            w["input_files"] = w["input_files"][: args.file_limit]
        w["total_rows"] = sum(
            pq.ParquetFile(str(f)).metadata.num_rows for f in w["input_files"]
        )
        logger.info(
            "  {} : {} shards, {:,} rows → {}",
            w["key"],
            len(w["input_files"]),
            w["total_rows"],
            w["output_dir"],
        )

    if not work_items:
        logger.warning("No work items (no preprocessed parquet for selected datasets).")
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
