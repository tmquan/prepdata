"""
04_embedding_reducer.py – Reduce NeMo Curator embeddings to 2D/3D for visualisation.

Follows the exact same CompositeStage pattern as EmbeddingCreatorStage:

    EmbeddingReducerStage  (CompositeStage – user-facing, @dataclass)
        └── EmbeddingReductionModelStage  (ProcessingStage – GPU worker)

GPU path  (RAPIDS)  : cuML  TSNE + UMAP  run on the GPU tensor directly
CPU fallback        : sklearn TSNE  +  umap-learn  (same API, much slower)

⚠ t-SNE and UMAP are global operations — they cannot be applied per-mini-batch.
  The pipeline is constructed with files_per_partition = len(input_files) so that
  all embeddings arrive in a single DocumentBatch, mirroring how EmbeddingCreatorStage
  processes each text partition independently.

Output layout
-------------
<embeddings_dir>/<dataset_key>/
    part-*.parquet          ← embeddings (from 02_embedding_extractor.py)
    metadata.json
    reduction/              ← written by this script
        reduced_2d_3d.parquet
        metadata.json

Output columns
--------------
    row_id, dataset_key, category,
    tsne_2d_x, tsne_2d_y,  tsne_3d_x, tsne_3d_y, tsne_3d_z,
    umap_2d_x, umap_2d_y,  umap_3d_x, umap_3d_y, umap_3d_z

Resume-safe: datasets whose reduction/ already contains reduced_2d_3d.parquet
are skipped.

Usage
-----
    python 04_embedding_reducer.py
    python 04_embedding_reducer.py --dataset vietnamese-legal-documents
    python 04_embedding_reducer.py --max_points 200000
    python 04_embedding_reducer.py --dry_run
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.text.io.reader import ParquetReader
from nemo_curator.stages.text.io.writer import ParquetWriter
from nemo_curator.tasks import DocumentBatch

# ─── Config ───────────────────────────────────────────────────────────────────
_CFG: DictConfig = OmegaConf.load(Path(__file__).parent / "config.yaml")

_REDUCTION_SUBDIR = "reduction"
# Single output shard — reduction always operates on the full dataset in one
# batch, so there is exactly one output file.  Partition naming keeps it
# consistent with the embedding shards produced by 02_embedding_extractor.py.
_OUTPUT_FILE = "part-00000-of-00001.parquet"
_OUTPUT_COLS = [
    "tsne_2d_x", "tsne_2d_y",
    "tsne_3d_x", "tsne_3d_y", "tsne_3d_z",
    "umap_2d_x", "umap_2d_y",
    "umap_3d_x", "umap_3d_y", "umap_3d_z",
]

# Detect RAPIDS once at module load so ray_stage_spec() can request GPU resources.
try:
    import cupy as cp
    import cudf
    from cuml.manifold import TSNE as cuTSNE
    from cuml.manifold.umap import UMAP as cuUMAP
    _USE_GPU = True
    logger.info("RAPIDS available – EmbeddingReductionModelStage will use cuML (GPU)")
except ImportError:
    _USE_GPU = False
    logger.warning("RAPIDS not available – EmbeddingReductionModelStage will use sklearn (CPU)")


# ─────────────────────────────────────────────────────────────────────────────
# Inner worker stage  (mirrors EmbeddingModelStage)
# ─────────────────────────────────────────────────────────────────────────────
class EmbeddingReductionModelStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """GPU (cuML) or CPU (sklearn) dimensionality reduction.

    Mirrors EmbeddingModelStage: it is not used directly but is composed
    inside EmbeddingReducerStage.decompose(), just as EmbeddingModelStage
    is composed inside EmbeddingCreatorStage.decompose().

    ⚠ Must receive the ENTIRE dataset as one DocumentBatch.
    """

    def __init__(
        self,
        embeddings_field: str = "embeddings",
        tsne_perplexity: int = 30,
        umap_n_neighbors: int = 15,
        umap_min_dist: float = 0.1,
        random_state: int = 42,
    ) -> None:
        self.embeddings_field = embeddings_field
        self.tsne_perplexity  = tsne_perplexity
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist    = umap_min_dist
        self.random_state     = random_state
        self.name = "embedding_reduction_model"

    # ── NeMo Curator stage interface ──────────────────────────────────────────
    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.embeddings_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], _OUTPUT_COLS

    def setup(self, worker_metadata=None) -> None:
        """Detect RAPIDS at worker startup (mirrors EmbeddingModelStage.setup)."""
        self._gpu = _USE_GPU
        logger.info(
            "EmbeddingReductionModelStage setup – backend: {}",
            "cuML (GPU)" if self._gpu else "sklearn (CPU)",
        )

    def ray_stage_spec(self) -> dict[str, Any]:
        """Request 1 GPU per worker when cuML is available."""
        return {"is_actor_stage": True, "num_gpus": 1 if _USE_GPU else 0}

    # ── Core reduction ────────────────────────────────────────────────────────
    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df   = batch.to_pandas()
        n    = len(df)
        perf = min(self.tsne_perplexity, max(1, (n - 1) // 3))
        neigh = min(self.umap_n_neighbors, n - 1) if n > 1 else 2

        logger.info(
            "  Reducing {:,} × {} – t-SNE perp={} UMAP neigh={} [{}]",
            n, len(df[self.embeddings_field].iloc[0]),
            perf, neigh,
            "GPU" if self._gpu else "CPU",
        )

        coords = (
            self._reduce_gpu(df, perf, neigh)
            if self._gpu
            else self._reduce_cpu(df, perf, neigh)
        )

        for col, vals in coords.items():
            df[col] = vals
        df = df.drop(columns=[self.embeddings_field])

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )

    # ── GPU backend (cuML) ────────────────────────────────────────────────────
    def _reduce_gpu(self, df: pd.DataFrame, perplexity: int, n_neighbors: int) -> dict:
        flat = cp.asarray(df[self.embeddings_field].list.leaves
                          if hasattr(df[self.embeddings_field], "list")
                          else np.stack(df[self.embeddings_field].tolist()))
        X = flat.reshape(len(df), -1).astype(cp.float32)

        steps = [
            ("t-SNE 2D", "tsne_2d", cuTSNE,
             dict(n_components=2, perplexity=float(perplexity), random_state=self.random_state,
                  n_iter=1000, method="fft", learning_rate_method="adaptive")),
            ("t-SNE 3D", "tsne_3d", cuTSNE,
             dict(n_components=3, perplexity=float(perplexity), random_state=self.random_state,
                  n_iter=1000, method="fft", learning_rate_method="adaptive")),
            ("UMAP 2D",  "umap_2d", cuUMAP,
             dict(n_components=2, n_neighbors=n_neighbors, min_dist=self.umap_min_dist,
                  random_state=self.random_state, metric="cosine")),
            ("UMAP 3D",  "umap_3d", cuUMAP,
             dict(n_components=3, n_neighbors=n_neighbors, min_dist=self.umap_min_dist,
                  random_state=self.random_state, metric="cosine")),
        ]
        return self._run_steps(X, steps, to_numpy=lambda t: cp.asnumpy(cp.asarray(t)))

    # ── CPU backend (sklearn + umap-learn) ────────────────────────────────────
    def _reduce_cpu(self, df: pd.DataFrame, perplexity: int, n_neighbors: int) -> dict:
        from sklearn.manifold import TSNE
        import umap as umap_lib

        X = np.stack(df[self.embeddings_field].tolist()).astype(np.float32)

        steps = [
            ("t-SNE 2D", "tsne_2d", TSNE,
             dict(n_components=2, perplexity=perplexity, random_state=self.random_state,
                  max_iter=1000, init="pca")),
            ("t-SNE 3D", "tsne_3d", TSNE,
             dict(n_components=3, perplexity=perplexity, random_state=self.random_state,
                  max_iter=1000, init="pca")),
            ("UMAP 2D",  "umap_2d", umap_lib.UMAP,
             dict(n_components=2, n_neighbors=n_neighbors, min_dist=self.umap_min_dist,
                  random_state=self.random_state, metric="cosine")),
            ("UMAP 3D",  "umap_3d", umap_lib.UMAP,
             dict(n_components=3, n_neighbors=n_neighbors, min_dist=self.umap_min_dist,
                  random_state=self.random_state, metric="cosine")),
        ]
        return self._run_steps(X, steps, to_numpy=lambda t: t.astype(np.float32))

    # ── Shared step runner ────────────────────────────────────────────────────
    def _run_steps(self, X, steps: list, to_numpy) -> dict:
        coord_cols: dict[str, list] = {}
        col_names_map = {
            "tsne_2d": ["tsne_2d_x", "tsne_2d_y"],
            "tsne_3d": ["tsne_3d_x", "tsne_3d_y", "tsne_3d_z"],
            "umap_2d": ["umap_2d_x", "umap_2d_y"],
            "umap_3d": ["umap_3d_x", "umap_3d_y", "umap_3d_z"],
        }
        with tqdm(total=len(steps), desc="  Reduction", unit="step") as pbar:
            for label, key, Cls, kwargs in steps:
                pbar.set_postfix_str(label)
                t0    = time.perf_counter()
                result = Cls(**kwargs).fit_transform(X)
                arr    = to_numpy(result)
                for i, col in enumerate(col_names_map[key]):
                    coord_cols[col] = arr[:, i].tolist()
                del result
                if _USE_GPU:
                    cp.get_default_memory_pool().free_all_blocks()
                gc.collect()
                pbar.update(1)
                logger.debug("    {} {:.1f}s", label, time.perf_counter() - t0)
        return coord_cols


# ─────────────────────────────────────────────────────────────────────────────
# Composite stage  (mirrors EmbeddingCreatorStage)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(kw_only=True)
class EmbeddingReducerStage(CompositeStage[DocumentBatch, DocumentBatch]):
    """Composite stage that reduces embeddings to 2D/3D coordinates.

    Mirrors EmbeddingCreatorStage: a @dataclass CompositeStage that wraps
    an inner model stage and exposes clean hyperparameter fields.

    Usage inside a Pipeline::

        Pipeline(
            name="embedding_reduction_pipeline",
            stages=[
                ParquetReader(
                    file_paths=input_files,
                    files_per_partition=len(input_files),  # ← all data in one batch
                    fields=["embeddings"],
                ),
                EmbeddingReducerStage(
                    embeddings_field="embeddings",
                    tsne_perplexity=30,
                    umap_n_neighbors=15,
                ),
                ParquetWriter(path=output_dir, fields=_OUTPUT_COLS),
            ],
        )
    """

    embeddings_field: str  = "embeddings"
    tsne_perplexity:  int  = 30
    umap_n_neighbors: int  = 15
    umap_min_dist:    float = 0.1
    random_state:     int  = 42

    def __post_init__(self) -> None:
        super().__init__()
        # Single sub-stage — analogous to EmbeddingCreatorStage wrapping
        # TokenizerStage + EmbeddingModelStage
        self.stages = [
            EmbeddingReductionModelStage(
                embeddings_field=self.embeddings_field,
                tsne_perplexity=self.tsne_perplexity,
                umap_n_neighbors=self.umap_n_neighbors,
                umap_min_dist=self.umap_min_dist,
                random_state=self.random_state,
            )
        ]

    def decompose(self) -> list[ProcessingStage]:
        return self.stages


# ─────────────────────────────────────────────────────────────────────────────
# Discovery
# ─────────────────────────────────────────────────────────────────────────────
def _discover_datasets(embeddings_dir: str, dataset_filter: str | None) -> list[dict]:
    root = Path(embeddings_dir)
    if not root.is_dir():
        logger.error("Embeddings directory not found: {}", embeddings_dir)
        return []

    items = []
    for ds_dir in sorted(root.iterdir()):
        if not ds_dir.is_dir():
            continue
        key = ds_dir.name
        if dataset_filter and dataset_filter != key:
            continue

        parquet_files = sorted(
            p for p in ds_dir.glob("part-*.parquet") if not p.name.startswith(".tmp")
        )
        if not parquet_files:
            continue

        total_rows = sum(pq.ParquetFile(str(p)).metadata.num_rows for p in parquet_files)
        ds_cfg     = _CFG.datasets.get(key, {})

        items.append({
            "key":           key,
            "category":      ds_cfg.get("category", "unknown") if ds_cfg else "unknown",
            "parquet_files": [str(p) for p in parquet_files],
            "total_rows":    total_rows,
            "reduction_dir": ds_dir / _REDUCTION_SUBDIR,
            "output_file":   ds_dir / _REDUCTION_SUBDIR / _OUTPUT_FILE,
        })
        logger.info("  {}: {} shard(s), {:,} rows", key, len(parquet_files), total_rows)

    return items


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reduce NeMo Curator embeddings to 2D/3D via EmbeddingReducerStage"
    )
    p.add_argument("--dataset",          default=None,                 help="Single dataset key (default: all)")
    p.add_argument("--embeddings_dir",   default=_CFG.embeddings_dir,  help="Embeddings root directory")
    p.add_argument("--max_points",       type=int,   default=0,        help="Subsample cap per dataset (0 = no cap)")
    p.add_argument("--tsne_perplexity",  type=int,   default=30)
    p.add_argument("--umap_neighbors",   type=int,   default=15)
    p.add_argument("--umap_min_dist",    type=float, default=0.1)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--no_csv",           action="store_true",          help="Skip writing CSV copy")
    p.add_argument("--dry_run",          action="store_true",          help="Show plan without processing")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    max_rows = args.max_points if args.max_points > 0 else None

    logger.info("backend        : {}", "cuML (GPU)" if _USE_GPU else "sklearn (CPU)")
    logger.info("embeddings_dir : {}", args.embeddings_dir)
    logger.info("max_points     : {}", max_rows or "no cap")

    items = _discover_datasets(args.embeddings_dir, args.dataset)
    if not items:
        logger.warning("No datasets found.")
        sys.exit(0)

    logger.info("Found {} dataset(s), {:,} total rows",
                len(items), sum(it["total_rows"] for it in items))

    if args.dry_run:
        for it in items:
            use = min(it["total_rows"], args.max_points) if args.max_points else it["total_rows"]
            logger.info("  {} : {:,} → reduce {:,}", it["key"], it["total_rows"], use)
        sys.exit(0)

    t_wall = time.perf_counter()

    for idx, it in enumerate(items, 1):
        key      = it["key"]
        out_file = it["output_file"]

        logger.info("─" * 60)
        logger.info("[{}/{}] {}  ({:,} rows)", idx, len(items), key, it["total_rows"])

        if out_file.exists():
            logger.info("  Already exists – skipping")
            continue

        # Optionally subsample: write a temporary single-shard parquet so the
        # ParquetReader can load it all in one partition.
        input_files = it["parquet_files"]
        tmp_shard   = None

        if max_rows and it["total_rows"] > max_rows:
            frames = [pd.read_parquet(p, columns=["embeddings"]) for p in input_files]
            sample = pd.concat(frames, ignore_index=True).sample(n=max_rows, random_state=args.seed)
            tmp_shard = str(it["reduction_dir"] / ".tmp_sample.parquet")
            it["reduction_dir"].mkdir(parents=True, exist_ok=True)
            sample.to_parquet(tmp_shard, index=False)
            input_files = [tmp_shard]
            logger.info("  Subsampled to {:,} rows", max_rows)

        it["reduction_dir"].mkdir(parents=True, exist_ok=True)

        # ── Build + run the NeMo Curator Pipeline ─────────────────────────────
        # files_per_partition = len(input_files) ensures ALL embeddings arrive
        # as a single DocumentBatch, which is required for t-SNE / UMAP.
        t0 = time.perf_counter()
        try:
            pipeline = Pipeline(
                name="embedding_reduction_pipeline",
                stages=[
                    ParquetReader(
                        file_paths=input_files,
                        files_per_partition=len(input_files),
                        fields=["embeddings"],
                    ),
                    EmbeddingReducerStage(
                        embeddings_field="embeddings",
                        tsne_perplexity=args.tsne_perplexity,
                        umap_n_neighbors=args.umap_neighbors,
                        umap_min_dist=args.umap_min_dist,
                        random_state=args.seed,
                    ),
                    ParquetWriter(
                        path=str(it["reduction_dir"]),
                        fields=_OUTPUT_COLS,
                    ),
                ],
            )
            pipeline.run()
        except Exception as exc:
            logger.error("  Pipeline failed for {}: {}", key, exc)
            if tmp_shard and os.path.exists(tmp_shard):
                os.remove(tmp_shard)
            continue

        if tmp_shard and os.path.exists(tmp_shard):
            os.remove(tmp_shard)

        elapsed = time.perf_counter() - t0
        logger.info("  Done in {:.1f}s ({:.1f} min) → {}", elapsed, elapsed / 60, out_file)

        # ParquetWriter produces a hash-named file; rename to the canonical
        # partition name so downstream tools can find it with part-*.parquet.
        written = [
            p for p in it["reduction_dir"].glob("*.parquet")
            if not p.name.startswith(".tmp") and p != out_file
        ]
        if written:
            written[0].rename(out_file)

        if not args.no_csv:
            pd.read_parquet(str(out_file)).to_csv(
                str(out_file.with_name("part-00000-of-00001.csv")), index=False
            )

        (it["reduction_dir"] / "metadata.json").write_text(json.dumps({
            "dataset_key":     key,
            "backend":         "cuML (GPU)" if _USE_GPU else "sklearn (CPU)",
            "num_rows":        it["total_rows"] if not max_rows else min(it["total_rows"], max_rows),
            "total_available": it["total_rows"],
            "tsne_perplexity": args.tsne_perplexity,
            "umap_neighbors":  args.umap_neighbors,
            "umap_min_dist":   args.umap_min_dist,
            "seed":            args.seed,
            "elapsed_s":       round(elapsed, 1),
        }, indent=2), encoding="utf-8")

    total = time.perf_counter() - t_wall
    logger.info("=" * 60)
    logger.info("All done in {:.1f}s ({:.1f} min)", total, total / 60)


if __name__ == "__main__":
    main()
