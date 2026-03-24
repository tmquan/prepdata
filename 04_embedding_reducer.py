"""
04_embedding_reducer.py — NeMo Curator pipeline: embeddings → 2D/3D coordinates.

Pipeline: ParquetReader → EmbeddingReductionModelStage → ParquetWriter.

GPU (RAPIDS cuML) when available; else sklearn + umap-learn.

Global methods (t-SNE, UMAP) require the full dataset in one partition:
``files_per_partition = len(input_files)``.

Layout
------
<pipeline_data_dir>/<dataset_key>/
    embedding/part-*.parquet   — input
    reduction/
        part-00000-of-00001.parquet  — output coordinates
        metadata.json

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
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import ParquetReader
from nemo_curator.stages.text.io.writer import ParquetWriter
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch

_CFG: DictConfig = OmegaConf.load(Path(__file__).parent / "config.yaml")


def _layout_subdir(name: str, default: str) -> str:
    v = OmegaConf.select(_CFG, f"curator_layout.{name}")
    return str(v) if v is not None else default


def _embedding_dir(pipeline_root: str | Path, dataset_key: str) -> Path:
    return Path(pipeline_root) / dataset_key / _layout_subdir("embeddings", "embeddings")


def _reduction_dir(pipeline_root: str | Path, dataset_key: str) -> Path:
    return Path(pipeline_root) / dataset_key / _layout_subdir("reductions", "reductions")


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


_OUTPUT_FILE = "part-00000-of-00001.parquet"
_OUTPUT_COLS = [
    "tsne_2d_x", "tsne_2d_y",
    "tsne_3d_x", "tsne_3d_y", "tsne_3d_z",
    "umap_2d_x", "umap_2d_y",
    "umap_3d_x", "umap_3d_y", "umap_3d_z",
]

try:
    import cupy as cp
    from cuml.manifold import TSNE as cuTSNE
    from cuml.manifold.umap import UMAP as cuUMAP
    _USE_GPU = True
    logger.info("RAPIDS available — cuML (GPU) reduction")
except ImportError:
    _USE_GPU = False
    logger.warning("RAPIDS unavailable — sklearn (CPU) reduction")


class EmbeddingReductionModelStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Dimensionality reduction on a single DocumentBatch (full embedding matrix)."""

    def __init__(
        self,
        embeddings_field: str = "embeddings",
        tsne_perplexity: int = 30,
        umap_n_neighbors: int = 15,
        umap_min_dist: float = 0.1,
        random_state: int = 42,
    ) -> None:
        self.embeddings_field = embeddings_field
        self.tsne_perplexity = tsne_perplexity
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist
        self.random_state = random_state
        self.name = "embedding_reduction_model"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.embeddings_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], _OUTPUT_COLS

    def setup(self, worker_metadata=None) -> None:
        self._gpu = _USE_GPU
        logger.info(
            "EmbeddingReductionModelStage setup — {}",
            "cuML (GPU)" if self._gpu else "sklearn (CPU)",
        )

    def ray_stage_spec(self) -> dict[str, Any]:
        return {"is_actor_stage": True, "num_gpus": 1 if _USE_GPU else 0}

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()
        n = len(df)
        perf = min(self.tsne_perplexity, max(1, (n - 1) // 3))
        neigh = min(self.umap_n_neighbors, n - 1) if n > 1 else 2
        logger.info(
            "  Reducing {:,} × {} — t-SNE perp={} UMAP k={} [{}]",
            n,
            len(df[self.embeddings_field].iloc[0]),
            perf,
            neigh,
            "GPU" if self._gpu else "CPU",
        )
        coords = self._reduce_gpu(df, perf, neigh) if self._gpu else self._reduce_cpu(df, perf, neigh)
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

    def _reduce_gpu(self, df: pd.DataFrame, perplexity: int, n_neighbors: int) -> dict:
        X = cp.asarray(np.stack(df[self.embeddings_field].tolist()).astype(np.float32))
        steps = [
            ("t-SNE 2D", "tsne_2d", cuTSNE,
             dict(n_components=2, perplexity=float(perplexity), random_state=self.random_state,
                  n_iter=1000, method="fft", learning_rate_method="adaptive")),
            ("t-SNE 3D", "tsne_3d", cuTSNE,
             dict(n_components=3, perplexity=float(perplexity), random_state=self.random_state,
                  n_iter=1000, method="fft", learning_rate_method="adaptive")),
            ("UMAP 2D", "umap_2d", cuUMAP,
             dict(n_components=2, n_neighbors=n_neighbors, min_dist=self.umap_min_dist,
                  random_state=self.random_state, metric="cosine")),
            ("UMAP 3D", "umap_3d", cuUMAP,
             dict(n_components=3, n_neighbors=n_neighbors, min_dist=self.umap_min_dist,
                  random_state=self.random_state, metric="cosine")),
        ]
        return self._run_steps(X, steps, to_numpy=lambda t: cp.asnumpy(cp.asarray(t)))

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
            ("UMAP 2D", "umap_2d", umap_lib.UMAP,
             dict(n_components=2, n_neighbors=n_neighbors, min_dist=self.umap_min_dist,
                  random_state=self.random_state, metric="cosine")),
            ("UMAP 3D", "umap_3d", umap_lib.UMAP,
             dict(n_components=3, n_neighbors=n_neighbors, min_dist=self.umap_min_dist,
                  random_state=self.random_state, metric="cosine")),
        ]
        return self._run_steps(X, steps, to_numpy=lambda t: t.astype(np.float32))

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
                t0 = time.perf_counter()
                result = Cls(**kwargs).fit_transform(X)
                arr = to_numpy(result)
                for i, col in enumerate(col_names_map[key]):
                    coord_cols[col] = arr[:, i].tolist()
                del result
                if _USE_GPU:
                    cp.get_default_memory_pool().free_all_blocks()
                gc.collect()
                pbar.update(1)
                logger.debug("    {} {:.1f}s", label, time.perf_counter() - t0)
        return coord_cols


def _discover_datasets(pipeline_root: str, dataset_filter: str | None) -> list[dict]:
    root = Path(pipeline_root)
    if not root.is_dir():
        logger.error("pipeline_data_dir not found: {}", pipeline_root)
        return []

    items = []
    for ds_path in sorted(root.iterdir()):
        if not ds_path.is_dir():
            continue
        key = ds_path.name
        if dataset_filter and dataset_filter != key:
            continue
        emb = _embedding_dir(root, key)
        if not emb.is_dir():
            continue
        parquet_files = _list_parquet_shards(emb)
        if not parquet_files:
            continue
        total_rows = sum(pq.ParquetFile(str(p)).metadata.num_rows for p in parquet_files)
        red = _reduction_dir(root, key)
        items.append({
            "key": key,
            "parquet_files": [str(p) for p in parquet_files],
            "total_rows": total_rows,
            "reduction_dir": red,
            "output_file": red / _OUTPUT_FILE,
        })
        logger.info("  {}: {} embedding shard(s), {:,} rows", key, len(parquet_files), total_rows)
    return items


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reduce embeddings to 2D/3D (NeMo Curator pipeline)")
    p.add_argument("--dataset", default=None)
    p.add_argument("--pipeline_data_dir", default=_CFG.pipeline_data_dir)
    p.add_argument("--max_points", type=int, default=0)
    p.add_argument("--tsne_perplexity", type=int, default=30)
    p.add_argument("--umap_neighbors", type=int, default=15)
    p.add_argument("--umap_min_dist", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_csv", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    max_rows = args.max_points if args.max_points > 0 else None

    logger.info("backend            : {}", "cuML (GPU)" if _USE_GPU else "sklearn (CPU)")
    logger.info("pipeline_data_dir  : {}", args.pipeline_data_dir)
    logger.info("max_points         : {}", max_rows or "no cap")

    items = _discover_datasets(args.pipeline_data_dir, args.dataset)
    if not items:
        logger.warning("No datasets with embedding/ parquet found.")
        sys.exit(0)

    logger.info("Found {} dataset(s), {:,} rows", len(items), sum(i["total_rows"] for i in items))

    if args.dry_run:
        for it in items:
            use = min(it["total_rows"], args.max_points) if args.max_points else it["total_rows"]
            logger.info("  {} : {:,} → {:,}", it["key"], it["total_rows"], use)
        sys.exit(0)

    t_wall = time.perf_counter()
    for idx, it in enumerate(items, 1):
        key, out_file = it["key"], it["output_file"]
        logger.info("─" * 60)
        logger.info("[{}/{}] {}  ({:,} rows)", idx, len(items), key, it["total_rows"])
        if out_file.exists():
            logger.info("  {} exists — skip", out_file.name)
            continue

        input_files = it["parquet_files"]
        tmp_shard = None
        if max_rows and it["total_rows"] > max_rows:
            frames = [pd.read_parquet(p, columns=["embeddings"]) for p in input_files]
            sample = pd.concat(frames, ignore_index=True).sample(n=max_rows, random_state=args.seed)
            it["reduction_dir"].mkdir(parents=True, exist_ok=True)
            tmp_shard = str(it["reduction_dir"] / ".tmp_sample.parquet")
            sample.to_parquet(tmp_shard, index=False)
            input_files = [tmp_shard]
            logger.info("  Subsampled to {:,} rows", max_rows)

        it["reduction_dir"].mkdir(parents=True, exist_ok=True)
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
                    EmbeddingReductionModelStage(
                        embeddings_field="embeddings",
                        tsne_perplexity=args.tsne_perplexity,
                        umap_n_neighbors=args.umap_neighbors,
                        umap_min_dist=args.umap_min_dist,
                        random_state=args.seed,
                    ),
                    ParquetWriter(path=str(it["reduction_dir"]), fields=_OUTPUT_COLS),
                ],
            )
            pipeline.run()
        except Exception as exc:
            logger.error("  Pipeline failed: {}", exc)
            if tmp_shard and os.path.exists(tmp_shard):
                os.remove(tmp_shard)
            continue

        if tmp_shard and os.path.exists(tmp_shard):
            os.remove(tmp_shard)

        elapsed = time.perf_counter() - t0
        logger.info("  Done in {:.1f}s → {}", elapsed, out_file)

        written = [
            p for p in it["reduction_dir"].glob("*.parquet")
            if not p.name.startswith(".tmp") and p.resolve() != out_file.resolve()
        ]
        if written:
            written[0].rename(out_file)

        if not args.no_csv:
            pd.read_parquet(str(out_file)).to_csv(
                str(out_file.with_name("part-00000-of-00001.csv")), index=False
            )

        (it["reduction_dir"] / "metadata.json").write_text(
            json.dumps({
                "dataset_key": key,
                "backend": "cuML (GPU)" if _USE_GPU else "sklearn (CPU)",
                "num_rows": it["total_rows"] if not max_rows else min(it["total_rows"], max_rows),
                "total_available": it["total_rows"],
                "tsne_perplexity": args.tsne_perplexity,
                "umap_neighbors": args.umap_neighbors,
                "umap_min_dist": args.umap_min_dist,
                "seed": args.seed,
                "elapsed_s": round(elapsed, 1),
            }, indent=2),
            encoding="utf-8",
        )

    logger.info("=" * 60)
    logger.info("All done in {:.1f}s ({:.1f} min)", time.perf_counter() - t_wall, (time.perf_counter() - t_wall) / 60)


if __name__ == "__main__":
    main()
