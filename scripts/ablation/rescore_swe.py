"""Re-grade SWE-bench cells whose patches scored 0 due to the old Modal
cgroup-v2 sandbox bug.

The agents already produced valid patches in ``<cell>/results.jsonl``; only
the grading step was broken. This script reruns ``_run_harness`` for each
row whose ``score.details.patch`` is non-empty and rewrites the row with
the new score. Empty patches are preserved (correctly score 0). Updated
``summary.json`` has ``accuracy = #(score==1) / n_done`` recomputed.

Usage:
    .venv/bin/python scripts/ablation/rescore_swe.py \\
        --cells cloud-only-haiku45-swe-n100,cloud-only-gpt5mini-swe-n100

    .venv/bin/python scripts/ablation/rescore_swe.py --all-swe

Idempotent: each cell writes ``_rescored_ids.txt`` listing task_ids that
have been successfully rescored; reruns skip those.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

# Make `openjarvis` importable when running this script directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from openjarvis.evals.scorers.swebench_harness import _run_harness  # noqa: E402

HYBRID_DIR = Path(os.path.expanduser("~/.openjarvis/experiments/hybrid"))
RUNS_DIR = HYBRID_DIR / "runs"
DOCS_TABLE = HYBRID_DIR / "docs" / "results-table.md"

MAX_WORKERS = 8
RETRY_ATTEMPTS = 3
TIMEOUT_S = 1800
PROGRESS_EVERY = 10


# ---------- IO helpers ----------

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _atomic_write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def _patch_of(row: Dict[str, Any]) -> Optional[str]:
    sc = row.get("score")
    if not isinstance(sc, dict):
        return None
    det = sc.get("details") or {}
    p = det.get("patch")
    if isinstance(p, str) and p.strip():
        return p
    return None


def _is_correct(row: Dict[str, Any]) -> bool:
    sc = row.get("score") or {}
    if not isinstance(sc, dict):
        return False
    return float(sc.get("score", 0) or 0) >= 1.0


# ---------- Re-score one row with retry ----------

def _rescore_row(
    task_id: str,
    patch: str,
    err_log: Path,
    err_log_lock: Lock,
) -> Optional[Dict[str, Any]]:
    """Call ``_run_harness`` with up to RETRY_ATTEMPTS retries.

    Returns the new score dict ``{"success": bool, "score": float,
    "details": {...}}`` on success. Returns ``None`` if all attempts
    failed; the caller leaves the row untouched and logs the error.
    """
    last_err = ""
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            r = _run_harness(task_id, patch, TIMEOUT_S)
            return {
                "success": bool(r.get("success", False)),
                "score": float(r.get("score", 0.0)),
                "details": r.get("details", {}),
            }
        except Exception as exc:  # noqa: BLE001
            last_err = f"attempt={attempt} {type(exc).__name__}: {exc}"
            time.sleep(min(2 ** attempt, 30))
    with err_log_lock:
        with err_log.open("a") as f:
            f.write(f"{task_id}\t{last_err}\n")
    return None


# ---------- Per-cell processing ----------

def _process_cell(cell_dir: Path) -> Dict[str, Any]:
    name = cell_dir.name
    results_path = cell_dir / "results.jsonl"
    if not results_path.exists():
        print(f"[SKIP] {name} — no results.jsonl")
        return {"cell": name, "skipped": True}

    rows = _read_jsonl(results_path)
    n_total = len(rows)

    # Pre-load idempotency tracker.
    tracker_path = cell_dir / "_rescored_ids.txt"
    already_done: set[str] = set()
    if tracker_path.exists():
        already_done = {
            line.strip() for line in tracker_path.read_text().splitlines()
            if line.strip()
        }

    # Old accuracy from current rows.
    old_resolved = sum(1 for r in rows if _is_correct(r))
    old_acc = old_resolved / n_total if n_total else 0.0

    # Build worklist of (idx, task_id, patch).
    worklist: List[Tuple[int, str, str]] = []
    for i, r in enumerate(rows):
        task_id = r.get("task_id") or ""
        patch = _patch_of(r)
        if not patch:
            continue
        if task_id in already_done:
            continue
        worklist.append((i, task_id, patch))

    print(
        f"[{name}] start: rows={n_total} with_patch={sum(1 for r in rows if _patch_of(r))} "
        f"already_rescored={len(already_done)} todo={len(worklist)} old_acc={old_acc:.3f}"
    )

    err_log = cell_dir / "_rescore_errors.log"
    err_log_lock = Lock()
    tracker_lock = Lock()
    rows_lock = Lock()
    n_done = 0
    n_new_resolved = 0
    n_failed = 0

    def _flush_tracker(task_id: str) -> None:
        with tracker_lock:
            with tracker_path.open("a") as f:
                f.write(task_id + "\n")
            already_done.add(task_id)

    # Periodic snapshotting: every PROGRESS_EVERY rescored rows, atomic-write
    # the updated results.jsonl so partial progress is durable.
    def _snapshot() -> None:
        with rows_lock:
            _atomic_write_jsonl(results_path, rows)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        fut_to_meta = {
            pool.submit(_rescore_row, tid, patch, err_log, err_log_lock):
                (i, tid)
            for (i, tid, patch) in worklist
        }
        for fut in as_completed(fut_to_meta):
            i, tid = fut_to_meta[fut]
            new_score = fut.result()
            if new_score is None:
                n_failed += 1
                n_done += 1
            else:
                with rows_lock:
                    old_score = rows[i].get("score") or {}
                    new_reason = ((new_score.get("details") or {}).get("reason")
                                  if isinstance(new_score.get("details"), dict) else None)
                    if new_reason == "no_report":
                        pass
                    else:
                        rows[i]["score"] = new_score
                _flush_tracker(tid)
                if new_score["success"]:
                    n_new_resolved += 1
                n_done += 1

            if n_done % PROGRESS_EVERY == 0:
                _snapshot()
                print(
                    f"[{name}] rescored={n_done}/{len(worklist)} "
                    f"new_resolved_so_far={n_new_resolved} failed={n_failed}"
                )

    # Final snapshot.
    _snapshot()

    # Rebuild summary.json.
    summary_path = cell_dir / "summary.json"
    summary: Dict[str, Any] = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except Exception:
            summary = {}
    new_resolved_total = sum(1 for r in rows if _is_correct(r))
    n_done_summary = summary.get("n_done", n_total)
    new_acc = new_resolved_total / n_done_summary if n_done_summary else 0.0
    summary["accuracy"] = new_acc
    summary_path.write_text(json.dumps(summary, indent=2))

    err_share = 0.0
    if worklist:
        err_share = n_failed / len(worklist)

    print(
        f"[DONE] {name} acc_old={old_acc:.3f} -> acc_new={new_acc:.3f} "
        f"(resolved: {old_resolved} -> {new_resolved_total}) "
        f"rescore_errors={n_failed}/{len(worklist)} ({err_share*100:.1f}%)"
    )
    return {
        "cell": name,
        "n_total": n_total,
        "old_acc": old_acc,
        "new_acc": new_acc,
        "old_resolved": old_resolved,
        "new_resolved": new_resolved_total,
        "rescored": len(worklist),
        "failed": n_failed,
        "cost_usd_total": summary.get("cost_usd_total", 0.0),
        "wall_time_s": summary.get("wall_time_s", 0.0),
        "tokens_local_total": summary.get("tokens_local_total", 0),
        "tokens_cloud_total": summary.get("tokens_cloud_total", 0),
        "n_done": n_done_summary,
        "n_target": summary.get("n_target", n_total),
        "bench": summary.get("bench", "swebench-verified"),
    }


# ---------- results-table.md updater ----------

def _format_table_row(s: Dict[str, Any]) -> str:
    bench = "SWE-bench"
    return (
        f"| `{s['cell']}` | {bench} | "
        f"{s['new_acc']:.3f} · ${s['cost_usd_total']:.2f} | "
        f"{int(s['wall_time_s'])}s | tools=— | "
        f"tokens_local={int(s['tokens_local_total'])} | "
        f"tokens_cloud={int(s['tokens_cloud_total'])} | "
        f"{s['n_done']}/{s['n_target']} |"
    )


def _update_results_table(summaries: List[Dict[str, Any]]) -> bool:
    if not DOCS_TABLE.exists():
        print(f"[WARN] {DOCS_TABLE} missing — skipping table update")
        return False
    text = DOCS_TABLE.read_text()
    lines = text.splitlines()
    updated = 0
    for s in summaries:
        if s.get("skipped"):
            continue
        cell = s["cell"]
        needle = f"| `{cell}` |"
        new_row = _format_table_row(s)
        replaced = False
        for i, line in enumerate(lines):
            if line.startswith(needle):
                lines[i] = new_row
                replaced = True
                updated += 1
                break
        if not replaced:
            # Append to end of file if section row missing.
            lines.append(new_row)
            updated += 1
    DOCS_TABLE.write_text("\n".join(lines) + "\n")
    print(f"[TABLE] updated {updated} rows in {DOCS_TABLE}")
    return True


# ---------- CLI ----------

def _resolve_cells(args: argparse.Namespace) -> List[Path]:
    if args.all_swe:
        cells = sorted(p for p in RUNS_DIR.glob("*-swe-n100") if p.is_dir())
        return [c for c in cells if (c / "results.jsonl").exists()]
    if not args.cells:
        raise SystemExit("Provide --cells or --all-swe")
    out: List[Path] = []
    for name in args.cells.split(","):
        name = name.strip()
        if not name:
            continue
        p = RUNS_DIR / name
        if not p.exists():
            print(f"[WARN] cell not found: {p}")
            continue
        out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=str, default="",
                    help="Comma-separated cell names under ~/.openjarvis/experiments/hybrid/runs/")
    ap.add_argument("--all-swe", action="store_true",
                    help="Auto-detect cells matching *-swe-n100")
    args = ap.parse_args()

    cells = _resolve_cells(args)
    if not cells:
        print("No cells to process")
        return 1
    print(f"Processing {len(cells)} cell(s): {[c.name for c in cells]}")

    summaries: List[Dict[str, Any]] = []
    t0 = time.time()
    for cell in cells:
        try:
            summaries.append(_process_cell(cell))
        except KeyboardInterrupt:
            print(f"[INTERRUPT] aborting on {cell.name}")
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] {cell.name}: {type(exc).__name__}: {exc}")
            summaries.append({"cell": cell.name, "error": str(exc)})
    elapsed = time.time() - t0

    # Final summary table.
    print("\n=== FINAL SUMMARY ===")
    print(f"{'cell':<48} {'old_acc':>8} {'new_acc':>8} {'resolved':>14} {'failed':>8}")
    total_old = 0
    total_new = 0
    for s in summaries:
        if "error" in s or s.get("skipped"):
            print(f"{s['cell']:<48} {'-':>8} {'-':>8} {'-':>14} -")
            continue
        print(
            f"{s['cell']:<48} {s['old_acc']:>8.3f} {s['new_acc']:>8.3f} "
            f"{s['old_resolved']:>5} -> {s['new_resolved']:<5} {s['failed']:>8}"
        )
        total_old += s["old_resolved"]
        total_new += s["new_resolved"]
    print(f"\nTotal resolved: {total_old} -> {total_new}")
    print(f"Wall time: {elapsed/60:.1f} min")

    _update_results_table(summaries)
    return 0


if __name__ == "__main__":
    sys.exit(main())
