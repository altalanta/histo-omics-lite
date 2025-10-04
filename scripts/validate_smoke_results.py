#!/usr/bin/env python3
"""Validate smoke test results for the histo-omics-lite pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def validate_smoke_results(results_dir: Path) -> bool:
    """Validate smoke test outputs against expected thresholds."""
    results_dir = results_dir.resolve()
    print(f"🔍 Validating smoke test outputs in {results_dir}")

    pipeline_summary = results_dir / "smoke_eval.json"
    if not pipeline_summary.exists():
        print(f"❌ Expected evaluation summary at {pipeline_summary}")
        return False

    summary_payload = _load_json(pipeline_summary)
    metrics = summary_payload.get("metrics", {})
    class_metrics = metrics.get("classification", {})
    calibration_metrics = metrics.get("calibration", {})

    auroc = class_metrics.get("auroc") or class_metrics.get("auroc_ovr")
    ece = calibration_metrics.get("ece")

    success = True
    if auroc is None:
        print("❌ AUROC metric missing from evaluation payload")
        success = False
    elif auroc < 0.6:
        print(f"❌ AUROC {auroc:.3f} is below expected smoke threshold (0.6)")
        success = False
    else:
        print(f"✅ AUROC {auroc:.3f} meets smoke threshold")

    if ece is None:
        print("⚠️  Calibration ECE missing; skipping threshold check")
    elif ece >= 0.2:
        print(f"⚠️  ECE {ece:.3f} exceeds 0.2 synthetic tolerance")
    else:
        print(f"✅ ECE {ece:.3f} within synthetic tolerance")

    runtime_info = summary_payload.get("runtime")
    if isinstance(runtime_info, dict):
        elapsed = runtime_info.get("seconds", 0)
        if elapsed and elapsed > 600:
            print(f"⚠️  Pipeline runtime {elapsed:.1f}s exceeds 10 minute smoke budget")
        else:
            print(f"✅ Pipeline runtime {elapsed:.1f}s within budget")

    embed_summary = results_dir / "smoke_embed.json"
    if embed_summary.exists():
        embed_payload = _load_json(embed_summary)
        embeddings_path = embed_payload.get("output_path")
        if embeddings_path:
            if Path(embeddings_path).exists():
                print(f"✅ Embeddings artifact located at {embeddings_path}")
            else:
                print(f"❌ Embeddings artifact missing at {embeddings_path}")
                success = False
        else:
            print("⚠️  Embedding summary missing output_path field")

    return success


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate smoke test results")
    parser.add_argument("--results-dir", type=Path, required=True, help="Directory containing smoke outputs")
    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"❌ {args.results_dir} does not exist")
        sys.exit(1)

    if validate_smoke_results(args.results_dir):
        print("\n✅ Smoke test validations passed")
        sys.exit(0)

    print("\n❌ Smoke test validations failed")
    sys.exit(1)


if __name__ == "__main__":
    main()
