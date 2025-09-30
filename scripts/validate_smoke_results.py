#!/usr/bin/env python3
"""Validate smoke test results against expected thresholds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def validate_smoke_results(results_dir: Path) -> bool:
    """Validate smoke test results.
    
    Args:
        results_dir: Directory containing smoke test results
        
    Returns:
        True if all validations pass, False otherwise
    """
    print(f"🔍 Validating smoke test results in {results_dir}")
    
    # Check pipeline results exist
    pipeline_results_path = results_dir / "pipeline_results.json"
    if not pipeline_results_path.exists():
        print(f"❌ Pipeline results not found: {pipeline_results_path}")
        return False
    
    # Load pipeline results
    with open(pipeline_results_path) as f:
        pipeline_results = json.load(f)
    
    success = True
    
    # Check pipeline completed successfully
    if not pipeline_results.get("summary", {}).get("success", False):
        print("❌ Pipeline did not complete successfully")
        success = False
    else:
        print("✅ Pipeline completed successfully")
    
    # Check pipeline duration (should be under 10 minutes for smoke test)
    duration = pipeline_results.get("summary", {}).get("total_elapsed", 0)
    if duration > 600:  # 10 minutes
        print(f"⚠️  Pipeline took {duration/60:.1f} minutes (longer than expected 10 minutes)")
    else:
        print(f"✅ Pipeline completed in {duration/60:.1f} minutes")
    
    # Validate individual model results
    artifacts_dir = results_dir / "artifacts"
    if artifacts_dir.exists():
        model_dirs = [d for d in artifacts_dir.iterdir() if d.is_dir()]
        
        if len(model_dirs) == 0:
            print("❌ No model results found")
            success = False
        else:
            print(f"📊 Found {len(model_dirs)} model results")
            
            for model_dir in model_dirs:
                model_success = validate_model_results(model_dir)
                if not model_success:
                    success = False
    else:
        print("❌ Artifacts directory not found")
        success = False
    
    # Check report generation
    report_path = results_dir / "reports" / "index.html"
    if report_path.exists():
        print("✅ Report generated successfully")
    else:
        print("❌ Report not generated")
        success = False
    
    return success


def validate_model_results(model_dir: Path) -> bool:
    """Validate results for a single model."""
    model_name = model_dir.name
    print(f"  📋 Validating {model_name}...")
    
    success = True
    
    # Check metrics file exists
    metrics_path = model_dir / "evaluation" / "metrics.json"
    if not metrics_path.exists():
        print(f"    ❌ Metrics file not found for {model_name}")
        return False
    
    # Load metrics
    with open(metrics_path) as f:
        metrics = json.load(f)
    
    # Validate AUROC > 0.6 (lenient threshold for synthetic data)
    if "auroc" in metrics:
        auroc_data = metrics["auroc"]
        if isinstance(auroc_data, dict):
            auroc = auroc_data.get("point_estimate", 0)
        else:
            auroc = auroc_data
        
        if auroc > 0.6:
            print(f"    ✅ AUROC: {auroc:.3f} (> 0.6)")
        else:
            print(f"    ❌ AUROC: {auroc:.3f} (≤ 0.6)")
            success = False
    else:
        print(f"    ❌ AUROC metric not found for {model_name}")
        success = False
    
    # Validate calibration ECE < 0.2
    if "ece" in metrics:
        ece = metrics["ece"]
        if ece < 0.2:
            print(f"    ✅ ECE: {ece:.3f} (< 0.2)")
        else:
            print(f"    ⚠️  ECE: {ece:.3f} (≥ 0.2)")
            # Don't fail on ECE for synthetic data
    
    # Check for confidence intervals
    if "auroc" in metrics and isinstance(metrics["auroc"], dict):
        if "lower_ci" in metrics["auroc"] and "upper_ci" in metrics["auroc"]:
            print(f"    ✅ Confidence intervals computed")
        else:
            print(f"    ❌ Confidence intervals missing")
            success = False
    
    return success


def main() -> None:
    """Main validation function."""
    parser = argparse.ArgumentParser(
        description="Validate smoke test results"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing smoke test results",
    )
    
    args = parser.parse_args()
    
    if not args.results_dir.exists():
        print(f"❌ Results directory does not exist: {args.results_dir}")
        sys.exit(1)
    
    success = validate_smoke_results(args.results_dir)
    
    if success:
        print("\n✅ All smoke test validations passed!")
        sys.exit(0)
    else:
        print("\n❌ Some smoke test validations failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()