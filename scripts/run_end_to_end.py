#!/usr/bin/env python3
"""End-to-end pipeline for histo-omics-lite benchmark.

This script orchestrates the complete pipeline:
1. Fetch/prepare dataset
2. Train models (with optional pretrain steps)
3. Evaluate models with bootstrap CI
4. Generate visualizations and reports
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from histo_omics_lite.data.adapters import prepare_public_demo_data
from histo_omics_lite.report import generate_static_report
from histo_omics_lite.tracking import setup_mlflow_tracking


def run_command(cmd: list[str], description: str, timeout: int = 300) -> dict[str, Any]:
    """Run a command and return result information."""
    print(f"\n🔄 {description}")
    print(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
        
        elapsed = time.time() - start_time
        print(f"✅ Completed in {elapsed:.1f}s")
        
        return {
            "success": True,
            "elapsed": elapsed,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"❌ Failed after {elapsed:.1f}s")
        print(f"Error: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        
        return {
            "success": False,
            "elapsed": elapsed,
            "error": str(e),
            "stdout": e.stdout,
            "stderr": e.stderr,
        }
    
    except subprocess.TimeoutExpired as e:
        elapsed = time.time() - start_time
        print(f"⏰ Timeout after {elapsed:.1f}s")
        
        return {
            "success": False,
            "elapsed": elapsed,
            "error": f"Timeout after {timeout}s",
            "stdout": "",
            "stderr": "",
        }


def main() -> None:
    """Main end-to-end pipeline."""
    parser = argparse.ArgumentParser(
        description="Run end-to-end histo-omics-lite benchmark"
    )
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "public_demo"],
        default="synthetic",
        help="Dataset to use (default: synthetic)",
    )
    parser.add_argument(
        "--config",
        default="fast_debug",
        help="Training config to use (default: fast_debug)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory for results (default: results)",
    )
    parser.add_argument(
        "--skip-pretrain",
        action="store_true",
        help="Skip SimCLR pretraining step",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=["clip", "image_linear_probe", "omics_mlp", "early_fusion", "late_fusion"],
        help="Models to evaluate",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples for public_demo dataset",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run in smoke test mode (faster, minimal evaluation)",
    )
    
    args = parser.parse_args()
    
    # Setup output directories
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.output_dir / "data"
    artifacts_dir = args.output_dir / "artifacts"
    reports_dir = args.output_dir / "reports"
    
    # Pipeline tracking
    pipeline_start = time.time()
    results = {
        "pipeline_config": vars(args),
        "steps": {},
        "summary": {},
    }
    
    # Setup MLflow tracking
    print("🔧 Setting up experiment tracking...")
    mlflow_logger = setup_mlflow_tracking(
        experiment_name="histo-omics-lite-benchmark",
        tracking_uri=f"file://{args.output_dir}/mlruns",
    )
    
    try:
        with mlflow_logger:
            # Log pipeline parameters
            mlflow_logger.log_params({"pipeline": vars(args)})
            
            # Step 1: Prepare dataset
            if args.dataset == "public_demo":
                print(f"\n📊 Step 1: Preparing {args.dataset} dataset...")
                dataset_path = prepare_public_demo_data(
                    output_dir=data_dir / "public_demo",
                    n_samples=args.n_samples,
                    tile_size=64,
                )
                results["steps"]["dataset_prep"] = {"success": True, "path": str(dataset_path)}
            else:
                # Use existing synthetic data generation
                dataset_cmd = ["python", "-m", "histo_omics_lite.data.synthetic", str(data_dir / "synthetic")]
                results["steps"]["dataset_prep"] = run_command(
                    dataset_cmd, f"Preparing {args.dataset} dataset"
                )
                dataset_path = data_dir / "synthetic"
            
            # Step 2: Train models
            training_results = {}
            
            for model_name in args.models:
                print(f"\n🧠 Step 2.{len(training_results)+1}: Training {model_name}...")
                
                # Prepare training command
                train_cmd = [
                    "python", "-m", "histo_omics_lite.training.train",
                    f"--config={args.config}",
                    f"--data.root={dataset_path}",
                    f"--model={model_name}",
                    f"--output-dir={artifacts_dir / model_name}",
                ]
                
                if args.smoke_test:
                    train_cmd.extend([
                        "--trainer.max_epochs=1",
                        "--trainer.limit_train_batches=2",
                        "--trainer.limit_val_batches=2",
                    ])
                
                training_results[model_name] = run_command(
                    train_cmd, f"Training {model_name}", timeout=600
                )
            
            results["steps"]["training"] = training_results
            
            # Step 3: Evaluate models
            print(f"\n📈 Step 3: Evaluating models...")
            
            evaluation_results = {}
            for model_name in args.models:
                if training_results[model_name]["success"]:
                    eval_cmd = [
                        "python", "-m", "histo_omics_lite.evaluation.evaluate",
                        f"--model-path={artifacts_dir / model_name / 'model.pt'}",
                        f"--data-path={dataset_path}",
                        f"--output-dir={artifacts_dir / model_name / 'evaluation'}",
                        "--bootstrap-ci",
                    ]
                    
                    evaluation_results[model_name] = run_command(
                        eval_cmd, f"Evaluating {model_name}"
                    )
            
            results["steps"]["evaluation"] = evaluation_results
            
            # Step 4: Generate visualizations
            print(f"\n🎨 Step 4: Generating visualizations...")
            
            vis_cmd = [
                "python", "-m", "histo_omics_lite.visualization.generate_plots",
                f"--artifacts-dir={artifacts_dir}",
                f"--output-dir={reports_dir / 'plots'}",
            ]
            
            results["steps"]["visualization"] = run_command(
                vis_cmd, "Generating visualizations"
            )
            
            # Step 5: Generate report
            print(f"\n📋 Step 5: Generating report...")
            
            try:
                report_path = generate_static_report(
                    results_dir=args.output_dir,
                    output_path=reports_dir / "index.html",
                )
                
                results["steps"]["report"] = {
                    "success": True,
                    "path": str(report_path),
                }
                
                # Log report as artifact
                mlflow_logger.log_artifact(report_path, "report.html")
                
            except Exception as e:
                results["steps"]["report"] = {
                    "success": False,
                    "error": str(e),
                }
            
            # Step 6: Run smoke tests (if enabled)
            if args.smoke_test:
                print(f"\n🧪 Step 6: Running smoke tests...")
                
                smoke_cmd = [
                    "python", "-m", "pytest", 
                    "tests/test_smoke.py",
                    f"--artifacts-dir={artifacts_dir}",
                    "-v",
                ]
                
                results["steps"]["smoke_tests"] = run_command(
                    smoke_cmd, "Running smoke tests"
                )
            
            # Summary
            pipeline_elapsed = time.time() - pipeline_start
            results["summary"] = {
                "total_elapsed": pipeline_elapsed,
                "success": all(
                    step.get("success", False) if isinstance(step, dict) else
                    all(substep.get("success", False) for substep in step.values())
                    for step in results["steps"].values()
                ),
            }
            
            # Log final metrics
            mlflow_logger.log_metrics({
                "pipeline_duration_minutes": pipeline_elapsed / 60,
                "pipeline_success": 1.0 if results["summary"]["success"] else 0.0,
            })
            
            # Save results
            results_file = args.output_dir / "pipeline_results.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2, default=str)
            
            mlflow_logger.log_artifact(results_file, "pipeline_results.json")
            
            # Print summary
            print(f"\n{'='*60}")
            print(f"🎉 Pipeline completed in {pipeline_elapsed/60:.1f} minutes")
            print(f"📊 Results saved to: {args.output_dir}")
            
            if results["summary"]["success"]:
                print(f"✅ All steps completed successfully!")
                if "report" in results["steps"] and results["steps"]["report"]["success"]:
                    print(f"📋 Report: {results['steps']['report']['path']}")
                
                run_info = mlflow_logger.get_run_info()
                print(f"🔬 MLflow Run: {run_info['run_id']}")
                
            else:
                print(f"❌ Some steps failed. Check results for details.")
                failed_steps = [
                    step_name for step_name, step_result in results["steps"].items()
                    if not (step_result.get("success", False) if isinstance(step_result, dict) else
                           all(substep.get("success", False) for substep in step_result.values()))
                ]
                print(f"Failed steps: {', '.join(failed_steps)}")
                sys.exit(1)
    
    except Exception as e:
        print(f"\n💥 Pipeline failed with error: {e}")
        results["error"] = str(e)
        results["summary"]["success"] = False
        
        # Save error results
        results_file = args.output_dir / "pipeline_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        sys.exit(1)


if __name__ == "__main__":
    main()