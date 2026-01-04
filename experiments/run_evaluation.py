#!/usr/bin/env python3
"""
Example usage of TrajectoryEvaluator for VINS-Mo evaluation
Using real data from realsense experiments
"""

from trajectory_evaluator import TrajectoryEvaluator
import numpy as np
import os

# ============================================================================
# Configuration
# ============================================================================

# Data paths
GT_PATH = "/home/linux/mxm/output_FastLio/Log/1227/traj_xy.txt"
BASELINE_PATH = "/home/linux/mxm/output_VINS-Mo/realsense/test/1227/o-o/1/vins_open_loop.tum"
OURS_PATH = "/home/linux/mxm/output_VINS-Mo/realsense/test/1227/o-d/1/"
OURS_FILE = OURS_PATH + "vins_open_loop.tum"
# Output directory
OUTPUT_DIR = OURS_PATH

# ============================================================================
# Evaluation
# ============================================================================

def main():
    print("\n" + "="*80)
    print("  VINS-Mo Trajectory Evaluation")
    print("  Dataset: RealSense 1227")
    print("="*80 + "\n")

    # Create evaluator with SE3 alignment
    # (Use Sim3 if you want to allow scale correction)
    evaluator = TrajectoryEvaluator(
        alignment_mode='se3',      # or 'sim3' for monocular
        start_clip_seconds=5.0,    # Discard first 3 seconds for initialization
        max_time_diff=0.05         # 20ms timestamp tolerance
    )

    # ========================================================================
    # Evaluate Baseline (without depth fusion)
    # ========================================================================
    if os.path.exists(BASELINE_PATH):
        print("\n" + "="*80)
        print("EVALUATING BASELINE (o-o: Open-loop without depth)")
        print("="*80)

        baseline_results = evaluator.evaluate(
            gt_path=GT_PATH,
            est_path=BASELINE_PATH,
            output_dir=OUTPUT_DIR,
            output_prefix='baseline_se3'
        )

        print(f"\nBaseline APE RMSE: {baseline_results['ape_metrics'].rmse:.6f} m")
    else:
        print(f"\n⚠️  Baseline file not found: {BASELINE_PATH}")
        baseline_results = None

    # ========================================================================
    # Evaluate Ours (with depth fusion)
    # ========================================================================
    if os.path.exists(OURS_FILE):
        print("\n" + "="*80)
        print("EVALUATING OURS (o-d: Open-loop with depth fusion)")
        print("="*80)

        ours_results = evaluator.evaluate(
            gt_path=GT_PATH,
            est_path=OURS_FILE,
            output_dir=OUTPUT_DIR,
            output_prefix='ours_se3'
        )

        print(f"\nOurs APE RMSE: {ours_results['ape_metrics'].rmse:.6f} m")
    else:
        print(f"\n⚠️  Ours file not found: {OURS_FILE}")
        ours_results = None

    # ========================================================================
    # Comparison Summary
    # ========================================================================
    if baseline_results and ours_results:
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)

        baseline_rmse = baseline_results['ape_metrics'].rmse
        ours_rmse = ours_results['ape_metrics'].rmse
        improvement = (baseline_rmse - ours_rmse) / baseline_rmse * 100

        print(f"\nAPE RMSE Comparison:")
        print(f"  Baseline: {baseline_rmse:.6f} m")
        print(f"  Ours:     {ours_rmse:.6f} m")
        print(f"  Improvement: {improvement:+.2f}%")

        if improvement > 0:
            print(f"\n✓ Depth fusion improves accuracy by {improvement:.2f}%")
        else:
            print(f"\n⚠️  Depth fusion degrades accuracy by {abs(improvement):.2f}%")

        print("\n" + "="*80 + "\n")

        # ====================================================================
        # Generate Three-Trajectory Comparison Plot
        # ====================================================================
        print("\n" + "="*80)
        print("GENERATING THREE-TRAJECTORY COMPARISON PLOT")
        print("="*80)

        evaluator.plot_three_trajectories(
            traj_gt=baseline_results['traj_gt_sync'],  # Same GT for both
            traj_baseline=baseline_results['traj_est_aligned'],
            traj_ours=ours_results['traj_est_aligned'],
            baseline_metrics=baseline_results['ape_metrics'],
            ours_metrics=ours_results['ape_metrics'],
            output_path=os.path.join(OUTPUT_DIR, 'comparison_three_trajectories.png')
        )

        # ====================================================================
        # Generate Performance Summary TXT
        # ====================================================================
        print("\n" + "="*80)
        print("GENERATING PERFORMANCE SUMMARY TXT")
        print("="*80)

        summary_file = os.path.join(OUTPUT_DIR, 'performance_summary.txt')

        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write(" " * 20 + "VINS-Mo Performance Summary\n")
            f.write(" " * 15 + "Dataset: RealSense 1227 (Open-Loop)\n")
            f.write("="*80 + "\n\n")

            # Dataset Info
            f.write("Dataset Information:\n")
            f.write("-"*80 + "\n")
            f.write(f"  GT Path:       {GT_PATH}\n")
            f.write(f"  Baseline Path: {BASELINE_PATH}\n")
            f.write(f"  Ours Path:     {OURS_FILE}\n")
            f.write(f"  Output Dir:    {OUTPUT_DIR}\n")
            f.write("\n")

            # Time Intervals
            f.write("Time Intervals:\n")
            f.write("-"*80 + "\n")

            # Load original GT (full range)
            from trajectory_evaluator import Trajectory
            import numpy as np

            # GT Full
            gt_full_data = np.loadtxt(GT_PATH)
            gt_full = Trajectory(
                timestamps=gt_full_data[:, 0],
                positions=gt_full_data[:, 1:4],
                orientations=gt_full_data[:, 4:8]
            )

            # Baseline Full (before clipping)
            baseline_full_data = np.loadtxt(BASELINE_PATH)
            baseline_full = Trajectory(
                timestamps=baseline_full_data[:, 0],
                positions=baseline_full_data[:, 1:4],
                orientations=baseline_full_data[:, 4:8]
            )

            # Ours Full (before clipping)
            ours_full_data = np.loadtxt(OURS_FILE)
            ours_full = Trajectory(
                timestamps=ours_full_data[:, 0],
                positions=ours_full_data[:, 1:4],
                orientations=ours_full_data[:, 4:8]
            )

            # GT full interval
            f.write(f"  GT Trajectory (Full Original):\n")
            f.write(f"    Range:       [{gt_full.timestamps[0]:.6f}s - {gt_full.timestamps[-1]:.6f}s]\n")
            f.write(f"    Duration:    {gt_full.timestamps[-1] - gt_full.timestamps[0]:.3f}s\n")
            f.write(f"    Poses:       {len(gt_full)}\n")
            f.write("\n")

            # Baseline full interval
            f.write(f"  Baseline Trajectory (Full Original, Before Preprocessing):\n")
            f.write(f"    Range:       [{baseline_full.timestamps[0]:.6f}s - {baseline_full.timestamps[-1]:.6f}s]\n")
            f.write(f"    Duration:    {baseline_full.timestamps[-1] - baseline_full.timestamps[0]:.3f}s\n")
            f.write(f"    Poses:       {len(baseline_full)}\n")
            f.write("\n")

            # Ours full interval
            f.write(f"  Ours Trajectory (Full Original, Before Preprocessing):\n")
            f.write(f"    Range:       [{ours_full.timestamps[0]:.6f}s - {ours_full.timestamps[-1]:.6f}s]\n")
            f.write(f"    Duration:    {ours_full.timestamps[-1] - ours_full.timestamps[0]:.3f}s\n")
            f.write(f"    Poses:       {len(ours_full)}\n")
            f.write("\n")

            # GT synchronized interval (used for evaluation)
            gt_traj = baseline_results['traj_gt_sync']
            f.write(f"  GT Trajectory (After Sync with Baseline):\n")
            f.write(f"    Range:       [{gt_traj.timestamps[0]:.6f}s - {gt_traj.timestamps[-1]:.6f}s]\n")
            f.write(f"    Duration:    {gt_traj.timestamps[-1] - gt_traj.timestamps[0]:.3f}s\n")
            f.write(f"    Poses:       {len(gt_traj)}\n")
            f.write(f"    Coverage:    {(gt_traj.timestamps[-1] - gt_traj.timestamps[0]) / (gt_full.timestamps[-1] - gt_full.timestamps[0]) * 100:.1f}% of full GT\n")
            f.write("\n")

            # Baseline after sync
            baseline_traj = baseline_results['traj_est_aligned']
            f.write(f"  Baseline Trajectory (After Sync & Alignment):\n")
            f.write(f"    Range:       [{baseline_traj.timestamps[0]:.6f}s - {baseline_traj.timestamps[-1]:.6f}s]\n")
            f.write(f"    Duration:    {baseline_traj.timestamps[-1] - baseline_traj.timestamps[0]:.3f}s\n")
            f.write(f"    Poses:       {len(baseline_traj)}\n")
            f.write(f"    Coverage:    {(baseline_traj.timestamps[-1] - baseline_traj.timestamps[0]) / (baseline_full.timestamps[-1] - baseline_full.timestamps[0]) * 100:.1f}% of original\n")
            f.write("\n")

            # Ours after sync
            ours_traj = ours_results['traj_est_aligned']
            f.write(f"  Ours Trajectory (After Sync & Alignment):\n")
            f.write(f"    Range:       [{ours_traj.timestamps[0]:.6f}s - {ours_traj.timestamps[-1]:.6f}s]\n")
            f.write(f"    Duration:    {ours_traj.timestamps[-1] - ours_traj.timestamps[0]:.3f}s\n")
            f.write(f"    Poses:       {len(ours_traj)}\n")
            f.write(f"    Coverage:    {(ours_traj.timestamps[-1] - ours_traj.timestamps[0]) / (ours_full.timestamps[-1] - ours_full.timestamps[0]) * 100:.1f}% of original\n")
            f.write("\n")

            # Common evaluation interval
            common_start_time = max(baseline_traj.timestamps[0], ours_traj.timestamps[0])
            common_end_time = min(baseline_traj.timestamps[-1], ours_traj.timestamps[-1])
            f.write(f"  Common Evaluation Interval (Baseline vs Ours):\n")
            f.write(f"    Start:       {common_start_time:.6f}s\n")
            f.write(f"    End:         {common_end_time:.6f}s\n")
            f.write(f"    Duration:    {common_end_time - common_start_time:.3f}s\n")
            f.write(f"    Coverage:    {(common_end_time - common_start_time) / (gt_full.timestamps[-1] - gt_full.timestamps[0]) * 100:.1f}% of full GT\n")
            f.write("\n")

            # Evaluation Settings
            f.write("Evaluation Settings:\n")
            f.write("-"*80 + "\n")
            f.write(f"  Alignment Mode:      SE3 (6-DOF)\n")
            f.write(f"  Start Clip:          3.0 seconds\n")
            f.write(f"  Max Time Diff:       0.02 seconds (20ms)\n")
            f.write("\n")

            # SE3 Results
            f.write("="*80 + "\n")
            f.write("SE3 ALIGNMENT RESULTS (Standard Evaluation)\n")
            f.write("="*80 + "\n\n")

            # Baseline
            baseline_ape = baseline_results['ape_metrics']
            baseline_align = baseline_results['alignment_result']

            f.write("Baseline (o-o: without depth fusion):\n")
            f.write("-"*80 + "\n")
            f.write(f"  APE RMSE:      {baseline_ape.rmse:.6f} m\n")
            f.write(f"  APE Mean:      {baseline_ape.mean:.6f} m\n")
            f.write(f"  APE Median:    {baseline_ape.median:.6f} m\n")
            f.write(f"  APE Std:       {baseline_ape.std:.6f} m\n")
            f.write(f"  APE Max:       {baseline_ape.max:.6f} m\n")
            f.write(f"  APE Min:       {baseline_ape.min:.6f} m\n")
            f.write(f"  Alignment Scale: {baseline_align.scale:.6f}\n")
            f.write(f"  Translation Norm: {np.linalg.norm(baseline_align.translation):.6f} m\n")
            f.write("\n")

            # Ours
            ours_ape = ours_results['ape_metrics']
            ours_align = ours_results['alignment_result']

            f.write("Ours (o-d: with depth fusion):\n")
            f.write("-"*80 + "\n")
            f.write(f"  APE RMSE:      {ours_ape.rmse:.6f} m\n")
            f.write(f"  APE Mean:      {ours_ape.mean:.6f} m\n")
            f.write(f"  APE Median:    {ours_ape.median:.6f} m\n")
            f.write(f"  APE Std:       {ours_ape.std:.6f} m\n")
            f.write(f"  APE Max:       {ours_ape.max:.6f} m\n")
            f.write(f"  APE Min:       {ours_ape.min:.6f} m\n")
            f.write(f"  Alignment Scale: {ours_align.scale:.6f}\n")
            f.write(f"  Translation Norm: {np.linalg.norm(ours_align.translation):.6f} m\n")
            f.write("\n")

            # Improvement
            f.write("Improvement (Baseline -> Ours):\n")
            f.write("-"*80 + "\n")
            rmse_imp = (baseline_ape.rmse - ours_ape.rmse) / baseline_ape.rmse * 100
            mean_imp = (baseline_ape.mean - ours_ape.mean) / baseline_ape.mean * 100
            median_imp = (baseline_ape.median - ours_ape.median) / baseline_ape.median * 100
            std_imp = (baseline_ape.std - ours_ape.std) / baseline_ape.std * 100
            max_imp = (baseline_ape.max - ours_ape.max) / baseline_ape.max * 100

            f.write(f"  RMSE:      {rmse_imp:+.2f}%\n")
            f.write(f"  Mean:      {mean_imp:+.2f}%\n")
            f.write(f"  Median:    {median_imp:+.2f}%\n")
            f.write(f"  Std:       {std_imp:+.2f}%\n")
            f.write(f"  Max:       {max_imp:+.2f}%\n")
            f.write("\n")

            if rmse_imp > 0:
                f.write(f"✓ Depth fusion IMPROVES accuracy by {rmse_imp:.2f}%\n")
            else:
                f.write(f"⚠️  Depth fusion DEGRADES accuracy by {abs(rmse_imp):.2f}%\n")
            f.write("\n")

        print(f"✓ Performance summary saved: {summary_file}")

        # If Sim3 results available, append to summary
        if 'baseline_sim3' in locals() and 'ours_sim3' in locals():
            with open(summary_file, 'a') as f:
                f.write("="*80 + "\n")
                f.write("SIM3 ALIGNMENT RESULTS (With Scale Correction)\n")
                f.write("="*80 + "\n\n")

                # Baseline Sim3
                baseline_sim3_ape = baseline_sim3['ape_metrics']
                baseline_sim3_align = baseline_sim3['alignment_result']

                f.write("Baseline (Sim3):\n")
                f.write("-"*80 + "\n")
                f.write(f"  APE RMSE:        {baseline_sim3_ape.rmse:.6f} m\n")
                f.write(f"  APE Mean:        {baseline_sim3_ape.mean:.6f} m\n")
                f.write(f"  Estimated Scale: {baseline_sim3_align.scale:.6f}\n")
                scale_drift_baseline = abs(1.0 - baseline_sim3_align.scale) * 100
                f.write(f"  Scale Drift:     {scale_drift_baseline:.2f}%\n")
                if scale_drift_baseline > 10:
                    f.write(f"  Status:          ⚠️  Significant scale drift\n")
                else:
                    f.write(f"  Status:          ✓ Scale stable\n")
                f.write("\n")

                # Ours Sim3
                ours_sim3_ape = ours_sim3['ape_metrics']
                ours_sim3_align = ours_sim3['alignment_result']

                f.write("Ours (Sim3):\n")
                f.write("-"*80 + "\n")
                f.write(f"  APE RMSE:        {ours_sim3_ape.rmse:.6f} m\n")
                f.write(f"  APE Mean:        {ours_sim3_ape.mean:.6f} m\n")
                f.write(f"  Estimated Scale: {ours_sim3_align.scale:.6f}\n")
                scale_drift_ours = abs(1.0 - ours_sim3_align.scale) * 100
                f.write(f"  Scale Drift:     {scale_drift_ours:.2f}%\n")
                if scale_drift_ours > 10:
                    f.write(f"  Status:          ⚠️  Significant scale drift\n")
                else:
                    f.write(f"  Status:          ✓ Scale stable\n")
                f.write("\n")

                # Scale Comparison
                f.write("Scale Stability Comparison:\n")
                f.write("-"*80 + "\n")
                f.write(f"  Baseline Scale Drift: {scale_drift_baseline:.2f}%\n")
                f.write(f"  Ours Scale Drift:     {scale_drift_ours:.2f}%\n")
                scale_improvement = scale_drift_baseline - scale_drift_ours
                f.write(f"  Improvement:          {scale_improvement:+.2f}%\n")
                f.write("\n")

                # SE3 vs Sim3 comparison
                f.write("SE3 vs Sim3 RMSE Improvement:\n")
                f.write("-"*80 + "\n")
                baseline_improvement_ratio = baseline_ape.rmse / baseline_sim3_ape.rmse
                ours_improvement_ratio = ours_ape.rmse / ours_sim3_ape.rmse
                f.write(f"  Baseline: {baseline_improvement_ratio:.2f}x\n")
                f.write(f"  Ours:     {ours_improvement_ratio:.2f}x\n")

                if baseline_improvement_ratio > 2.0:
                    f.write(f"  Baseline: ⚠️  Significant scale drift detected!\n")
                else:
                    f.write(f"  Baseline: ✓ Scale stable\n")

                if ours_improvement_ratio > 2.0:
                    f.write(f"  Ours:     ⚠️  Significant scale drift detected!\n")
                else:
                    f.write(f"  Ours:     ✓ Scale stable\n")
                f.write("\n")

                f.write("="*80 + "\n")
                f.write("Generated by VINS-Mo Trajectory Evaluator\n")
                f.write("="*80 + "\n")

        print(f"\n✓ Performance summary updated with Sim3 results")


    # ========================================================================
    # Optional: Sim3 Evaluation (with scale correction)
    # ========================================================================
    print("\n" + "="*80)
    print("EVALUATING WITH Sim3 (Scale Correction Enabled)")
    print("="*80)

    evaluator_sim3 = TrajectoryEvaluator(
        alignment_mode='sim3',
        start_clip_seconds=3.0,
        max_time_diff=0.02
    )

    if os.path.exists(BASELINE_PATH):
        print("\n[Baseline with Sim3]")
        baseline_sim3 = evaluator_sim3.evaluate(
            gt_path=GT_PATH,
            est_path=BASELINE_PATH,
            output_dir=OUTPUT_DIR,
            output_prefix='baseline_sim3'
        )

        # Check scale drift
        scale = baseline_sim3['alignment_result'].scale
        scale_drift = abs(1.0 - scale) * 100

        print(f"\nScale Analysis:")
        print(f"  Estimated scale: {scale:.6f}")
        print(f"  Scale drift: {scale_drift:.2f}%")

        if scale_drift > 10:
            print(f"  ⚠️  Significant scale drift detected!")
        else:
            print(f"  ✓ Scale is stable")

    if os.path.exists(OURS_FILE):
        print("\n[Ours with Sim3]")
        ours_sim3 = evaluator_sim3.evaluate(
            gt_path=GT_PATH,
            est_path=OURS_FILE,
            output_dir=OUTPUT_DIR,
            output_prefix='ours_sim3'
        )

        scale = ours_sim3['alignment_result'].scale
        scale_drift = abs(1.0 - scale) * 100

        print(f"\nScale Analysis:")
        print(f"  Estimated scale: {scale:.6f}")
        print(f"  Scale drift: {scale_drift:.2f}%")

        if scale_drift > 10:
            print(f"  ⚠️  Significant scale drift detected!")
        else:
            print(f"  ✓ Scale is stable")

    print("\n" + "="*80)
    print("✓ All evaluations completed!")
    print(f"  Results saved to: {OUTPUT_DIR}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
