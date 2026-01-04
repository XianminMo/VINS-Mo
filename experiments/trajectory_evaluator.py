#!/usr/bin/env python3
"""
Academic Publication Grade Trajectory Evaluator for VINS/SLAM Systems

Features:
- TUM format trajectory loading with automatic time synchronization
- Start clipping to handle initialization drift
- Umeyama alignment (SE3 and Sim3)
- APE metrics calculation
- High-quality publication-ready visualization

Author: Generated for VINS-Mo Evaluation
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize_scalar
from dataclasses import dataclass
from typing import Tuple, Optional
import warnings

# ============================================================================
# Configuration for Publication-Quality Plots
# ============================================================================

PLOT_CONFIG = {
    "font.family": 'serif',
    "font.serif": ['Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'Times'],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "axes.labelweight": 'bold',
    "legend.fontsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.format": 'png',
    "savefig.bbox": 'tight',
    "mathtext.fontset": 'stix',
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": ':',
    "grid.linewidth": 0.8,
}

# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Trajectory:
    """Trajectory data structure"""
    timestamps: np.ndarray  # (N,)
    positions: np.ndarray   # (N, 3) - x, y, z
    orientations: np.ndarray  # (N, 4) - qw, qx, qy, qz

    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, idx):
        """Support slicing"""
        return Trajectory(
            timestamps=self.timestamps[idx],
            positions=self.positions[idx],
            orientations=self.orientations[idx]
        )


@dataclass
class AlignmentResult:
    """Result of trajectory alignment"""
    rotation: np.ndarray      # 3x3 rotation matrix
    translation: np.ndarray   # 3x1 translation vector
    scale: float              # scale factor
    rmse: float              # RMSE after alignment


@dataclass
class APEMetrics:
    """Absolute Pose Error metrics"""
    rmse: float
    mean: float
    median: float
    std: float
    min: float
    max: float
    errors: np.ndarray  # per-frame errors


# ============================================================================
# Core Trajectory Evaluator Class
# ============================================================================

class TrajectoryEvaluator:
    """
    Academic-grade trajectory evaluator for VINS/SLAM systems.

    Parameters:
        alignment_mode: 'se3' (6-DOF) or 'sim3' (7-DOF with scale)
        start_clip_seconds: Discard first N seconds of estimate (for initialization)
        max_time_diff: Maximum timestamp difference for association (seconds)
    """

    def __init__(
        self,
        alignment_mode: str = 'se3',
        start_clip_seconds: float = 0.0,
        max_time_diff: float = 0.02
    ):
        self.alignment_mode = alignment_mode.lower()
        self.start_clip_seconds = start_clip_seconds
        self.max_time_diff = max_time_diff

        if self.alignment_mode not in ['se3', 'sim3']:
            raise ValueError("alignment_mode must be 'se3' or 'sim3'")

        # Apply plot configuration
        rcParams.update(PLOT_CONFIG)

    # ========================================================================
    # File I/O
    # ========================================================================

    def load_tum_trajectory(self, filepath: str) -> Trajectory:
        """
        Load trajectory from TUM format file.

        Format: timestamp x y z qx qy qz qw

        Returns:
            Trajectory object with orientations as (qw, qx, qy, qz)
        """
        data = np.loadtxt(filepath)

        if data.shape[1] < 8:
            raise ValueError(f"Expected 8 columns (timestamp x y z qx qy qz qw), got {data.shape[1]}")

        timestamps = data[:, 0]
        positions = data[:, 1:4]  # x, y, z
        # Convert from (qx, qy, qz, qw) to (qw, qx, qy, qz)
        orientations = np.column_stack([data[:, 7], data[:, 4:7]])

        return Trajectory(timestamps, positions, orientations)

    # ========================================================================
    # Data Preprocessing
    # ========================================================================

    def clip_start(self, traj: Trajectory) -> Trajectory:
        """Remove first N seconds from trajectory"""
        if self.start_clip_seconds <= 0:
            return traj

        start_time = traj.timestamps[0] + self.start_clip_seconds
        mask = traj.timestamps >= start_time

        clipped_count = len(traj) - np.sum(mask)
        print(f"  Clipped {clipped_count} poses ({self.start_clip_seconds:.1f}s) from start")

        return traj[mask]

    def synchronize_trajectories(
        self,
        traj_ref: Trajectory,
        traj_est: Trajectory
    ) -> Tuple[Trajectory, Trajectory]:
        """
        Synchronize two trajectories by timestamp association.

        Uses nearest-neighbor matching within max_time_diff threshold.
        """
        timestamps_ref = traj_ref.timestamps
        timestamps_est = traj_est.timestamps

        # Print time intervals BEFORE synchronization
        print(f"\n  Time Intervals (Before Synchronization):")
        print(f"    GT:  [{timestamps_ref[0]:.6f}s - {timestamps_ref[-1]:.6f}s]  Duration: {timestamps_ref[-1] - timestamps_ref[0]:.3f}s")
        print(f"    EST: [{timestamps_est[0]:.6f}s - {timestamps_est[-1]:.6f}s]  Duration: {timestamps_est[-1] - timestamps_est[0]:.3f}s")

        # Find best time offset
        def compute_overlap(offset: float) -> float:
            """Compute number of matched timestamps with given offset"""
            adjusted_est = timestamps_est + offset
            matches = 0
            for t_est in adjusted_est:
                diffs = np.abs(timestamps_ref - t_est)
                if np.min(diffs) < self.max_time_diff:
                    matches += 1
            return -matches  # Negative for minimization

        # Search for best offset in reasonable range
        result = minimize_scalar(
            compute_overlap,
            bounds=(-10.0, 10.0),
            method='bounded'
        )
        best_offset = result.x

        print(f"  Time offset: {best_offset:.6f}s")

        # Apply offset and match timestamps
        timestamps_est_adjusted = timestamps_est + best_offset

        matched_indices_ref = []
        matched_indices_est = []

        for i, t_est in enumerate(timestamps_est_adjusted):
            diffs = np.abs(timestamps_ref - t_est)
            min_idx = np.argmin(diffs)

            if diffs[min_idx] < self.max_time_diff:
                matched_indices_ref.append(min_idx)
                matched_indices_est.append(i)

        matched_indices_ref = np.array(matched_indices_ref)
        matched_indices_est = np.array(matched_indices_est)

        sync_ratio = len(matched_indices_est) / len(traj_est) * 100
        print(f"  Synchronized: {len(matched_indices_est)}/{len(traj_est)} poses ({sync_ratio:.1f}%)")

        if sync_ratio < 30:
            warnings.warn(
                f"Low synchronization ratio ({sync_ratio:.1f}%). "
                "Check timestamp alignment!"
            )

        # Get synchronized trajectories
        traj_ref_sync = traj_ref[matched_indices_ref]
        traj_est_sync = traj_est[matched_indices_est]

        # Print time intervals AFTER synchronization (common interval)
        print(f"\n  Common Evaluation Interval:")
        print(f"    Start: {traj_ref_sync.timestamps[0]:.6f}s")
        print(f"    End:   {traj_ref_sync.timestamps[-1]:.6f}s")
        print(f"    Duration: {traj_ref_sync.timestamps[-1] - traj_ref_sync.timestamps[0]:.3f}s")
        print(f"    Matched poses: {len(traj_ref_sync)}")

        return traj_ref_sync, traj_est_sync

    # ========================================================================
    # Umeyama Alignment
    # ========================================================================

    def umeyama_alignment(
        self,
        positions_src: np.ndarray,
        positions_dst: np.ndarray,
        with_scale: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Umeyama algorithm for point cloud alignment.

        Finds R, t, s such that: dst = s * R @ src + t

        Args:
            positions_src: (N, 3) source positions
            positions_dst: (N, 3) destination positions
            with_scale: If True, estimate scale (Sim3); else scale=1 (SE3)

        Returns:
            R: 3x3 rotation matrix
            t: 3x1 translation vector
            s: scale factor
        """
        assert positions_src.shape == positions_dst.shape
        assert positions_src.shape[1] == 3

        # Center the point clouds
        centroid_src = np.mean(positions_src, axis=0)
        centroid_dst = np.mean(positions_dst, axis=0)

        src_centered = positions_src - centroid_src
        dst_centered = positions_dst - centroid_dst

        # Compute scale
        if with_scale:
            src_scale = np.sqrt(np.mean(np.sum(src_centered**2, axis=1)))
            dst_scale = np.sqrt(np.mean(np.sum(dst_centered**2, axis=1)))
            scale = dst_scale / src_scale
        else:
            scale = 1.0

        # Compute rotation via SVD
        H = src_centered.T @ dst_centered
        U, S, Vt = np.linalg.svd(H)

        rotation = Vt.T @ U.T

        # Ensure proper rotation (det = 1)
        if np.linalg.det(rotation) < 0:
            Vt[-1, :] *= -1
            rotation = Vt.T @ U.T

        # Compute translation
        translation = centroid_dst - scale * rotation @ centroid_src

        return rotation, translation, scale

    def align_trajectory(
        self,
        traj_ref: Trajectory,
        traj_est: Trajectory
    ) -> Tuple[Trajectory, AlignmentResult]:
        """
        Align estimated trajectory to reference using Umeyama.

        Returns:
            aligned_traj: Aligned estimated trajectory
            alignment_result: Alignment parameters and metrics
        """
        with_scale = (self.alignment_mode == 'sim3')

        # Compute alignment
        R_align, t_align, s_align = self.umeyama_alignment(
            traj_est.positions,
            traj_ref.positions,
            with_scale=with_scale
        )

        # Apply transformation
        positions_aligned = s_align * (traj_est.positions @ R_align.T) + t_align

        # Compute RMSE after alignment
        errors = np.linalg.norm(positions_aligned - traj_ref.positions, axis=1)
        rmse = np.sqrt(np.mean(errors**2))

        aligned_traj = Trajectory(
            timestamps=traj_est.timestamps.copy(),
            positions=positions_aligned,
            orientations=traj_est.orientations.copy()
        )

        alignment_result = AlignmentResult(
            rotation=R_align,
            translation=t_align,
            scale=s_align,
            rmse=rmse
        )

        print(f"\n  [{self.alignment_mode.upper()}] Alignment:")
        print(f"    Rotation: det(R) = {np.linalg.det(R_align):.6f}")
        print(f"    Translation: {np.linalg.norm(t_align):.6f} m")
        print(f"    Scale: {s_align:.6f}")
        print(f"    RMSE: {rmse:.6f} m")

        return aligned_traj, alignment_result

    # ========================================================================
    # Metrics Calculation
    # ========================================================================

    def calculate_ape(
        self,
        traj_ref: Trajectory,
        traj_est: Trajectory
    ) -> APEMetrics:
        """Calculate Absolute Pose Error metrics"""
        errors = np.linalg.norm(
            traj_est.positions - traj_ref.positions,
            axis=1
        )

        return APEMetrics(
            rmse=np.sqrt(np.mean(errors**2)),
            mean=np.mean(errors),
            median=np.median(errors),
            std=np.std(errors),
            min=np.min(errors),
            max=np.max(errors),
            errors=errors
        )

    # ========================================================================
    # Visualization
    # ========================================================================

    def plot_trajectory_comparison(
        self,
        traj_ref: Trajectory,
        traj_est: Trajectory,
        ape_metrics: APEMetrics,
        output_path: str = 'trajectory_comparison.png'
    ):
        """
        Create publication-quality trajectory comparison plot.

        Features:
        - Top-down XY view with equal aspect ratio
        - GT as gray dashed line
        - Estimate colored by APE error magnitude
        - Start/End markers
        - Error colorbar
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        # Ground Truth trajectory
        ax.plot(
            traj_ref.positions[:, 0],
            traj_ref.positions[:, 1],
            color='#666666',
            linestyle='--',
            linewidth=3,
            alpha=0.7,
            label='Ground Truth',
            zorder=1
        )

        # Estimated trajectory with error coloring
        errors_normalized = ape_metrics.errors / ape_metrics.max

        # Create colored line segments
        points = traj_est.positions[:, :2].reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        from matplotlib.collections import LineCollection
        lc = LineCollection(
            segments,
            cmap='hot_r',
            linewidths=3,
            alpha=0.9,
            zorder=2
        )
        lc.set_array(errors_normalized[:-1])
        line = ax.add_collection(lc)

        # Colorbar
        cbar = plt.colorbar(line, ax=ax, pad=0.02)
        cbar.set_label('APE Error [m]', fontsize=14, weight='bold')
        # Update colorbar ticks to show actual error values
        cbar_ticks = cbar.get_ticks()
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels([f'{t * ape_metrics.max:.3f}' for t in cbar_ticks])

        # Start marker
        ax.scatter(
            traj_ref.positions[0, 0],
            traj_ref.positions[0, 1],
            color='green',
            marker='o',
            s=300,
            edgecolors='white',
            linewidths=3,
            label='Start',
            zorder=5
        )

        # End markers
        ax.scatter(
            traj_ref.positions[-1, 0],
            traj_ref.positions[-1, 1],
            color='red',
            marker='s',
            s=300,
            edgecolors='white',
            linewidths=3,
            label='End (GT)',
            zorder=5
        )

        ax.scatter(
            traj_est.positions[-1, 0],
            traj_est.positions[-1, 1],
            color='darkred',
            marker='s',
            s=300,
            edgecolors='white',
            linewidths=3,
            label='End (Est)',
            zorder=5
        )

        # Styling
        ax.set_xlabel('X [m]', fontsize=14, weight='bold')
        ax.set_ylabel('Y [m]', fontsize=14, weight='bold')
        ax.set_title(
            f'Trajectory Comparison (APE RMSE: {ape_metrics.rmse:.4f} m)',
            fontsize=16,
            weight='bold',
            pad=20
        )
        ax.legend(loc='best', fontsize=12, framealpha=0.95)
        ax.axis('equal')
        ax.grid(True, linestyle=':', alpha=0.4, linewidth=0.8)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Trajectory plot saved: {output_path}")
        plt.close()

    def plot_error_evolution(
        self,
        traj_est: Trajectory,
        ape_metrics: APEMetrics,
        output_path: str = 'error_evolution.png'
    ):
        """
        Plot APE error over time.
        """
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))

        # Relative time axis
        time_axis = traj_est.timestamps - traj_est.timestamps[0]

        # Error curve
        ax.plot(
            time_axis,
            ape_metrics.errors,
            color='#D62728',
            linewidth=2,
            alpha=0.8,
            label='APE',
            zorder=2
        )

        # Mean line
        ax.axhline(
            y=ape_metrics.mean,
            color='blue',
            linestyle='--',
            linewidth=2,
            alpha=0.8,
            label=f'Mean: {ape_metrics.mean:.4f} m',
            zorder=1
        )

        # Std deviation band
        ax.fill_between(
            time_axis,
            ape_metrics.mean - ape_metrics.std,
            ape_metrics.mean + ape_metrics.std,
            color='blue',
            alpha=0.2,
            label=f'±1σ: {ape_metrics.std:.4f} m',
            zorder=0
        )

        # Statistics box
        stats_text = (
            f"RMSE:   {ape_metrics.rmse:.4f} m\n"
            f"Mean:   {ape_metrics.mean:.4f} m\n"
            f"Median: {ape_metrics.median:.4f} m\n"
            f"Std:    {ape_metrics.std:.4f} m\n"
            f"Max:    {ape_metrics.max:.4f} m\n"
            f"Min:    {ape_metrics.min:.4f} m"
        )

        ax.text(
            0.98, 0.97,
            stats_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, pad=0.8),
            family='monospace'
        )

        # Styling
        ax.set_xlabel('Time [s]', fontsize=14, weight='bold')
        ax.set_ylabel('Translation Error [m]', fontsize=14, weight='bold')
        ax.set_title(
            'Absolute Pose Error (APE) Evolution',
            fontsize=16,
            weight='bold',
            pad=20
        )
        ax.legend(loc='upper left', fontsize=12, framealpha=0.95)
        ax.grid(True, linestyle=':', alpha=0.4, linewidth=0.8)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Error evolution plot saved: {output_path}")
        plt.close()

    def plot_combined_evaluation(
        self,
        traj_ref: Trajectory,
        traj_est: Trajectory,
        ape_metrics: APEMetrics,
        output_path: str = 'evaluation_combined.png'
    ):
        """
        Create combined evaluation figure with trajectory and error plots.
        """
        fig = plt.figure(figsize=(18, 7))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1])

        # ====================================================================
        # Left: Trajectory comparison
        # ====================================================================
        ax1 = fig.add_subplot(gs[0, 0])

        # Ground Truth
        ax1.plot(
            traj_ref.positions[:, 0],
            traj_ref.positions[:, 1],
            color='#666666',
            linestyle='--',
            linewidth=3,
            alpha=0.7,
            label='Ground Truth',
            zorder=1
        )

        # Estimated trajectory with error coloring
        errors_normalized = ape_metrics.errors / ape_metrics.max
        points = traj_est.positions[:, :2].reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        from matplotlib.collections import LineCollection
        lc = LineCollection(
            segments,
            cmap='hot_r',
            linewidths=3,
            alpha=0.9,
            zorder=2
        )
        lc.set_array(errors_normalized[:-1])
        line = ax1.add_collection(lc)

        # Markers
        ax1.scatter(
            traj_ref.positions[0, 0],
            traj_ref.positions[0, 1],
            color='green',
            marker='o',
            s=250,
            edgecolors='white',
            linewidths=3,
            label='Start',
            zorder=5
        )

        ax1.scatter(
            traj_ref.positions[-1, 0],
            traj_ref.positions[-1, 1],
            color='red',
            marker='s',
            s=250,
            edgecolors='white',
            linewidths=3,
            label='End',
            zorder=5
        )

        ax1.set_xlabel('X [m]', fontsize=14, weight='bold')
        ax1.set_ylabel('Y [m]', fontsize=14, weight='bold')
        ax1.set_title(
            f'Trajectory Comparison (RMSE: {ape_metrics.rmse:.4f} m)',
            fontsize=16,
            weight='bold',
            pad=15
        )
        ax1.legend(loc='best', fontsize=11, framealpha=0.95)
        ax1.axis('equal')
        ax1.grid(True, linestyle=':', alpha=0.4)

        # ====================================================================
        # Right: Error evolution
        # ====================================================================
        ax2 = fig.add_subplot(gs[0, 1])

        time_axis = traj_est.timestamps - traj_est.timestamps[0]

        ax2.plot(
            time_axis,
            ape_metrics.errors,
            color='#D62728',
            linewidth=2,
            alpha=0.8,
            zorder=2
        )

        ax2.axhline(
            y=ape_metrics.mean,
            color='blue',
            linestyle='--',
            linewidth=2,
            alpha=0.8,
            label=f'Mean: {ape_metrics.mean:.4f} m'
        )

        ax2.fill_between(
            time_axis,
            ape_metrics.mean - ape_metrics.std,
            ape_metrics.mean + ape_metrics.std,
            color='blue',
            alpha=0.2,
            label=f'±1σ: {ape_metrics.std:.4f} m'
        )

        # Statistics
        stats_text = (
            f"RMSE:   {ape_metrics.rmse:.4f} m\n"
            f"Mean:   {ape_metrics.mean:.4f} m\n"
            f"Median: {ape_metrics.median:.4f} m\n"
            f"Std:    {ape_metrics.std:.4f} m\n"
            f"Max:    {ape_metrics.max:.4f} m"
        )

        ax2.text(
            0.97, 0.97,
            stats_text,
            transform=ax2.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, pad=0.6),
            family='monospace'
        )

        ax2.set_xlabel('Time [s]', fontsize=14, weight='bold')
        ax2.set_ylabel('Translation Error [m]', fontsize=14, weight='bold')
        ax2.set_title(
            'APE Evolution',
            fontsize=16,
            weight='bold',
            pad=15
        )
        ax2.legend(loc='upper left', fontsize=11, framealpha=0.95)
        ax2.grid(True, linestyle=':', alpha=0.4)
        ax2.set_xlim(left=0)
        ax2.set_ylim(bottom=0)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Combined evaluation plot saved: {output_path}")
        plt.close()

    def plot_three_trajectories(
        self,
        traj_gt: Trajectory,
        traj_baseline: Trajectory,
        traj_ours: Trajectory,
        baseline_metrics: APEMetrics,
        ours_metrics: APEMetrics,
        output_path: str = 'three_trajectories.png'
    ):
        """
        Plot GT + Baseline + Ours on the same figure.

        Features:
        - Ground truth as gray dashed line
        - Baseline as blue line
        - Ours as red line
        - Start/End markers
        - RMSE comparison in legend
        """
        fig, ax = plt.subplots(1, 1, figsize=(14, 11))

        # Ground Truth
        ax.plot(
            traj_gt.positions[:, 0],
            traj_gt.positions[:, 1],
            color='#666666',
            linestyle='--',
            linewidth=3.5,
            alpha=0.8,
            label='Ground Truth',
            zorder=1
        )

        # Baseline
        ax.plot(
            traj_baseline.positions[:, 0],
            traj_baseline.positions[:, 1],
            color='#0173B2',
            linewidth=2.5,
            alpha=0.85,
            label=f'Baseline (RMSE: {baseline_metrics.rmse:.3f} m)',
            zorder=2
        )

        # Ours
        ax.plot(
            traj_ours.positions[:, 0],
            traj_ours.positions[:, 1],
            color='#D62728',
            linewidth=2.5,
            alpha=0.9,
            label=f'Ours (RMSE: {ours_metrics.rmse:.3f} m)',
            zorder=3
        )

        # Start marker
        ax.scatter(
            traj_gt.positions[0, 0],
            traj_gt.positions[0, 1],
            color='green',
            marker='o',
            s=350,
            edgecolors='white',
            linewidths=3,
            label='Start',
            zorder=5
        )

        # End marker (GT)
        ax.scatter(
            traj_gt.positions[-1, 0],
            traj_gt.positions[-1, 1],
            color='red',
            marker='s',
            s=350,
            edgecolors='white',
            linewidths=3,
            label='End',
            zorder=5
        )

        # Improvement text
        improvement = (baseline_metrics.rmse - ours_metrics.rmse) / baseline_metrics.rmse * 100
        improvement_text = f"Improvement: {improvement:+.2f}%"
        color_imp = 'green' if improvement > 0 else 'red'

        ax.text(
            0.02, 0.98,
            improvement_text,
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment='top',
            fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=color_imp, alpha=0.3, pad=1.0),
            color=color_imp
        )

        # Styling
        ax.set_xlabel('X [m]', fontsize=14, weight='bold')
        ax.set_ylabel('Y [m]', fontsize=14, weight='bold')
        ax.set_title(
            'Trajectory Comparison: GT + Baseline + Ours',
            fontsize=16,
            weight='bold',
            pad=20
        )
        ax.legend(loc='best', fontsize=12, framealpha=0.95, shadow=True)
        ax.axis('equal')
        ax.grid(True, linestyle=':', alpha=0.4, linewidth=0.8)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Three trajectories plot saved: {output_path}")
        plt.close()

    # ========================================================================
    # Main Evaluation Pipeline
    # ========================================================================

    def evaluate(
        self,
        gt_path: str,
        est_path: str,
        output_dir: str = '.',
        output_prefix: str = 'eval'
    ) -> dict:
        """
        Complete evaluation pipeline.

        Args:
            gt_path: Path to ground truth trajectory (TUM format)
            est_path: Path to estimated trajectory (TUM format)
            output_dir: Directory to save output plots
            output_prefix: Prefix for output files

        Returns:
            Dictionary containing all evaluation results
        """
        print("\n" + "="*80)
        print("  TRAJECTORY EVALUATION - Academic Publication Grade")
        print("="*80)

        # Load trajectories
        print("\n1. Loading trajectories...")
        traj_gt = self.load_tum_trajectory(gt_path)
        traj_est = self.load_tum_trajectory(est_path)
        print(f"  GT:  {len(traj_gt)} poses")
        print(f"  EST: {len(traj_est)} poses")

        # Clip start
        print("\n2. Preprocessing...")
        traj_est = self.clip_start(traj_est)
        print(f"  EST after clipping: {len(traj_est)} poses")

        # Synchronize
        print("\n3. Time synchronization...")
        traj_gt_sync, traj_est_sync = self.synchronize_trajectories(traj_gt, traj_est)

        # Align
        print("\n4. Trajectory alignment...")
        traj_est_aligned, alignment_result = self.align_trajectory(
            traj_gt_sync,
            traj_est_sync
        )

        # Calculate metrics
        print("\n5. Calculating APE metrics...")
        ape_metrics = self.calculate_ape(traj_gt_sync, traj_est_aligned)

        print("\n  APE Statistics:")
        print(f"    RMSE:   {ape_metrics.rmse:.6f} m")
        print(f"    Mean:   {ape_metrics.mean:.6f} m")
        print(f"    Median: {ape_metrics.median:.6f} m")
        print(f"    Std:    {ape_metrics.std:.6f} m")
        print(f"    Max:    {ape_metrics.max:.6f} m")
        print(f"    Min:    {ape_metrics.min:.6f} m")

        # Visualization
        print("\n6. Generating plots...")

        import os
        os.makedirs(output_dir, exist_ok=True)

        # Combined plot (recommended)
        self.plot_combined_evaluation(
            traj_gt_sync,
            traj_est_aligned,
            ape_metrics,
            output_path=os.path.join(output_dir, f'{output_prefix}_combined.png')
        )

        # Individual plots
        self.plot_trajectory_comparison(
            traj_gt_sync,
            traj_est_aligned,
            ape_metrics,
            output_path=os.path.join(output_dir, f'{output_prefix}_trajectory.png')
        )

        self.plot_error_evolution(
            traj_est_aligned,
            ape_metrics,
            output_path=os.path.join(output_dir, f'{output_prefix}_error.png')
        )

        print("\n" + "="*80)
        print("✓ Evaluation completed successfully!")
        print("="*80 + "\n")

        return {
            'ape_metrics': ape_metrics,
            'alignment_result': alignment_result,
            'traj_gt_sync': traj_gt_sync,
            'traj_est_aligned': traj_est_aligned
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    """
    Example usage for VINS trajectory evaluation.

    Typical use cases:
    1. Stereo/Deep VINS with correct scale -> use SE3 alignment
    2. Monocular VINS with scale drift -> use Sim3 alignment
    """

    # Example 1: SE3 alignment (for systems with correct scale)
    print("\n" + "="*80)
    print("EXAMPLE 1: SE3 Alignment (Stereo/Deep VINS)")
    print("="*80)

    evaluator_se3 = TrajectoryEvaluator(
        alignment_mode='se3',
        start_clip_seconds=2.0,  # Discard first 2 seconds
        max_time_diff=0.02       # 20ms timestamp tolerance
    )

    # Uncomment and modify paths for your data:
    # results_se3 = evaluator_se3.evaluate(
    #     gt_path='/path/to/groundtruth.txt',
    #     est_path='/path/to/vins_estimate.txt',
    #     output_dir='./evaluation_results',
    #     output_prefix='vins_se3'
    # )

    # Example 2: Sim3 alignment (for monocular VINS with scale drift)
    print("\n" + "="*80)
    print("EXAMPLE 2: Sim3 Alignment (Monocular VINS)")
    print("="*80)

    evaluator_sim3 = TrajectoryEvaluator(
        alignment_mode='sim3',
        start_clip_seconds=3.0,
        max_time_diff=0.02
    )

    # Uncomment and modify paths for your data:
    # results_sim3 = evaluator_sim3.evaluate(
    #     gt_path='/path/to/groundtruth.txt',
    #     est_path='/path/to/vins_mono_estimate.txt',
    #     output_dir='./evaluation_results',
    #     output_prefix='vins_sim3'
    # )

    print("\n" + "="*80)
    print("For actual evaluation, uncomment the evaluation calls above")
    print("and provide paths to your trajectory files.")
    print("="*80 + "\n")
