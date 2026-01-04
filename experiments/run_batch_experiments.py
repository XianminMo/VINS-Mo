#!/usr/bin/env python3
"""
Automated Batch Experiment Runner for VINS Depth Weight Sensitivity Analysis
Modifies config parameters, runs ROS system, evaluates results, and aggregates data.
"""

import os
import sys
import time
import yaml
import shutil
import subprocess
import signal
import re
import json
from datetime import datetime
from pathlib import Path

# ================= Configuration =================
DATASET_ROOT = "/home/linux/mxm/data/Euroc"
CONFIG_FILE = "/home/linux/mxm/proj/VINS-Mo/src/VINS-Mo/config/euroc/euroc_config.yaml"
BACKUP_FILE = CONFIG_FILE + ".bak"
PLOT_SCRIPT = "/home/linux/mxm/proj/VINS-Mo/src/VINS-Mo/experiments/plot_comprehensive.py"
LAUNCH_FILE = "/home/linux/mxm/proj/VINS-Mo/src/VINS-Mo/vins_estimator/launch/euroc.launch"

# Experiment parameters
DATASETS = ["V2_03_difficult", "MH_05_difficult"]
WEIGHTS = [1.0, 2.5, 5.0, 10.0]  # depth_constraint.weight values

# ROS settings
PLAYBACK_RATE = 1.5  # Speed up bag playback

# Output base directory
EXPERIMENTS_DIR = "/home/linux/mxm/proj/VINS-Mo/src/VINS-Mo/experiments"

# ================= Helper Functions =================

def backup_config():
    """Backup the config file"""
    print(f"[Backup] Creating backup: {BACKUP_FILE}")
    shutil.copy2(CONFIG_FILE, BACKUP_FILE)

def restore_config():
    """Restore config from backup"""
    if os.path.exists(BACKUP_FILE):
        print(f"[Restore] Restoring config from backup")
        shutil.copy2(BACKUP_FILE, CONFIG_FILE)
    else:
        print(f"[Warning] No backup file found at {BACKUP_FILE}")

def modify_config(weight):
    """Modify the depth_constraint.weight parameter in config"""
    print(f"[Config] Setting depth_constraint.weight = {weight}")

    with open(CONFIG_FILE, 'r') as f:
        lines = f.readlines()

    modified = False
    for i, line in enumerate(lines):
        # Match the depth_constraint.weight line
        if re.match(r'^depth_constraint\.weight:\s*[\d.]+', line):
            lines[i] = f"depth_constraint.weight: {weight}\n"
            modified = True
            break

    if not modified:
        raise ValueError("Could not find 'depth_constraint.weight' in config file")

    with open(CONFIG_FILE, 'w') as f:
        f.writelines(lines)

    print(f"[Config] Updated successfully")

def update_output_path_in_config(output_path):
    """Update the output_path in config"""
    print(f"[Config] Setting output_path = {output_path}")

    with open(CONFIG_FILE, 'r') as f:
        lines = f.readlines()

    modified = False
    for i, line in enumerate(lines):
        if re.match(r'^output_path:', line):
            lines[i] = f'output_path: "{output_path}"\n'
            modified = True
            break

    if not modified:
        raise ValueError("Could not find 'output_path' in config file")

    with open(CONFIG_FILE, 'w') as f:
        f.writelines(lines)

def get_bag_path(dataset):
    """Get the path to the rosbag file"""
    # Try both locations: DATASET_ROOT/dataset.bag and DATASET_ROOT/dataset/dataset.bag
    bag_path1 = os.path.join(DATASET_ROOT, f"{dataset}.bag")
    bag_path2 = os.path.join(DATASET_ROOT, dataset, f"{dataset}.bag")

    if os.path.exists(bag_path1):
        return bag_path1
    elif os.path.exists(bag_path2):
        return bag_path2
    else:
        raise FileNotFoundError(f"Bag file not found at: {bag_path1} or {bag_path2}")
    return bag_path

def get_gt_path(dataset):
    """Get the path to ground truth TUM file"""
    gt_path = os.path.join(DATASET_ROOT, dataset, "mav0", "state_groundtruth_estimate0", "data.tum")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
    return gt_path

def kill_ros_nodes():
    """Kill all ROS nodes to ensure clean state"""
    print("[ROS] Killing all ROS nodes...")
    try:
        subprocess.run(["killall", "-9", "roslaunch", "roscore", "rosbag", "rosmaster"],
                      stderr=subprocess.DEVNULL)
        time.sleep(2)
    except Exception as e:
        print(f"[Warning] Error killing ROS nodes: {e}")

def run_vins_experiment(dataset, weight, output_dir):
    """Run a single VINS experiment"""
    print(f"\n{'='*80}")
    print(f"[Experiment] Dataset: {dataset}, Weight: {weight}")
    print(f"{'='*80}\n")

    # Get paths
    bag_path = get_bag_path(dataset)
    gt_path = get_gt_path(dataset)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Update config with output path
    update_output_path_in_config(output_dir)

    # Kill any existing ROS nodes
    kill_ros_nodes()

    # Start roscore
    print("[ROS] Starting roscore...")
    roscore_proc = subprocess.Popen(["roscore"],
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
    time.sleep(3)

    # Launch VINS estimator
    print("[ROS] Launching VINS estimator...")
    vins_proc = subprocess.Popen(["roslaunch", "vins_estimator", "euroc.launch"],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True)
    time.sleep(5)

    # Play bag file
    print(f"[ROS] Playing bag file (rate={PLAYBACK_RATE}x): {bag_path}")
    bag_proc = subprocess.run(["rosbag", "play", bag_path, "-r", str(PLAYBACK_RATE)],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)

    if bag_proc.returncode != 0:
        print(f"[Error] Bag playback failed: {bag_proc.stderr}")
    else:
        print("[ROS] Bag playback completed")

    # Wait a bit for processing to complete
    time.sleep(5)

    # Kill VINS and roscore
    print("[ROS] Stopping VINS and roscore...")
    vins_proc.terminate()
    time.sleep(2)
    vins_proc.kill()
    roscore_proc.terminate()
    time.sleep(2)
    roscore_proc.kill()

    kill_ros_nodes()

    # Check if output trajectory was generated
    # Try multiple possible output filenames
    possible_files = [
        os.path.join(output_dir, "vins_result_no_loop.tum"),
        os.path.join(output_dir, "vins_closed_loop.tum"),
        os.path.join(output_dir, "vins_open_loop.tum")
    ]

    est_file = None
    for pf in possible_files:
        if os.path.exists(pf):
            est_file = pf
            break

    if est_file is None:
        print(f"[Error] No output trajectory found in: {output_dir}")
        print(f"[Error] Checked: {', '.join([os.path.basename(f) for f in possible_files])}")
        return None

    print(f"[Success] Trajectory saved: {est_file}")
    return est_file

def evaluate_trajectory(gt_file, est_file, output_dir, name):
    """Evaluate trajectory using plot_comprehensive.py"""
    print(f"\n[Evaluation] Running evaluation for {name}...")

    cmd = [
        "python3", PLOT_SCRIPT,
        "--gt_file", gt_file,
        "--est_file", est_file,
        "--output_dir", output_dir,
        "--name", name
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            print(f"[Error] Evaluation failed: {result.stderr}")
            return None

        # Parse JSON result from output
        for line in result.stdout.split('\n'):
            if line.startswith("RESULT_JSON::"):
                json_str = line.split("RESULT_JSON::")[1]
                metrics = json.loads(json_str)
                print(f"[Evaluation] APE RMSE: {metrics['ape_rmse']:.4f} m")
                return metrics

        print("[Warning] Could not find RESULT_JSON in output")
        return None

    except subprocess.TimeoutExpired:
        print("[Error] Evaluation timed out")
        return None
    except Exception as e:
        print(f"[Error] Evaluation exception: {e}")
        return None

def main():
    """Main experiment loop"""
    print("\n" + "="*80)
    print("VINS Depth Weight Sensitivity Analysis - Batch Experiment Runner")
    print("="*80 + "\n")

    # Create experiment directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = os.path.join(EXPERIMENTS_DIR, f"batch_{timestamp}")
    os.makedirs(batch_dir, exist_ok=True)
    print(f"[Setup] Batch directory: {batch_dir}\n")

    # Backup config
    try:
        backup_config()
    except Exception as e:
        print(f"[Error] Failed to backup config: {e}")
        return 1

    # Prepare results list
    results = []

    try:
        # Iterate through experiments
        for dataset in DATASETS:
            for weight in WEIGHTS:
                experiment_name = f"{dataset}_w{weight}"
                experiment_dir = os.path.join(batch_dir, experiment_name)

                try:
                    # Modify config
                    modify_config(weight)

                    # Run experiment
                    est_file = run_vins_experiment(dataset, weight, experiment_dir)

                    if est_file is None:
                        print(f"[Skip] Experiment failed for {experiment_name}")
                        results.append({
                            'dataset': dataset,
                            'weight': weight,
                            'status': 'FAILED',
                            'error': 'Trajectory generation failed'
                        })
                        continue

                    # Evaluate
                    gt_file = get_gt_path(dataset)
                    metrics = evaluate_trajectory(gt_file, est_file, experiment_dir, experiment_name)

                    if metrics is None:
                        print(f"[Skip] Evaluation failed for {experiment_name}")
                        results.append({
                            'dataset': dataset,
                            'weight': weight,
                            'status': 'EVAL_FAILED',
                            'error': 'Evaluation failed'
                        })
                        continue

                    # Record results
                    results.append({
                        'dataset': dataset,
                        'weight': weight,
                        'status': 'SUCCESS',
                        **metrics
                    })

                except Exception as e:
                    print(f"[Error] Exception in experiment {experiment_name}: {e}")
                    results.append({
                        'dataset': dataset,
                        'weight': weight,
                        'status': 'ERROR',
                        'error': str(e)
                    })

    finally:
        # Always restore config
        restore_config()
        kill_ros_nodes()

    # Save summary CSV
    summary_file = os.path.join(batch_dir, "summary.csv")
    print(f"\n[Summary] Writing results to {summary_file}")

    with open(summary_file, 'w') as f:
        # Header
        f.write("dataset,weight,status,ape_rmse,ape_mean,ape_max,ape_std,rpe_rmse,rpe_mean,rpe_max,rpe_std\n")

        # Data rows
        for r in results:
            dataset = r['dataset']
            weight = r['weight']
            status = r['status']

            if status == 'SUCCESS':
                f.write(f"{dataset},{weight},{status},"
                       f"{r['ape_rmse']:.6f},{r['ape_mean']:.6f},{r['ape_max']:.6f},{r['ape_std']:.6f},"
                       f"{r['rpe_rmse']:.6f},{r['rpe_mean']:.6f},{r['rpe_max']:.6f},{r['rpe_std']:.6f}\n")
            else:
                f.write(f"{dataset},{weight},{status},,,,,,,,,\n")

    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    for r in results:
        print(f"{r['dataset']:<20} w={r['weight']:<5} -> {r['status']}")
        if r['status'] == 'SUCCESS':
            print(f"  APE RMSE: {r['ape_rmse']:.4f} m")

    print(f"\n[Complete] All experiments finished. Results saved to: {batch_dir}")
    print("="*80 + "\n")

    return 0

if __name__ == "__main__":
    sys.exit(main())
