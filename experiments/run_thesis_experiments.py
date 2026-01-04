#!/usr/bin/env python3
"""
Automated VIO Thesis Experiment Suite
Author: Claude Code
Date: 2025-12-08

Automates the complete experimental loop:
1. Modify Config
2. Launch ROS
3. Play Bag
4. Harvest Data
5. Evaluate & Plot
6. Aggregate Statistics
"""

import os
import sys
import time
import shutil
import subprocess
import signal
import re
import json
import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd

# ===================== CONFIGURATION =====================

# Paths
PROJECT_ROOT = Path("/home/linux/mxm/proj/VINS-Mo/src/VINS-Mo")
CONFIG_FILE = PROJECT_ROOT / "config/euroc/euroc_config.yaml"
CONFIG_BACKUP = CONFIG_FILE.parent / "euroc_config.yaml.bak"
DATASET_ROOT = Path("/home/linux/mxm/data/EuRoC")
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
PLOT_SCRIPT = EXPERIMENTS_DIR / "plot_comprehensive.py"

# Experiment Matrix
DATASETS = [
    "V1_01_easy",
    "V1_03_difficult",
    "V2_01_easy",
    "V2_03_difficult",
    "MH_01_easy",
    "MH_02_easy",
    "MH_04_difficult",
    "MH_05_difficult"
]

CONFIGS = {
    "Baseline": {"weight_mode": -1, "weight": 0.0, "description": "No Depth Constraint"},  # Baseline without depth
    "Config_A": {"weight_mode": 0, "weight": 1.0, "description": "Fixed Conservative"},
    "Config_B": {"weight_mode": 0, "weight": 2.5, "description": "Fixed Balanced"},
    "Config_C": {"weight_mode": 0, "weight": 5.0, "description": "Fixed Aggressive"},
    "Config_D": {"weight_mode": 1, "weight": 2.5, "description": "Adaptive Strategy"}  # weight not used in mode 1
}

TRIALS = 3

# ROS Settings
ROS_LAUNCH_WAIT = 5  # seconds to wait for ROS to start
ROS_SHUTDOWN_WAIT = 8  # seconds to wait after bag finishes

# ===================== HELPER FUNCTIONS =====================

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)

def print_subheader(text):
    """Print a formatted subheader"""
    print("\n" + "-"*80)
    print(f"  {text}")
    print("-"*80)

def backup_config():
    """Backup the configuration file"""
    print(f"📁 Backing up config: {CONFIG_FILE} -> {CONFIG_BACKUP}")
    shutil.copy2(CONFIG_FILE, CONFIG_BACKUP)

def restore_config():
    """Restore the configuration file from backup"""
    if CONFIG_BACKUP.exists():
        print(f"📁 Restoring config from backup: {CONFIG_BACKUP} -> {CONFIG_FILE}")
        shutil.copy2(CONFIG_BACKUP, CONFIG_FILE)
    else:
        print("⚠️  Warning: Backup file not found, cannot restore config")

def modify_config(weight_mode, weight_value):
    """
    Modify config file using regex to preserve comments

    Args:
        weight_mode: -1 for baseline (no depth), 0 for fixed, 1 for adaptive
        weight_value: weight value (used only in mode 0)
    """
    print(f"✏️  Modifying config: weight_mode={weight_mode}, weight={weight_value}")

    with open(CONFIG_FILE, 'r') as f:
        content = f.read()

    # Handle baseline mode (disable depth constraint)
    if weight_mode == -1:
        # Disable depth constraint estimation
        content = re.sub(
            r'(depth_constraint\.estimate_scale_shift:\s*)\d+',
            r'\g<1>0',
            content
        )
        # Also disable fast init to use traditional initialization
        content = re.sub(
            r'(use_fast_init:\s*)\d+',
            r'\g<1>0',
            content
        )
        print(f"✅ Baseline mode: depth constraint and fast init disabled")
    else:
        # Enable depth constraint estimation
        content = re.sub(
            r'(depth_constraint\.estimate_scale_shift:\s*)\d+',
            r'\g<1>1',
            content
        )
        # Modify weight_mode
        content = re.sub(
            r'(depth_constraint\.weight_mode:\s*)\d+',
            rf'\g<1>{weight_mode}',
            content
        )
        # Modify weight (fixed weight parameter)
        content = re.sub(
            r'(depth_constraint\.weight:\s*)\d+\.?\d*',
            rf'\g<1>{weight_value}',
            content
        )

    with open(CONFIG_FILE, 'w') as f:
        f.write(content)

    print(f"✅ Config modified successfully")

def get_dataset_paths(dataset_name):
    """Get paths for a dataset"""
    dataset_path = DATASET_ROOT / dataset_name
    bag_file = DATASET_ROOT / f"{dataset_name}.bag"  # Bag file is in parent directory
    gt_file = dataset_path / "mav0" / "state_groundtruth_estimate0" / "data.tum"

    return {
        "dataset_path": dataset_path,
        "bag_file": bag_file,
        "gt_file": gt_file
    }

def get_vins_output_path():
    """
    Get the VINS output path from the config file
    Returns the path where vins_closed_loop.tum will be saved
    """
    with open(CONFIG_FILE, 'r') as f:
        content = f.read()

    # Extract output_path from config
    match = re.search(r'output_path:\s*["\']?([^"\'\n]+)["\']?', content)
    if match:
        output_path = Path(match.group(1).strip())
        return output_path / "vins_closed_loop.tum"
    else:
        raise ValueError("Could not find output_path in config file")

def launch_ros():
    """Launch ROS in background"""
    print(f"🚀 Launching ROS (roslaunch vins_estimator euroc.launch)...")

    # Source ROS environment
    cmd = """
    source /opt/ros/noetic/setup.bash && \
    source /home/linux/mxm/proj/VINS-Mo/devel/setup.bash && \
    roslaunch vins_estimator euroc.launch
    """

    process = subprocess.Popen(
        cmd,
        shell=True,
        executable="/bin/bash",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid  # Create new process group for clean killing
    )

    print(f"⏳ Waiting {ROS_LAUNCH_WAIT}s for ROS to initialize...")
    time.sleep(ROS_LAUNCH_WAIT)

    return process

def play_bag(bag_file):
    """Play rosbag at 1x speed and wait for completion"""
    print(f"▶️  Playing bag: {bag_file} at 1x speed")

    if not bag_file.exists():
        raise FileNotFoundError(f"Bag file not found: {bag_file}")

    # Force 1x playback rate with -r 1.0
    cmd = f"""
    source /opt/ros/noetic/setup.bash && \
    rosbag play -r 1.0 {bag_file}
    """

    result = subprocess.run(
        cmd,
        shell=True,
        executable="/bin/bash",
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"⚠️  rosbag play returned non-zero: {result.returncode}")
        print(f"STDERR: {result.stderr}")

    print(f"✅ Bag playback completed")
    print(f"⏳ Waiting {ROS_SHUTDOWN_WAIT}s for VINS to flush data...")
    time.sleep(ROS_SHUTDOWN_WAIT)

    return result.returncode == 0

def kill_ros(ros_process):
    """Gracefully kill ROS process"""
    print("🛑 Shutting down ROS...")

    try:
        # Send SIGINT to the entire process group
        os.killpg(os.getpgid(ros_process.pid), signal.SIGINT)

        # Wait for graceful shutdown
        try:
            ros_process.wait(timeout=10)
            print("✅ ROS shutdown completed")
        except subprocess.TimeoutExpired:
            print("⚠️  Graceful shutdown timeout, forcing kill...")
            os.killpg(os.getpgid(ros_process.pid), signal.SIGKILL)
            ros_process.wait()
            print("✅ ROS forcefully killed")
    except ProcessLookupError:
        print("⚠️  ROS process already terminated")
    except Exception as e:
        print(f"⚠️  Error killing ROS: {e}")

    # Extra cleanup - kill any remaining ROS nodes
    subprocess.run("killall -9 rosmaster roscore roslaunch 2>/dev/null || true", shell=True)
    time.sleep(2)

def harvest_data(vins_output_file, target_dir):
    """Copy VINS output to experiment directory"""
    print(f"📦 Harvesting data from {vins_output_file} to {target_dir}")

    if not vins_output_file.exists():
        print(f"❌ ERROR: VINS output file not found: {vins_output_file}")
        return False

    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / "vins_closed_loop.tum"
    shutil.copy2(vins_output_file, target_file)

    print(f"✅ Data harvested to {target_file}")
    return True

def evaluate_trajectory(gt_file, est_file, output_dir, config_name):
    """Run plot_comprehensive.py and extract metrics"""
    print(f"📊 Evaluating trajectory with plot_comprehensive.py...")

    cmd = [
        "python3",
        str(PLOT_SCRIPT),
        "--gt_file", str(gt_file),
        "--est_file", str(est_file),
        "--output_dir", str(output_dir),
        "--name", config_name,
        "--no_show"
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            print(f"⚠️  Evaluation script returned non-zero: {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return None

        # Extract JSON from output
        for line in result.stdout.split('\n'):
            if line.startswith('RESULT_JSON::'):
                json_str = line.replace('RESULT_JSON::', '').strip()
                metrics = json.loads(json_str)
                print(f"✅ Metrics extracted: APE_RMSE={metrics['ape_rmse']:.4f}m")
                return metrics

        print("⚠️  Could not find RESULT_JSON in output")
        print(f"Output:\n{result.stdout}")
        return None

    except subprocess.TimeoutExpired:
        print("❌ Evaluation timeout!")
        return None
    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        return None

def run_single_experiment(dataset_name, config_name, config_params, trial_num, output_base_dir, results_csv):
    """Run a single experiment"""
    print_subheader(f"Dataset: {dataset_name} | Config: {config_name} | Trial: {trial_num}")

    # Get paths
    dataset_info = get_dataset_paths(dataset_name)

    # Check if dataset exists
    if not dataset_info["bag_file"].exists():
        print(f"❌ ERROR: Bag file not found: {dataset_info['bag_file']}")
        return False

    if not dataset_info["gt_file"].exists():
        print(f"❌ ERROR: Ground truth not found: {dataset_info['gt_file']}")
        return False

    # Modify config
    modify_config(config_params["weight_mode"], config_params["weight"])

    # Create output directory
    run_dir = output_base_dir / dataset_name / config_name / f"trial_{trial_num}"
    run_dir.mkdir(parents=True, exist_ok=True)

    ros_process = None
    success = False

    try:
        # Launch ROS
        ros_process = launch_ros()

        # Play bag
        bag_success = play_bag(dataset_info["bag_file"])

        if not bag_success:
            print("⚠️  Bag playback had issues, but continuing...")

        # Kill ROS
        kill_ros(ros_process)
        ros_process = None

        # Harvest data
        vins_output = get_vins_output_path()
        harvest_success = harvest_data(vins_output, run_dir)

        if not harvest_success:
            print("❌ Data harvesting failed")
            return False

        # Evaluate
        est_file = run_dir / "vins_closed_loop.tum"
        metrics = evaluate_trajectory(
            dataset_info["gt_file"],
            est_file,
            run_dir,
            config_name
        )

        if metrics is None:
            print("❌ Evaluation failed")
            return False

        # Save results immediately
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result_row = {
            "timestamp": timestamp,
            "dataset": dataset_name,
            "config": config_name,
            "weight_mode": config_params["weight_mode"],
            "weight": config_params["weight"],
            "description": config_params["description"],
            "trial": trial_num,
            "ape_rmse": metrics["ape_rmse"],
            "ape_mean": metrics["ape_mean"],
            "ape_max": metrics["ape_max"],
            "ape_std": metrics["ape_std"],
            "ape_median": metrics["ape_median"],
            "ape_min": metrics["ape_min"],
            "rpe_rmse": metrics["rpe_rmse"],
            "rpe_mean": metrics["rpe_mean"],
            "rpe_max": metrics["rpe_max"],
            "rpe_std": metrics["rpe_std"],
            "rpe_median": metrics["rpe_median"],
            "rpe_min": metrics["rpe_min"]
        }

        # Append to CSV
        df = pd.DataFrame([result_row])
        df.to_csv(results_csv, mode='a', header=not results_csv.exists(), index=False)

        print(f"✅ Results saved to {results_csv}")
        success = True

    except KeyboardInterrupt:
        print("\n⚠️  Keyboard interrupt detected!")
        raise
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always clean up ROS
        if ros_process is not None:
            kill_ros(ros_process)

    return success

def generate_statistics_report(results_csv, output_dir):
    """Generate statistical summary from results"""
    print_header("Generating Statistics Report")

    if not results_csv.exists():
        print("❌ No results file found")
        return

    df = pd.read_csv(results_csv)

    # Group by dataset and config, calculate mean and std
    grouped = df.groupby(['dataset', 'config']).agg({
        'ape_rmse': ['mean', 'std', 'min', 'max'],
        'ape_max': ['mean', 'std'],
        'rpe_rmse': ['mean', 'std', 'min', 'max'],
        'weight_mode': 'first',
        'weight': 'first',
        'description': 'first'
    }).reset_index()

    # Flatten column names
    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]

    # Save statistics
    stats_file = output_dir / "statistics_report.csv"
    grouped.to_csv(stats_file, index=False)

    print(f"✅ Statistics saved to {stats_file}")

    # Print summary table
    print("\n" + "="*120)
    print("SUMMARY STATISTICS")
    print("="*120)
    print(grouped.to_string(index=False))
    print("="*120)

def main():
    parser = argparse.ArgumentParser(description='Run VIO Thesis Experiments')
    parser.add_argument('--test', action='store_true',
                        help='Run smoke test (V2_03_difficult, Config_B & Config_D, 1 trial)')
    parser.add_argument('--datasets', nargs='+',
                        help='Override datasets list')
    parser.add_argument('--configs', nargs='+',
                        help='Override configs list (e.g., Config_B Config_D)')
    parser.add_argument('--trials', type=int, default=TRIALS,
                        help='Number of trials per experiment')
    args = parser.parse_args()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base_dir = EXPERIMENTS_DIR / f"thesis_final_{timestamp}"
    output_base_dir.mkdir(parents=True, exist_ok=True)
    results_csv = output_base_dir / "summary_results.csv"

    print_header(f"VIO Thesis Experiment Suite - {timestamp}")
    print(f"📁 Output directory: {output_base_dir}")
    print(f"📊 Results CSV: {results_csv}")

    # Determine experiment matrix
    if args.test:
        datasets_to_run = ["V2_03_difficult"]
        configs_to_run = ["Config_B", "Config_D"]
        trials_to_run = 1
        print("🧪 SMOKE TEST MODE: V2_03_difficult × [Config_B, Config_D] × 1 trial")
    else:
        datasets_to_run = args.datasets if args.datasets else DATASETS
        configs_to_run = args.configs if args.configs else list(CONFIGS.keys())
        trials_to_run = args.trials
        print(f"🔬 FULL EXPERIMENT MODE: {len(datasets_to_run)} datasets × {len(configs_to_run)} configs × {trials_to_run} trials")

    print(f"   Datasets: {datasets_to_run}")
    print(f"   Configs: {configs_to_run}")
    print(f"   Trials: {trials_to_run}")

    # Backup config
    backup_config()

    try:
        # Run experiment loop
        total_experiments = len(datasets_to_run) * len(configs_to_run) * trials_to_run
        current_experiment = 0

        for dataset in datasets_to_run:
            for config_name in configs_to_run:
                config_params = CONFIGS[config_name]
                for trial in range(1, trials_to_run + 1):
                    current_experiment += 1
                    print_header(f"Experiment {current_experiment}/{total_experiments}")

                    success = run_single_experiment(
                        dataset, config_name, config_params, trial,
                        output_base_dir, results_csv
                    )

                    if not success:
                        print(f"⚠️  Experiment failed, continuing to next...")

        # Generate statistics report
        generate_statistics_report(results_csv, output_base_dir)

        print_header("✅ ALL EXPERIMENTS COMPLETED!")
        print(f"📁 Results saved to: {output_base_dir}")

    except KeyboardInterrupt:
        print("\n\n⚠️  INTERRUPTED BY USER!")
    except Exception as e:
        print(f"\n\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always restore config
        restore_config()
        print("\n✅ Config restored from backup")

if __name__ == "__main__":
    main()
