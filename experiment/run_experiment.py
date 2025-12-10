import subprocess
import time
import os
import signal
import shutil
import sys

# ================= 配置区域 =================
# 1. 数据集设置
BAG_PATH = "/home/linux/mxm/data/EuRoC/V1_03_difficult.bag" # 请修改为你实际的 bag 路径
LAUNCH_CMD = ["roslaunch", "vins_estimator", "euroc.launch"]

# 2. 输出路径配置 (根据你的描述修改)
SOURCE_INIT_FILE = "/home/linux/mxm/output/experiments_initial/vio_init_window.txt"
SOURCE_RUN_FILE  = "/home/linux/mxm/output/experiments_initial/vins_closed_loop.tum"

# 3. 结果保存目录
RESULT_DIR = "/home/linux/mxm/output/experiments_initial/Depth_anything/V103_test/1209/window_0.3s/original/1" 


# 5. 实验参数
SLICE_DURATION = 20.0   # 切片时长
STEP_SIZE = 2.0         # 步长
START_TIME = 0
END_TIME = 109
WAIT_TIME = 5.0         # 留给 VINS 计算和写文件的时间
# ===========================================

def run():
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    current_start = START_TIME
    idx = 0

    print(f"========== 开始自动化测试 ==========")
    print(f"检测目标文件: {SOURCE_INIT_FILE}")

    while current_start < END_TIME:
        test_id = f"{current_start:.1f}"
        print(f"\n=== Test {idx}: Start at {test_id}s ===")
        
        # 1. 【关键】清理旧文件 
        # 必须在启动前删掉源文件，否则无法区分是这次生成的还是上次残留的
        if os.path.exists(SOURCE_INIT_FILE):
            os.remove(SOURCE_INIT_FILE)
        if os.path.exists(SOURCE_RUN_FILE):
            os.remove(SOURCE_RUN_FILE)

        # 2. 启动 VINS
        # 我们不再重定向 stdout 到文件，而是丢弃它 (DEVNULL)，或者你可以让它输出到终端
        # 这里选择 DEVNULL 保持清爽，因为我们只看结果文件
        process = subprocess.Popen(
            LAUNCH_CMD, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL, 
            preexec_fn=os.setsid
        )
        time.sleep(2) # 等待节点启动 Warmup

        # 3. 播放 Bag 切片
        print(f"  -> Playing bag...")
        bag_cmd = ["rosbag", "play", BAG_PATH, "-s", str(current_start), "-u", str(SLICE_DURATION)]
        subprocess.call(bag_cmd, stdout=subprocess.DEVNULL)

        # 4. 等待计算
        print(f"  -> Waiting {WAIT_TIME}s for computation...")
        time.sleep(WAIT_TIME)

        # 5. 【核心修改】通过检查文件是否存在来判断成功
        is_success = False
        if os.path.exists(SOURCE_INIT_FILE) and os.path.getsize(SOURCE_INIT_FILE) > 0:
            is_success = True
        
        # 6. 处理结果
        if is_success:
            print("\033[92m  [SUCCESS] Init file generated!\033[0m") # 绿色
            
            # 复制文件
            dst_init = os.path.join(RESULT_DIR, f"traj_{test_id}_init.txt")
            shutil.copy(SOURCE_INIT_FILE, dst_init)
            
            # 检查是否有 Run 文件 (有时候初始化成功但不一定有 Run 文件，视你代码逻辑而定)
            if os.path.exists(SOURCE_RUN_FILE) and os.path.getsize(SOURCE_RUN_FILE) > 0:
                dst_run = os.path.join(RESULT_DIR, f"traj_{test_id}_run.txt")
                shutil.copy(SOURCE_RUN_FILE, dst_run)
            else:
                print("  [Info] Run file not found (maybe output delayed?)")
                
        else:
            print("\033[91m  [FAILED] No output file generated.\033[0m") # 红色
            # 生成空标记文件，用于统计分母
            open(os.path.join(RESULT_DIR, f"traj_{test_id}_fail.mark"), 'a').close()

        # 7. 杀死进程
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGINT)
            process.wait(timeout=2)
        except:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        
        current_start += STEP_SIZE
        idx += 1

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        # 清理残留进程
        subprocess.call(["pkill", "-f", "roslaunch"])
        subprocess.call(["pkill", "-f", "vins_node"])
        print("\nStopped by user.")