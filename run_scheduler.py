import numpy as np
import subprocess
import time
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import platform

# Настройки
if platform.system() == 'Windows':
    EXE_PATH = os.path.join(os.getcwd(), 'target', 'release', 'v27_sdd_universal.exe')  #r"C:\Users\Ivan\Desktop\v27_sdd_universal\target\release\v27_sdd_universal.exe"
elif platform.system() == 'Linux':
    EXE_PATH = os.path.join(os.getcwd(), 'target', 'release', 'v27_sdd_universal') #r'/home/ivan/sdd_universal/target/release/v27_sdd_universal'

MAX_CPU = max(1, (os.cpu_count() - 2))
LOG_FILE = "run_scheduler.log"
START_AT = 0

# Настройка логирования
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

# Хендлер для файла
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Хендлер для консоли
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# MC(tau_0), constant theta=0.12, case 'MC-TAU'
MC_tau_0 = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 8.2, 10.4, 12.6, 14.8, 17, 19.2, 21.4, 23.6, 25.8, 28, 30.2, 32.4, 34.6, 36.8, 39, 41.2, 43.4, 45.6, 47.8, 50]

# SF(tau_0), constant theta=0.12, case 'SF-TAU'
SF_tau_0 = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 8.2, 10.4, 12.6, 14.8, 17, 19.2, 21.4, 23.6, 25.8, 28, 30.2, 32.4, 34.6, 36.8, 39, 41.2, 43.4, 45.6, 47.8, 50]

# NARMA(tau_0), constant theta=0.12, case 'NARMA-TAU'
NARMA_tau_0 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 30, 35, 40, 45, 50]

# MC(theta), constant theta=false, case 'MC-THETA'
MC_thetas = [0.0005, 0.001, 0.002, 0.004, 0.008, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.36, 0.37, 0.39, 0.41, 0.43, 0.45, 0.47, 0.49, 0.51, 0.53, 0.55, 0.57, 0.59]

# SF(theta), constant theta=false, case 'SF-THETA'
SF_thetas = [0.0005, 0.001, 0.002, 0.004, 0.008, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.36, 0.37, 0.39, 0.41, 0.43, 0.45, 0.47, 0.49, 0.51, 0.53, 0.55, 0.57, 0.59, 0.61, 0.63, 0.65, 0.67, 0.69, 0.71, 0.73, 0.75, 0.77, 0.79]

# NARMA(theta), constant theta=false, case 'NARMA-THETA'
NARMA_thetas = [0.0005, 0.001, 0.002, 0.004, 0.008, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.36, 0.37, 0.39, 0.41, 0.43, 0.45, 0.47, 0.49, 0.51, 0.53, 0.55, 0.57, 0.59, 0.61, 0.63, 0.65, 0.67, 0.69, 0.71, 0.73, 0.75, 0.77, 0.79]

# MC(beta, theta), constant theta=false, case 'MC-2D'
# SF(beta, theta), constant theta=false, case 'SF-2D'
# NARMA(beta, theta), constant theta=false, case 'NARMA-2D'
beta_theta = [(0.0005, 0), (0.0005, 0.2311), (0.0005, 0.4522), (0.0005, 0.6733), (0.0005, 0.8944), (0.0005, 1.1156), (0.0005, 1.3367), (0.0005, 1.5578), (0.0005, 1.7789), (0.0005, 2), (0.0782, 0), (0.0782, 0.2311), (0.0782, 0.4522), (0.0782, 0.6733), (0.0782, 0.8944), (0.0782, 1.1156), (0.0782, 1.3367), (0.0782, 1.5578), (0.0782, 1.7789), (0.0782, 2), (0.1559, 0), (0.1559, 0.2311), (0.1559, 0.4522), (0.1559, 0.6733), (0.1559, 0.8944), (0.1559, 1.1156), (0.1559, 1.3367), (0.1559, 1.5578), (0.1559, 1.7789), (0.1559, 2), (0.2337, 0), (0.2337, 0.2311), (0.2337, 0.4522), (0.2337, 0.6733), (0.2337, 0.8944), (0.2337, 1.1156), (0.2337, 1.3367), (0.2337, 1.5578), (0.2337, 1.7789), (0.2337, 2), (0.3114, 0), (0.3114, 0.2311), (0.3114, 0.4522), (0.3114, 0.6733), (0.3114, 0.8944), (0.3114, 1.1156), (0.3114, 1.3367), (0.3114, 1.5578), (0.3114, 1.7789), (0.3114, 2), (0.3891, 0), (0.3891, 0.2311), (0.3891, 0.4522), (0.3891, 0.6733), (0.3891, 0.8944), (0.3891, 1.1156), (0.3891, 1.3367), (0.3891, 1.5578), (0.3891, 1.7789), (0.3891, 2), (0.4668, 0), (0.4668, 0.2311), (0.4668, 0.4522), (0.4668, 0.6733), (0.4668, 0.8944), (0.4668, 1.1156), (0.4668, 1.3367), (0.4668, 1.5578), (0.4668, 1.7789), (0.4668, 2), (0.5446, 0), (0.5446, 0.2311), (0.5446, 0.4522), (0.5446, 0.6733), (0.5446, 0.8944), (0.5446, 1.1156), (0.5446, 1.3367), (0.5446, 1.5578), (0.5446, 1.7789), (0.5446, 2), (0.6223, 0), (0.6223, 0.2311), (0.6223, 0.4522), (0.6223, 0.6733), (0.6223, 0.8944), (0.6223, 1.1156), (0.6223, 1.3367), (0.6223, 1.5578), (0.6223, 1.7789), (0.6223, 2), (0.7, 0), (0.7, 0.2311), (0.7, 0.4522), (0.7, 0.6733), (0.7, 0.8944), (0.7, 1.1156), (0.7, 1.3367), (0.7, 1.5578), (0.7, 1.7789), (0.7, 2)]



# globals for tracking which command indices have finished
completed_indices = []
completed_lock = threading.Lock()

def build_command(task, case, theta, beta, tau_0, constant_theta, every_second=False):
    cmd = [
        EXE_PATH,
        "--task", task,
        "--case", case,
        "--theta", str(theta),
        "--beta", str(beta),
        "--tau-0", str(tau_0)
    ]
    if constant_theta:
        cmd.append("--constant-theta")
    if every_second:
        cmd.append("--every-second")
    return cmd


def run_process(cmd, index, total):
    logger.info(f"[{index}/{total}] Start process: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        logger.info(f"[{index}/{total}] Done: {' '.join(cmd)}, returncode: {result.returncode}")
        with completed_lock:
            completed_indices.append(index)
        if result.returncode == 3221225786:
            logger.warning(f"[{index}/{total}] Process stopped by user: {' '.join(cmd)}")
            log_finished_indices()
        if result.returncode != 0:
            logger.error(f"[{index}/{total}] Process error: {' '.join(cmd)}, stderr: {result.stderr}, stdout: {result.stdout}")
    except subprocess.TimeoutExpired:
        logger.error(f"[{index}/{total}] Process timeout: {' '.join(cmd)}")
        with completed_lock:
            completed_indices.append(index)
    except Exception as e:
        logger.error(f"[{index}/{total}] Process exception: {' '.join(cmd)}, {str(e)}")
        with completed_lock:
            completed_indices.append(index)


def log_finished_indices():
    # log which indices finished
    with completed_lock:
        sorted_done = sorted(completed_indices)
    logger.info(f"Completed indices: {sorted_done}")
    logger.info("All done")


def main():
    if not os.path.exists(os.path.join(os.getcwd(), 'results')):
        logger.info("`results` directory doesn't exist, creating...")
        os.mkdir(os.path.join(os.getcwd(), 'results'))
        time.sleep(0.2)

    logger.info(f'MAX_CPU: {MAX_CPU}')

    commands = []

    # Для tau_0
    for tau in SF_tau_0:
        commands.append(build_command("santa", "SF-TAU", 0.12, 0.5, tau, True))
    for tau in NARMA_tau_0:
        commands.append(build_command("narma", "NARMA-TAU", 0.12, 0.5, tau, True))
    for tau in MC_tau_0:
        commands.append(build_command("mc", "MC-TAU", 0.12, 0.5, tau, True))

    # Для thetas
    for theta in SF_thetas:
        commands.append(build_command("santa", "SF-THETA", theta, 0.5, 1.0, False))
    for theta in NARMA_thetas:
        commands.append(build_command("narma", "NARMA-THETA", theta, 0.5, 1.0, False))
    for theta in MC_thetas:
        commands.append(build_command("mc", "MC-THETA", theta, 0.5, 1.0, False))

    # Для beta_theta
    for beta, theta in beta_theta:
        commands.append(build_command("mc", "MC-2D", theta, beta, 1.0, False))
        commands.append(build_command("santa", "SF-2D", theta, beta, 1.0, False))
        commands.append(build_command("narma", "NARMA-2D", theta, beta, 1.0, False))

    logger.info(f"Total commands: {len(commands)}")

    with ThreadPoolExecutor(max_workers=MAX_CPU) as executor:
        futures = []
        for i in range(START_AT, len(commands)):
            cmd = commands[i]
            if len(futures) >= MAX_CPU:
                # Ждем завершения одного
                for future in as_completed(futures[:1]):
                    futures.remove(future)
            future = executor.submit(run_process, cmd, i, len(commands))
            futures.append(future)

        # Ждем завершения всех оставшихся
        for future in as_completed(futures):
            pass

    log_finished_indices()

if __name__ == "__main__":
    main()