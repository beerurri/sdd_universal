import numpy as np
import subprocess
import time
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
import platform
from params import MC_tau_0, SF_tau_0, NARMA_tau_0, MC_thetas, SF_thetas, NARMA_thetas, theta_beta

# Настройки
if platform.system() == 'Windows':
    EXE_PATH = os.path.join(os.getcwd(), 'target', 'release', 'v27_sdd_universal.exe')  #r"C:\Users\Ivan\Desktop\v27_sdd_universal\target\release\v27_sdd_universal.exe"
elif platform.system() == 'Linux':
    EXE_PATH = os.path.join(os.getcwd(), 'target', 'release', 'v27_sdd_universal') #r'/home/ivan/sdd_universal/target/release/v27_sdd_universal'

MAX_CPU = max(1, (os.cpu_count() - 2))
LOG_FILE = "run_scheduler.log"
START_AT = 267

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

# nohup target/release/v27_sdd_universal --task mc --case MC-TAU --theta 0.12 --beta 0.5 --tau-0 50 --constant-theta


def run_process(cmd, index, total):
    logger.info(f"[{index}/{total}] Start process: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            logger.info(f"[{index}/{total}] Done: {' '.join(cmd)}, returncode: {result.returncode}")
            with completed_lock:
                completed_indices.append(index)
            return

        if result.returncode == 3221225786:
            logger.warning(f"[{index}/{total}] Process stopped by user: {' '.join(cmd)}")
            log_finished_indices()
            return

        if result.returncode != 0:
            logger.error(f"[{index}/{total}] Process error: {' '.join(cmd)}, returncode: {result.returncode}, stderr:\n{result.stderr}\n, stdout:\n{result.stdout}\n")

    except subprocess.TimeoutExpired:
        logger.error(f"[{index}/{total}] Process timeout: {' '.join(cmd)}")

    except Exception as e:
        logger.error(f"[{index}/{total}] Process exception: {' '.join(cmd)},\n{str(e)}")


def log_finished_indices():
    # log which indices finished
    with completed_lock:
        sorted_done = sorted(completed_indices)

    logger.info(f"Completed indices: {sorted_done}")
    logger.info("All done")


def make_commands():
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

    # Для theta_beta
    for theta, beta in theta_beta:
        commands.append(build_command("santa", "SF-2D", theta, beta, 1.0, False))
        commands.append(build_command("narma", "NARMA-2D", theta, beta, 1.0, False))

    for theta, beta in theta_beta:
        commands.append(build_command("mc", "MC-2D", theta, beta, 1.0, False))

    return commands


def main():
    if not os.path.exists(os.path.join(os.getcwd(), 'results')):
        logger.info("`results` directory doesn't exist, creating...")
        os.mkdir(os.path.join(os.getcwd(), 'results'))
        time.sleep(0.2)

    logger.info(f'MAX_CPU: {MAX_CPU}')

    commands = make_commands()

    logger.info(f"Total commands: {len(commands)}")

    with ThreadPoolExecutor(max_workers=MAX_CPU) as executor:
        futures = set()
        # iterate through command indices and submit jobs, keeping at most MAX_CPU in flight
        for i in range(START_AT, len(commands)):
            cmd = commands[i]
            # if we've hit the limit, wait for at least one to finish
            if len(futures) >= MAX_CPU:
                # wait until any future completes, then remove it from the set
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                futures.difference_update(done)
            future = executor.submit(run_process, cmd, i, len(commands))
            futures.add(future)

        # Ждем завершения всех оставшихся
        if futures:
            for future in as_completed(futures):
                pass

    log_finished_indices()

if __name__ == "__main__":
    main()
