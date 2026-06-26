import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
PIPELINE_SCRIPT = PROJECT_DIR / "run_pipeline_once.py"
INTERVAL_SECONDS = 5 * 60


def current_local_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_pipeline() -> int:
    result = subprocess.run(
        [sys.executable, str(PIPELINE_SCRIPT)],
        cwd=PROJECT_DIR,
    )
    return result.returncode


def main() -> None:
    print("=== Five Solo Scheduler ===", flush=True)

    try:
        while True:
            print(f"\nCurrent local time: {current_local_time()}", flush=True)

            exit_code = run_pipeline()
            if exit_code != 0:
                print(
                    f"오류: run_pipeline_once.py 실행에 실패했습니다. "
                    f"종료 코드: {exit_code}",
                    flush=True,
                )
                print("다음 5분 주기까지 계속 대기합니다.", flush=True)

            print("\nWaiting for next 5-minute cycle...", flush=True)
            time.sleep(INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\nScheduler stopped by user.", flush=True)


if __name__ == "__main__":
    main()
