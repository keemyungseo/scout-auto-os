import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

STEPS = [
    ("select_hold_symbols.py", "Step 1: HOLD 종목 선정"),
    ("check_hold_signals.py", "Step 2: HOLD 신호 확인"),
    ("virtual_trader.py", "Step 3: 가상 진입 확인"),
    ("virtual_position_manager.py", "Step 4: 가상 포지션 관리"),
    ("virtual_exit_manager.py", "Step 5: 가상 청산 확인"),
]


def run_step(script_name: str, step_message: str) -> int:
    script_path = PROJECT_DIR / script_name
    print(f"\n=== {step_message} ===", flush=True)
    print(f"실행: {script_name}", flush=True)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_DIR,
    )
    return result.returncode


def main() -> None:
    print("로컬 가상 거래 파이프라인을 시작합니다.", flush=True)

    for script_name, step_message in STEPS:
        exit_code = run_step(script_name, step_message)

        if exit_code != 0:
            print(
                f"\n오류: {script_name} 실행에 실패했습니다. "
                f"종료 코드: {exit_code}",
                flush=True,
            )
            print("파이프라인을 중단합니다.", flush=True)
            sys.exit(exit_code)

    print("\n전체 파이프라인 1회 실행 완료", flush=True)


if __name__ == "__main__":
    main()
