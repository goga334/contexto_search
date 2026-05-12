from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_cli_smoke(tmp_path: Path) -> None:
    out_json = tmp_path / "benchmark.json"
    out_summary = tmp_path / "summary.csv"
    out_traces = tmp_path / "traces.csv"
    command = [
        sys.executable,
        "-m",
        "rgsn.benchmark",
        "--embeddings",
        str(ROOT / "tests" / "fixtures" / "tiny_words.vec"),
        "--targets",
        str(ROOT / "tests" / "fixtures" / "tiny_targets.txt"),
        "--strategies",
        "random,best_neighbor",
        "--seed-words",
        "road,tree,water",
        "--budget",
        "4",
        "--random-seed",
        "5",
        "--out-json",
        str(out_json),
        "--out-summary-csv",
        str(out_summary),
        "--out-traces-csv",
        str(out_traces),
    ]

    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)

    assert "best_neighbor" in result.stdout
    assert out_json.exists()
    assert out_summary.exists()
    assert out_traces.exists()
