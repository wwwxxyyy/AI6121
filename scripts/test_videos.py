#!/usr/bin/env python3
"""Run the project's only two tests: complete reconstruction of both demo videos."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ("hydrant", "Hydrant", 3000, 33, ["--rectify-alpha", "-1", "--disparities", "224"]),
    ("garden", "Yunnan Garden", 6000, 61, []),
)


def main():
    outputs = ROOT / "outputs"
    outputs.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="video_tests_", dir=outputs))
    env = dict(os.environ, MPLBACKEND="Agg", OPENBLAS_NUM_THREADS="1")
    report = {"output": str(output), "tests": []}
    print(f"Video test results: {output}", flush=True)
    for name, title, nfeatures, expected_frames, stereo_options in CASES:
        folder = output / name
        start = time.perf_counter()
        result = {"video": name, "passed": False, "stage_seconds": {}}
        stages = (
            ("sfm", ["run_sfm.py", "--video", f"demos/{name}/input.mp4",
                     "--fps", "2", "--max-frames", "0", "--refine-focal",
                     "--nfeatures", str(nfeatures), "--output", str(folder)]),
            ("verify", ["scripts/verify_reconstruction.py", str(folder)]),
            ("stereo", ["scripts/stereo_cloud.py", str(folder), *stereo_options]),
            ("preview", ["scripts/render_cloud_views.py", str(folder), "--title", title, "--stereo"]),
        )
        try:
            for stage, arguments in stages:
                log_path = output / f"{name}_{stage}.log"
                print(f"[{name}] {stage} (log: {log_path.name})", flush=True)
                step_start = time.perf_counter()
                with log_path.open("w") as log:
                    subprocess.run([sys.executable, *arguments], cwd=ROOT, env=env,
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
                result["stage_seconds"][stage] = time.perf_counter() - step_start
            metrics = json.loads((folder / "metrics.json").read_text())
            history = json.loads((folder / "ba_history.json").read_text())
            if not (metrics["accepted"] and metrics["sampled_frames"] == expected_frames
                    and metrics["registered_frames"] == expected_frames
                    and metrics["all_sampled_frames_used"] and history
                    and all(ba["converged"] and ba["nfev"] <= 100 for ba in history)):
                raise RuntimeError("Incomplete frame participation or BA/geometry acceptance failed")
            result.update(passed=True, registered_frames=metrics["registered_frames"],
                          sparse_points=metrics["points"],
                          mean_reprojection_error_px=metrics["final_reprojection_error_px"]["mean"])
            print(f"[{name}] PASS: {expected_frames}/{expected_frames} frames, "
                  f"{metrics['points']} sparse points", flush=True)
        except (subprocess.CalledProcessError, OSError, ValueError, KeyError, RuntimeError) as exc:
            result["error"] = str(exc)
            print(f"[{name}] FAIL: {exc}; see logs in {output}", file=sys.stderr, flush=True)
        result["elapsed_seconds"] = time.perf_counter() - start
        report["tests"].append(result)
        (output / "test_results.json").write_text(json.dumps(report, indent=2) + "\n")
    passed = sum(case["passed"] for case in report["tests"])
    print(f"{passed}/{len(CASES)} video tests passed. Report: {output / 'test_results.json'}", flush=True)
    return 0 if passed == len(CASES) else 2


if __name__ == "__main__":
    raise SystemExit(main())
