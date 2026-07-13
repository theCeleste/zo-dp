import sys
from pathlib import Path


LLAMA_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LLAMA_DIR))

from run_formal_experiment import command_for, expand_suite, load_config  # noqa: E402


def main():
    config_path = LLAMA_DIR / "configs" / "formal_experiments.yaml"
    config = load_config(config_path)
    expected_counts = {
        "performance_smoke": 3,
        "mezo_baseline": 9,
        "dpzero_budget_sweep": 36,
    }
    all_ids = set()
    all_outputs = set()
    for suite, expected_count in expected_counts.items():
        jobs = expand_suite(config, suite)
        if len(jobs) != expected_count:
            raise AssertionError(f"{suite}: expected {expected_count} jobs, got {len(jobs)}")
        for job in jobs:
            if job["experiment_id"] in all_ids or str(job["output_dir"]) in all_outputs:
                raise AssertionError(f"Duplicate formal experiment identity: {job}")
            all_ids.add(job["experiment_id"])
            all_outputs.add(str(job["output_dir"]))
            command = command_for(job)
            required = (
                "--experiment_suite",
                "--experiment_id",
                "--result_file",
                "--train_set_seed",
                "--load_best_model_at_end",
            )
            if any(flag not in command for flag in required):
                raise AssertionError(f"{job['experiment_id']}: incomplete command {command}")
            if job["method"] == "dpzero" and "--dpzero" not in command:
                raise AssertionError(f"{job['experiment_id']}: missing DPZero flag")
            if job["method"] == "mezo" and "--dpzero" in command:
                raise AssertionError(f"{job['experiment_id']}: non-private baseline contains DPZero flag")
    print(f"PASS formal config: {len(all_ids)} unique jobs across {len(expected_counts)} suites")


if __name__ == "__main__":
    main()
