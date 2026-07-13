import argparse
import hashlib
import itertools
import json
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


LLAMA_DIR = Path(__file__).resolve().parent


def load_config(path):
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("version") != 1:
        raise ValueError(f"Unsupported experiment config version: {config.get('version')}")
    return config


def expand_suite(config, suite_name):
    suite = config["suites"][suite_name]
    common = dict(config["common"])
    common.update(suite.get("overrides", {}))
    epsilons = suite.get("dp_epsilons", [None])
    jobs = []
    for mode, seed, epsilon in itertools.product(suite["modes"], suite["seeds"], epsilons):
        method = suite["method"]
        if method != "dpzero" and epsilon is not None:
            raise ValueError(f"Suite {suite_name} sets epsilon for non-DP method {method}")
        epsilon_tag = f"-eps{epsilon:g}" if epsilon is not None else ""
        experiment_id = f"{suite_name}-{method}-{mode}{epsilon_tag}-seed{seed}"
        output_dir = Path(config["output_root"]) / suite_name / experiment_id
        jobs.append({
            "suite": suite_name,
            "experiment_id": experiment_id,
            "description": suite["description"],
            "privacy_accounting_note": config.get("privacy_accounting_note"),
            "model_name": config["model_name"],
            "method": method,
            "mode": mode,
            "seed": seed,
            "dp_epsilon": epsilon,
            "output_dir": output_dir,
            "common": common,
            "mode_config": config["modes"][mode],
        })
    return jobs


def command_for(job):
    common = job["common"]
    command = [
        sys.executable,
        "run.py",
        "--model_name", job["model_name"],
        "--task_name", common["task_name"],
        "--output_dir", str(job["output_dir"]),
        "--result_file", str(job["output_dir"] / "eval_metrics.json"),
        "--experiment_suite", job["suite"],
        "--experiment_id", job["experiment_id"],
        "--tag", job["experiment_id"],
        "--seed", str(job["seed"]),
        "--train_set_seed", str(job["seed"]),
        "--num_train", str(common["num_train"]),
        "--num_dev", str(common["num_dev"]),
        "--num_eval", str(common["num_eval"]),
        "--max_length", str(common["max_length"]),
        "--max_steps", str(common["max_steps"]),
        "--per_device_train_batch_size", str(common["batch_size"]),
        "--gradient_accumulation_steps", str(common["gradient_accumulation_steps"]),
        "--learning_rate", str(common["learning_rate"]),
        "--zo_eps", str(common["zo_eps"]),
        "--lr_scheduler_type", "constant",
        "--logging_steps", str(common["logging_steps"]),
        "--evaluation_strategy", "steps",
        "--eval_steps", str(common["eval_steps"]),
        "--save_strategy", "steps",
        "--save_steps", str(common["eval_steps"]),
        "--save_total_limit", "1",
        "--load_best_model_at_end",
    ]
    if common.get("load_bfloat16"):
        command.append("--load_bfloat16")
    if common.get("train_as_classification"):
        command.append("--train_as_classification")

    if job["method"] == "dpzero":
        command.extend([
            "--trainer", "zo",
            "--dpzero",
            "--dp_epsilon", str(job["dp_epsilon"]),
            "--dp_delta", str(common["dp_delta"]),
            "--dpzero_clip_threshold", str(common["dp_clip"]),
        ])
    elif job["method"] == "mezo":
        command.extend(["--trainer", "zo"])
    else:
        raise ValueError(f"Unsupported formal method: {job['method']}")

    mode_config = job["mode_config"]
    if job["mode"] == "lora":
        command.extend([
            "--lora",
            "--lora_r", str(mode_config["lora_r"]),
            "--lora_alpha", str(mode_config["lora_alpha"]),
        ])
    elif job["mode"] == "prefix":
        command.extend(["--prefix_tuning", "--num_prefix", str(mode_config["num_prefix"]), "--no_reparam"])
        if mode_config.get("prefix_init_by_real_act"):
            command.append("--prefix_init_by_real_act")
    elif job["mode"] == "head":
        command.append("--head_tuning")
    return command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=LLAMA_DIR / "configs" / "formal_experiments.yaml")
    parser.add_argument("--suite", required=True)
    parser.add_argument("--index", type=int)
    parser.add_argument("--run", action="store_true", help="Execute one selected job; default is dry-run")
    parser.add_argument("--resume", action="store_true", help="Allow an existing output directory")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.suite not in config["suites"]:
        raise ValueError(f"Unknown suite {args.suite!r}; choose from {sorted(config['suites'])}")
    jobs = expand_suite(config, args.suite)
    selected = jobs if args.index is None else [jobs[args.index]]

    print(f"Suite {args.suite}: {len(jobs)} jobs")
    for index, job in enumerate(jobs):
        marker = "*" if job in selected else " "
        print(f"{marker} [{index:02d}] {job['experiment_id']} -> {job['output_dir']}")

    if not args.run:
        if args.index is not None:
            print("Command:", shlex.join(command_for(selected[0])))
        print("Dry run only. Add --index N --run to execute exactly one job.")
        return
    if args.index is None:
        raise ValueError("Execution requires --index N; bulk execution is intentionally disabled")

    job = selected[0]
    output_dir = LLAMA_DIR / job["output_dir"]
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"Output exists: {output_dir}. Use --resume only for an intended resume.")
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = dict(job)
    snapshot["output_dir"] = str(job["output_dir"])
    snapshot["command"] = command_for(job)
    snapshot["config_sha256"] = hashlib.sha256(args.config.read_bytes()).hexdigest()
    snapshot_file = output_dir / "run_config.json"
    if snapshot_file.exists():
        existing = json.loads(snapshot_file.read_text(encoding="utf-8"))
        if existing.get("command") != snapshot["command"]:
            raise ValueError("Existing run_config.json does not match the requested resume command")
    else:
        snapshot_file.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    subprocess.run(command_for(job), cwd=LLAMA_DIR, check=True)


if __name__ == "__main__":
    main()
