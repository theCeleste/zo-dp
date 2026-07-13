import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def load_records(root):
    records = []
    for benchmark_file in root.rglob("benchmark.json"):
        benchmark = json.loads(benchmark_file.read_text(encoding="utf-8"))
        metrics_file = benchmark_file.parent / "eval_metrics.json"
        evaluation = json.loads(metrics_file.read_text(encoding="utf-8")) if metrics_file.exists() else {}
        gpu_peaks = [gpu["peak_reserved_bytes"] for gpu in benchmark["environment"]["gpus"]]
        records.append({
            "suite": benchmark["experiment_suite"],
            "experiment_id": benchmark["experiment_id"],
            "method": benchmark["method"],
            "mode": benchmark["mode"],
            "seed": benchmark["seed"],
            "epsilon": (benchmark.get("privacy") or {}).get("epsilon"),
            "eval_metric": next(iter(evaluation.values()), None),
            "training_wall_seconds": benchmark["training_wall_seconds"],
            "train_steps_per_second": benchmark["trainer_metrics"].get("train_steps_per_second"),
            "train_samples_per_second": benchmark["trainer_metrics"].get("train_samples_per_second"),
            "peak_reserved_bytes_max_gpu": max(gpu_peaks) if gpu_peaks else None,
            "trainable_parameters": benchmark["parameters"]["trainable"],
            "benchmark_file": str(benchmark_file),
        })
    return records


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def summarize(records):
    groups = defaultdict(list)
    for record in records:
        groups[(record["suite"], record["method"], record["mode"], record["epsilon"])].append(record)
    result = []
    for key, rows in sorted(groups.items(), key=lambda item: str(item[0])):
        suite, method, mode, epsilon = key
        result.append({
            "suite": suite,
            "method": method,
            "mode": mode,
            "epsilon": epsilon,
            "runs": len(rows),
            "mean_eval_metric": mean([row["eval_metric"] for row in rows]),
            "mean_training_wall_seconds": mean([row["training_wall_seconds"] for row in rows]),
            "mean_train_steps_per_second": mean([row["train_steps_per_second"] for row in rows]),
            "mean_peak_reserved_bytes_max_gpu": mean([row["peak_reserved_bytes_max_gpu"] for row in rows]),
        })
    return result


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("result/formal"))
    parser.add_argument("--output", type=Path, default=Path("result/formal/summary"))
    args = parser.parse_args()
    records = load_records(args.root)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "runs.csv", records)
    grouped = summarize(records)
    write_csv(args.output / "groups.csv", grouped)
    (args.output / "groups.json").write_text(json.dumps(grouped, indent=2), encoding="utf-8")
    print(f"Summarized {len(records)} runs into {len(grouped)} groups at {args.output}")


if __name__ == "__main__":
    main()
