"""Export small, public-facing results without publishing data or checkpoints."""
import argparse
import csv
import io
import json
import math
import re
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Export a completed Darcy evaluation")
    parser.add_argument("--result-dir", type=Path,
                        default=root / "outputs/darcy_test_seed2026_01")
    parser.add_argument("--report-name", default="darcy_test_seed2026_01")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.report_name):
        raise ValueError("Report names may contain only letters, digits, underscores and hyphens.")

    source = args.result_dir.resolve()
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    if summary.get("completed") is not True or summary.get("split") != "test":
        raise ValueError("A completed test evaluation is required.")
    rows = summary["metrics"]
    if len(rows) != 2 or {row["model"] for row in rows} != {"DeepONet", "FNO"}:
        raise ValueError("Expected one DeepONet row and one FNO row.")
    n_test = rows[0]["n_test"]
    if type(n_test) is not int or n_test < 1:
        raise ValueError("Expected a positive test sample count.")
    for row in rows:
        if row["n_test"] != n_test:
            raise ValueError("The model rows use different test sample counts.")
        for key in ("test_relative_l2_mean_percent", "test_max_absolute_error_mean",
                    "test_max_absolute_error_worst"):
            if not math.isfinite(row[key]) or row[key] < 0:
                raise ValueError(f"Invalid metric: {key}")

    comparison = (source / "comparison.csv").read_bytes()
    csv_rows = list(csv.DictReader(io.StringIO(comparison.decode("utf-8"))))
    if len(csv_rows) != 2:
        raise ValueError("Unexpected comparison.csv row count.")
    for row in rows:
        matches = [saved for saved in csv_rows if saved["model"] == row["model"]]
        if len(matches) != 1:
            raise ValueError("CSV and JSON model names disagree.")
        for key, value in row.items():
            saved = matches[0][key]
            if (isinstance(value, str) and saved != value) or (
                not isinstance(value, str) and float(saved) != value
            ):
                raise ValueError(f"CSV and JSON disagree: {row['model']}, {key}")

    # Select fields explicitly; do not publish machine-specific absolute paths.
    public = {key: summary[key] for key in (
        "evaluated_at_utc", "dataset_sha256", "split", "grid", "metric_domain",
        "solution_scale", "relative_l2_denominator_epsilon", "std_note",
        "figure_sample_indices", "metrics", "evaluation_environment",
        "evaluation_code_sha256",
    )}
    for key in ("task", "input_field", "reference_method", "dataset_metadata"):
        if key in summary:
            public[key] = summary[key]
    public["checkpoints"] = {
        name: {key: info[key] for key in (
            "checkpoint_sha256", "model_config", "normalization", "training_config",
            "model_init_seed", "training_environment", "parameter_real_components",
        )}
        for name, info in summary["checkpoints"].items()
    }
    result_rel = Path("docs/results") / args.report_name
    figure_rel = Path("docs/figures") / args.report_name
    contents = {
        root / result_rel / "comparison.csv": comparison,
        root / result_rel / "per_sample.csv": (source / "per_sample.csv").read_bytes(),
        root / result_rel / "metadata.json": (
            json.dumps(public, indent=2, allow_nan=False) + "\n"
        ).encode("utf-8"),
    }
    indices = summary["figure_sample_indices"]
    if (not isinstance(indices, list) or len(indices) != len(set(indices))
            or any(type(i) is not int or not 0 <= i < n_test for i in indices)):
        raise ValueError("Invalid preselected figure sample indices.")
    for index in indices:
        name = f"sample_{index:03d}.png"
        contents[root / figure_rel / name] = (source / name).read_bytes()

    # Preflight every destination before creating any output file.
    for path, content in contents.items():
        if path.exists() and path.read_bytes() != content:
            raise FileExistsError(f"Refusing to replace a different report file: {path}")
    for path, content in contents.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open("xb") as file:
                file.write(content)

    print("Public result files exported. Existing identical files were preserved.")
    print("Copy the Markdown section below into the repository-root README.md.\n")
    print("## Darcy baseline test results\n")
    print(f"Both validation-selected checkpoints were evaluated on the same {n_test} held-out test samples.")
    print("No training or checkpoint selection was performed using the test set.\n")
    print("| Model | Selected update | Mean test relative L2 (%) | Mean sample maximum absolute error | Worst absolute error |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for row in sorted(rows, key=lambda value: value["model"]):
        print(f"| {row['model']} | {row['best_step']} | "
              f"{row['test_relative_l2_mean_percent']:.4f} | "
              f"{row['test_max_absolute_error_mean']:.6e} | "
              f"{row['test_max_absolute_error_worst']:.6e} |")
    print(f"\nRelative L2 is computed per sample and then averaged over {n_test} samples.")
    print("Absolute errors are measured after restoring the physical solution scale.")
    print("All grid points, including boundary points, are included. The worst absolute")
    print("error is the maximum over all test samples and grid points.\n")
    print(f"- [Full metric table]({result_rel.as_posix()}/comparison.csv)")
    print(f"- [Per-sample errors]({result_rel.as_posix()}/per_sample.csv)")
    print(f"- [Recorded settings, environments and fingerprints]({result_rel.as_posix()}/metadata.json)")
    print("\n### Prediction examples\n")
    print(f"Preselected figure sample indices: {indices}.")
    print("Within each figure, the reference and predictions share a color scale;")
    print("the two absolute-error panels also share a color scale.\n")
    for index in indices:
        print(f"![Darcy test sample {index}]({figure_rel.as_posix()}/sample_{index:03d}.png)\n")
    print("### Interpretation and limitations\n")
    if summary.get("task", "a_to_u") == "a_to_u":
        print("These results describe the binary-coefficient, fixed-source Darcy task,")
    else:
        print("These results describe the fixed-coefficient, varying-source Darcy task,")
    print("with one training initialization seed per model. They do not establish universal")
    print("model superiority. Test-sample variability is not variability across training seeds.")
    print("Reference-solution accuracy must be assessed for the selected task and grid.")


if __name__ == "__main__":
    main()
