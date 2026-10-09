"""Standard forecast metrics for supplied predictions; no training or inference."""
import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd


def metric_dict(observed, predicted, eps=1e-8):
    """Compute count-scale metrics using the documented conventions."""
    observed = np.asarray(observed, dtype=np.float64).ravel()
    predicted = np.asarray(predicted, dtype=np.float64).ravel()
    if observed.size == 0 or observed.shape != predicted.shape:
        raise ValueError("Observed and predicted arrays must be nonempty and equally sized")
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError("Observed and predicted values must be finite")
    if (observed < 0).any():
        raise ValueError("Observed case counts must be nonnegative")
    error = predicted - observed
    mse = float(np.mean(error ** 2))
    rmse = float(np.sqrt(mse))
    observed_ss = float(np.sum((observed - observed.mean()) ** 2))
    r2 = np.nan if observed_ss < eps else float(1.0 - np.sum(error ** 2) / observed_ss)
    if np.std(observed) < eps or np.std(predicted) < eps:
        pcc = np.nan
    else:
        pcc = float(np.corrcoef(observed, predicted)[0, 1])
    return {
        "MAE": float(np.mean(np.abs(error))),
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "NRMSE_range": float(rmse / (observed.max() - observed.min() + eps)),
        "SMAPE_%": float(100.0 * np.mean(
            2.0 * np.abs(error) / (np.abs(observed) + np.abs(predicted) + eps)
        )),
        "ND": float(np.sum(np.abs(error)) / (np.sum(np.abs(observed)) + eps)),
        "MAPE_%_eps1": float(100.0 * np.mean(
            np.abs(error) / np.maximum(np.abs(observed), 1.0)
        )),
        "PCC": pcc,
    }


def evaluate_table(frame, by=None):
    """Evaluate every supplied row overall and optionally within explicit groups."""
    aliases = {}
    if "y_true" not in frame and "observed" in frame:
        aliases["observed"] = "y_true"
    if "y_pred" not in frame and "predicted" in frame:
        aliases["predicted"] = "y_pred"
    frame = frame.rename(columns=aliases).copy()
    missing = {"y_true", "y_pred"}.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if "split" in frame and frame["split"].nunique(dropna=False) > 1:
        raise ValueError("Multiple splits found; select one explicitly with --split")
    rows = [{"scope": "overall", **metric_dict(frame["y_true"], frame["y_pred"])}]
    if by:
        if by not in frame:
            raise ValueError(f"Grouping column not found: {by}")
        for value, group in frame.groupby(by, sort=False, dropna=False):
            rows.append({"scope": f"{by}={value}", **metric_dict(group["y_true"], group["y_pred"])})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="CSV with observations/predictions")
    parser.add_argument("--output", type=Path, help="Optional metric CSV destination")
    parser.add_argument("--by", help="Optional grouping column, such as region or strain")
    parser.add_argument("--split", help="Evaluate only a specified value in the split column")
    args = parser.parse_args()
    started = perf_counter()
    frame = pd.read_csv(args.input)
    if args.split is not None:
        if "split" not in frame:
            raise ValueError("The input has no split column")
        frame = frame.loc[frame["split"].astype(str).eq(args.split)]
    metrics = evaluate_table(frame, by=args.by)
    evaluation_seconds = perf_counter() - started
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(args.output, index=False)
    print(metrics.to_string(index=False))
    print(f"Evaluation-only elapsed time: {evaluation_seconds:.6f} seconds")


if __name__ == "__main__":
    main()
