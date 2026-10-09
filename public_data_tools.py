"""Generic checks for aggregated weekly case tables; no model implementation."""
import argparse
from datetime import date
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


REGION_SHEETS = {
    "NSW": "NSW", "Vic": "Vic", "Qld": "QLD", "SA": "SA",
    "WA": "WA", "Tas": "TAS", "NT": "NT",
}
REQUIRED_COLUMNS = {"region", "year_week", "infection_count", "strain_type"}


def iso_week_monday(value):
    """Validate a six-digit ISO year-week code and return its Monday."""
    code = str(value).strip()
    if not re.fullmatch(r"\d{6}(?:\.0+)?", code):
        raise ValueError(f"Invalid ISO year-week code: {value!r}")
    code = code.split(".", 1)[0]
    return pd.Timestamp(date.fromisocalendar(int(code[:4]), int(code[4:]), 1))


def read_aggregated_cases(path):
    """Read a CSV or region-sheet Excel workbook without modifying observations."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        if "region" not in frame and "state" in frame:
            frame = frame.rename(columns={"state": "region"})
    elif path.suffix.lower() == ".xlsx":
        parts = []
        with pd.ExcelFile(path) as workbook:
            for sheet, region in REGION_SHEETS.items():
                if sheet in workbook.sheet_names:
                    part = pd.read_excel(workbook, sheet_name=sheet)
                    part["region"] = region
                    parts.append(part)
        if not parts:
            raise ValueError("No supported region worksheets were found")
        frame = pd.concat(parts, ignore_index=True)
    else:
        raise ValueError("Supported input formats are .csv and .xlsx")
    return validate_cases(frame)


def validate_cases(frame):
    """Check keys and counts; do not impute, reallocate strains, or merge weather."""
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    result = frame.loc[:, ["region", "year_week", "infection_count", "strain_type"]].copy()
    if result.empty:
        raise ValueError("The input table is empty")
    if result.isna().any().any():
        raise ValueError("Required fields contain missing values")
    for column in ["region", "strain_type"]:
        result[column] = result[column].astype(str).str.strip()
        if result[column].eq("").any():
            raise ValueError(f"Empty values in {column}")
    result["date"] = result["year_week"].map(iso_week_monday)
    result["infection_count"] = pd.to_numeric(result["infection_count"], errors="raise")
    counts = result["infection_count"].to_numpy(dtype=float)
    if not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("Counts must be finite and nonnegative")
    if result.duplicated(["region", "date", "strain_type"]).any():
        raise ValueError("Duplicate region-week-strain keys")
    return result.sort_values(["region", "date", "strain_type"]).reset_index(drop=True)


def summarize_cases(frame):
    """Return descriptive summaries of the supplied rows, without filling gaps."""
    return {
        "rows": int(len(frame)),
        "regions": sorted(frame["region"].unique().tolist()),
        "strains": sorted(frame["strain_type"].unique().tolist()),
        "first_week_monday": frame["date"].min().date().isoformat(),
        "last_week_monday": frame["date"].max().date().isoformat(),
        "distinct_weeks": int(frame["date"].nunique()),
        "zero_count_rows": int(frame["infection_count"].eq(0).sum()),
        "total_counts_in_supplied_rows": float(frame["infection_count"].sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Aggregated case CSV or Excel")
    parser.add_argument("--output", type=Path, help="Optional JSON summary destination")
    args = parser.parse_args()
    summary = summarize_cases(read_aggregated_cases(args.input))
    serialized = json.dumps(summary, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
