# SIKR study: limited supporting materials

## Code availability

The core implementation is subject to a confidentiality period required by the funding body in connection with ongoing patent and intellectual-property matters. It is not included in this release. Publication does not automatically authorize release of the complete implementation; further disclosure requires the relevant approvals.

This package contains only generic data-validation and evaluation utilities. It does not include the model architecture, training procedure, parameter-generation network, Kalman teacher, dynamical calibration or control solvers, trained weights, or fitted parameter files. It is not a complete reproduction package for the reported forecasting and control experiments.

## Included files

~~~text
README.md
DATA_SOURCES.md
public_data_tools.py
evaluate_predictions.py
requirements.txt
examples/
    example_weekly_cases.csv
    example_predictions.csv
~~~

- public_data_tools.py checks the structure of aggregated weekly case tables and produces a descriptive summary.
- evaluate_predictions.py computes standard metrics from observation/prediction pairs supplied by the user. It does not train a model or generate predictions.
- DATA_SOURCES.md identifies official public data portals.
- Both example CSV files contain artificial demonstration values, explicitly marked synthetic_demo. They are not public surveillance records, paper predictions, or evidence for the study's results.

## Requirements and quick start

Use Python 3.9 or later. Install the dependencies and run from this directory:

~~~bash
python -m pip install -r requirements.txt
python public_data_tools.py --input examples/example_weekly_cases.csv
python evaluate_predictions.py --input examples/example_predictions.csv --by region --split test
~~~

Optional output files:

~~~bash
python public_data_tools.py --input examples/example_weekly_cases.csv --output outputs/data_summary.json
python evaluate_predictions.py --input examples/example_predictions.csv --by strain --split test --output outputs/metrics.csv
~~~

The examples exercise file reading and metric calculation only. Their output must not be reported as experimental performance.

## Case-table interface

CSV input requires the columns region, year_week, infection_count, and strain_type. A state column can be used instead of region. year_week is a six-digit ISO year-week code, such as 201301. Counts may be fractional but must be finite and nonnegative.

Excel input is supported for aggregated tables with region worksheets NSW, Vic, Qld, SA, WA, Tas, and NT. Each worksheet must contain year_week, infection_count, and strain_type. ACT is not selected by this helper.

The helper rejects missing required fields, invalid ISO weeks, negative/non-finite counts, and duplicate region-week-strain keys. It does not fill missing observations, redistribute untyped infections, combine weather variables, normalize features, construct graphs, or define experimental splits.

## Prediction-table interface and metrics

Input requires y_true and y_pred, or the aliases observed and predicted. Optional region, strain, split, and other grouping columns are retained for evaluation. Use --by to choose one grouping column and --split to select one split. A table containing multiple splits must be filtered explicitly; training and test records are not silently combined.

Metrics are MAE, MSE, RMSE, R2, NRMSE_range, SMAPE_%, ND, MAPE_%_eps1, and PCC:

- NRMSE_range is RMSE divided by the observed range, with a small denominator stabilizer.
- ND is total absolute error divided by total absolute observed counts.
- SMAPE is reported as a percentage, using the symmetric absolute-count denominator.
- MAPE is reported as a percentage with a minimum observed-count denominator of one.
- R2 and PCC are undefined for the corresponding constant-series cases and are reported as missing values.

Every supplied row is evaluated separately. If overlapping forecast windows are present, the same calendar date may contribute at multiple lead times. Filter the desired horizon before calling the utility when evaluating a particular lead time.

The printed elapsed time measures CSV reading and metric evaluation only. It is not model training time or forecasting time.

## Data access

Official public sources are linked in DATA_SOURCES.md. The study's local processed workbooks, weather extracts, prediction outputs, and calibration results are not bundled here. A current official extract is not necessarily identical to the historical extract used in the study.

Use source-provider terms and attribution requirements when obtaining or redistributing data. This release does not grant additional rights over third-party datasets or the restricted core implementation.
