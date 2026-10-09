# SIKR-PIGNN
Multi-strain metapopulation transmission dynamics and physics-informed graph forecasting of regional influenza

# Multi-strain SIKR forecasting and control analysis

This directory contains two independent notebooks for seven Australian regions and three influenza categories: A(H3N2), A(H1N1), and B. All notebook explanations, comments, and runtime messages are in English. Input and output locations are resolved relative to the project folder, without machine-specific absolute paths.

## Repository layout

Keep the notebooks and their input workbooks together:

```text
your-repository/
├── README.md
├── Main_Forecast_SIKR.ipynb
├── Main_Control_SIKR.ipynb
├── FinalDataAustralia2013-2017.xlsx
└── 3DataAustraliaNoAgeNoSex.xlsx
```

The forecasting workbook already present in the source directory is byte-for-byte identical to the workbook previously named `FinalDataAustralia2013-2017_filtered.xlsx`. The shorter, co-located filename is used here without changing the input data. Neither input workbook is modified by either notebook.

The original `Main_TimeVarying_SIKR.ipynb` and `Main_Control.ipynb` are preserved. The new copies have cleared execution outputs and execution counts. Model calculations and numerical settings are unchanged. 

## Environment and execution

The notebooks use Python, NumPy, pandas, openpyxl, SciPy, Matplotlib, PyTorch, and IPython. They can run on CPU; the forecasting notebook uses CUDA automatically when available. The local verification environment is Python 3.9.6, PyTorch 2.5.0+cpu, pandas 1.3.2, and SciPy 1.7.1.

Create an environment and install the dependencies:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS instead:
# source .venv/bin/activate
python -m pip install numpy pandas openpyxl scipy matplotlib torch ipython ipykernel jupyterlab nbconvert
python -m ipykernel install --user --name sikr --display-name "Python (SIKR)"
jupyter lab
```

Run these commands from the cloned repository folder. Open a notebook, select the Python (SIKR) kernel, and use **Restart Kernel and Run All Cells**. The two notebooks can be run independently and do not need each other's saved outputs.

The path resolver looks for the required workbook in the kernel working directory, its parent directories, or the script directory when `__file__` is available. If Jupyter was launched elsewhere, set the optional environment variable `SIKR_PROJECT_DIR` to the folder containing the notebook and workbook before starting Jupyter. No environment variable is needed when the kernel starts in the repository folder.

For headless execution from that folder:

```bash
jupyter nbconvert --to notebook --execute Main_Forecast_SIKR.ipynb --output Main_Forecast_SIKR_executed.ipynb --ExecutePreprocessor.timeout=-1
jupyter nbconvert --to notebook --execute Main_Control_SIKR.ipynb --output Main_Control_SIKR_executed.ipynb --ExecutePreprocessor.timeout=-1
```

These commands create executed copies rather than overwriting the source notebooks. Running all forecasting cells starts the full configured training workload; it is not a quick smoke test.

## Input schema

Both workbooks use long-format worksheets named `NSW`, `Vic`, `Qld`, `SA`, `WA`, `Tas`, and `NT`. Internal region order is `NSW, Vic, QLD, SA, WA, TAS, NT`. ACT is not loaded.

| Column | Meaning | Forecasting | Control calibration |
|---|---|---|---|
| `year_week` | ISO year-week code, for example `201301` | Required | Required |
| `infection_count` | Weekly count for a region and category | Required | Required |
| `strain_type` | `A(H3N2)`, `A(H1N1)`, or `B` | Required | Required |
| `tp_m_day` | Precipitation feature in metres/day | Required | Not used |
| `t2m_K` | Air-temperature feature in kelvin | Required | Not used |
| `d2m_K` | Dew-point-temperature feature in kelvin | Required | Not used |

Duplicate week-category keys within a region raise an error. Missing region-week-category case combinations are filled with zero, as in the source notebooks. Forecasting weather values are aggregated by region-week and interpolated over time. Train-only normalization is preserved; weather interpolation is performed before the split, as in the source implementation.

## 1. Main_Forecast_SIKR.ipynb

### Purpose and model

The forecasting notebook uses `FinalDataAustralia2013-2017.xlsx` (2013-W01 through 2017-W52, 261 weeks). A pure neural encoder, history-dependent graph learning, graph convolution, and GRU encoding/decoding produce multi-step state predictions. A shared head generates time-varying transmission rates, and a positive trainable scalar represents the common dispersal coefficient. The objective combines data, equation-residual, parameter-smoothness, and ensemble Kalman teacher losses.

The fixed physical graph remains a symmetric complete-graph prior with unit row sums. It is separate from the learned prediction graph and is **not** replaced by the gravity graph used for control calibration. The state order is `[S, I_H3N2, I_H1N1, I_B, R]`. The implementation uses weekly notification counts directly as targets for infectious-state predictions. Susceptible and recovered states are latent, with `B_i / mu_i` used as a reference population scale. The teacher assimilates only observations available at the forecast origin, not future targets.

Among epidemiological parameters, only the transmission-rate head and `d0` are optimized. Neural weights are also optimized. Recruitment, natural mortality, immunity loss, disease mortality, recovery, and strain-specific movement factors are fixed. The relations are `dS = dR = d0` and `dI[k] = q[k] * d0`.

### Retained defaults

| Setting | Default |
|---|---:|
| History length | 12 weeks |
| Forecast horizon | 4 weeks |
| Chronological train/validation/test ratio | 3:1:1 |
| Split sizes | 156 / 52 / 53 weeks |
| Batch size | 16 |
| Hidden dimension | 32 |
| Dropout | 0.18 |
| Neural / physics learning rates | 0.001 / 0.0001 |
| Maximum epochs per run | 90 |
| Early-stopping patience | 18 |
| Independently seeded training runs | 10 |
| Equation / parameter / teacher loss settings | maximum 0.05 / 0.003 / 0.01 |
| Kalman ensemble size / batch interval | 8 / 4 |

Each forecast window has all targets inside its assigned split. Historical inputs may precede that split. The resulting train/validation/test window counts are 141/49/50. Early stopping and selection of the exported run use validation data loss. Seeds are drawn from system entropy and recorded, so separate executions need not produce identical metrics.

For a quick structural check, change only `run_training=False` and `smoke_test=True` in the new notebook's configuration and run all cells. Restore the retained defaults for full experiments.

### Forecasting outputs

New files are written to `forecast_sikr_outputs/`:

- `best_model.pt`: state dictionary of the validation-selected model.
- `config.json`, `runtime.json`: configuration and recorded timings in seconds.
- `random_seed_training_runs.csv`, `training_history.csv`: run-level information and the selected run's training history.
- `random_seed_trial_metrics.csv`, `random_seed_metric_summary.csv`: test metrics for every run and mean/standard-deviation summaries, overall and by region/category.
- `random_seed_test_runtime.csv`: test inference-plus-metric time for each run.
- `test_metrics.csv`: metrics for the validation-selected run.
- `test_predictions_and_beta.csv`: that run's test predictions, observations, transmission rates, forecast origins, target dates, and lead times.
- `test_arrays.npz`: selected test prediction, observation, transmission-rate, and origin arrays.

Reported metrics are MAE, MSE, RMSE, R2, `NRMSE_range`, `SMAPE_%`, ND, `MAPE_%_eps1`, and PCC. NRMSE is RMSE divided by the observed range; ND is total absolute error divided by total absolute observed counts. SMAPE and MAPE are percentages; MAPE uses a minimum denominator of one. Metrics flatten forecast windows, so one calendar week can contribute at several lead times. PCC and R2 are undefined for constant reference series as handled by the code. Only test-series details are exported by this notebook; it does not automatically export full training and validation trajectories.

## 2. Main_Control_SIKR.ipynb

### Purpose and calibration

The control notebook uses all available weekly data in `3DataAustraliaNoAgeNoSex.xlsx` (2008-W01 through 2022-W52). It estimates 21 constant transmission rates and one shared dispersal coefficient using conditional one-week nonlinear least squares, not a train/validation/test forecasting benchmark. Weekly counts are treated as incidence and compared with the integrated infection flow over each one-week ODE step. The state is reconstructed from the preceding observation for each conditional transition. This is not a single uninterrupted 15-year autonomous trajectory.

The fixed dynamical parameters are retained. The gravity prior uses the supplied regional populations and state-capital coordinates. Edges are retained at 2% of the maximum raw gravity weight, with each region's strongest incident edge also retained. Symmetric normalization and the Laplacian construction are unchanged. With the supplied values, there are 10 undirected edges. The fitted dispersal coefficient depends on this particular network normalization and should not be transferred directly to a differently scaled graph.

Calibration retains six deterministic initializations, 220 maximum function evaluations, transmission-rate bounds `[0.0001, 5]`, dispersal bounds `[0, 5]`, and four RK4 substeps per week. The lowest least-squares objective is selected. Local Jacobian-based intervals and boundary/conditioning diagnostics are exported.

### Scenarios

Every scenario starts from the same last fitted state and is projected for 26 weeks. The no-control baseline and the following single-factor changes are calculated:

| Scenario | Parameter change | Intensities |
|---|---|---|
| Transmission reduction | `(1 - u_beta) * beta` | 0.10, 0.20, 0.30, 0.40 |
| Increased recovery | `gamma + u_gamma` | 0.10, 0.20, 0.40, 0.80 per week |
| Reduced movement | `(1 - u_d) * d0` | 0.25, 0.50, 0.75, 1.00 |

Treatment enters both infectious outflow and recovered inflow. Movement reduction scales susceptible, infectious, and recovered dispersal consistently. Relative reductions are computed against the baseline; negative effects are retained rather than truncated.

### Control outputs

New files are written to `control_sikr_outputs/`:

- `gravity_network.csv`: regional distances, raw weights, retained edges, and normalized weights.
- `estimated_parameters.csv`, `multistart_fit_summary.csv`, `fit_metrics.csv`, `calibration_diagnostics.csv`: estimates and calibration diagnostics.
- `fit_weekly_predictions.csv`: conditional predictions across the observed period; the first row for each region/category is an initialization rather than a fitted transition.
- `control_scenario_summary.csv`: 26-week baseline/scenario cumulative incidence and relative effects.
- `control_weekly_trajectories.csv`: regional and category-specific weekly incidence and cumulative trajectories.
- `calibration_and_control_arrays.npz`: numerical arrays for further analysis.
- `control_measures_3x4_gravity.png` and `.pdf`: three controls by total/category cumulative incidence.
- `mobility_restriction_by_region.png` and `.pdf`: seven regional movement-restriction comparisons.

## File preservation and reproducibility

Neither new notebook reads results from the old notebooks or changes the input workbooks. Existing `tuned_outputs` and `control_outputs_gravity` results are untouched. Re-running a new notebook overwrites matching filenames only inside its own new output directory; archive that directory before a run if its previous results must be retained.

Saved source-notebook results and the historical comparison table are not newly reproduced experiment results. Forecasting and autonomous control calibration use different observation operators and physical networks, and their fitted parameters should not be presented as interchangeable.
