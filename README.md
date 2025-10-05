# SkyHack Flight Difficulty Scoring

This repository contains an end-to-end workflow that ingests two weeks of ORD departure data, builds a daily Flight Difficulty Score, and packages the core deliverables requested in the United Airlines frontline operations challenge.

## Quick summary

- **Goal:** Highlight flights most likely to experience operational difficulty (≥15 min departure delay) so frontline teams can pre-stage resources.
- **Approach:** Enriched feature set (turn buffers, passenger mix, baggage pressure, temporal context) feeding a calibrated HistGradientBoosting model with rule-based recall boosters and SHAP explainability.
- **Impact:** Holdout recall 97% at 54% precision; illustrative cost model shows ~$626K/day savings vs. reactive operations. Artifacts include the scored dataset, insights report, visualisations, and SQL/EDA outputs.

## Project layout

```
├── data/
│   ├── raw/                # Original CSV extracts (already provided)
│   ├── interim/            # DuckDB file and any cached tables
│   └── processed/          # Feature tables (Parquet)
├── artifacts/
│   ├── figures/            # PNG visualizations generated during EDA
│   ├── tables/             # Aggregated CSV / JSON outputs (EDA + SQL + insights)
│   └── models/             # Serialized difficulty model artifact
├── sql/                    # DuckDB SQL notebooks executed during the pipeline
├── src/                    # Python source modules (ingestion, features, scoring, EDA)
├── test_databaes.csv       # Final flight-level difficulty export
├── reports/report.txt      # Narrative summary of findings & recommendations
└── README.md               # This file
```

## Environment setup

1. Ensure Python 3.12 is available (a `.venv` is already configured inside the Codespace).
2. Install dependencies:

```bash
/workspaces/skyhack/.venv/bin/python -m pip install -r requirements.txt
```

> The Codespace provisioning step has already executed the command above; rerun it only if new packages are added.

## Reproducing the analysis

Run the orchestrated pipeline from the project root:

```bash
/workspaces/skyhack/.venv/bin/python -m src.pipeline
```

The run performs the following actions:

1. Loads all raw CSVs into pandas and DuckDB.
2. Engineers flight-level, passenger, baggage, and special-service features—including temporal context such as minutes-since-first departure and bank position.
3. Trains a HistGradientBoosting model (with class weighting), applies isotonic calibration, and blends in a rule-based lift to prioritize recall on obvious high-risk flights.
4. Generates daily difficulty rankings and classifications (Difficult / Medium / Easy), attaches SHAP driver text & interaction plots, and exports everything to `test_databaes.csv`.
5. Produces EDA visualizations, summary statistics, SQL-driven tables, cost-benefit analytics, and actionable insights.
6. Saves serialized artifacts (model + calibrator bundle, metrics, recommended actions) for transparency.

### Local execution guide for reviewers / judges

If you are evaluating this submission on your own machine (outside the Codespace), follow these steps:

1. **Clone the repository and enter the project directory**
	```bash
	git clone https://github.com/rishi-dtu-official/skyhack.git
	cd skyhack
	```
2. **Create and activate a Python 3.12 virtual environment**
	```bash
	python3 -m venv .venv
	source .venv/bin/activate
	```
	*(Windows PowerShell)*
	```powershell
	python -m venv .venv
	.venv\Scripts\Activate.ps1
	```
3. **Install project dependencies**
	```bash
	pip install -r requirements.txt
	```
4. **(Optional) Remove heavy artifacts for a clean run**
	```bash
	rm -rf artifacts data/processed test_databaes.csv visualisation
	```
5. **Execute the end-to-end pipeline**
	```bash
	python -m src.pipeline
	```
6. **Review outputs**
	- `test_databaes.csv` (flight difficulty export)
	- `reports/report.txt` (narrative summary)
	- `artifacts/figures/` (EDA plots + SHAP interaction)
	- `visualisation/` (calibration curve, confusion matrix, cost comparison, feature bars)
	- `artifacts/tables/` (metrics, feature importances, cost-benefit, SQL results)
7. **Deactivate the virtual environment when finished**
	```bash
	deactivate
	```

## Key deliverables

- `test_databaes.csv`: Flight identifiers, feature highlights, raw & blended probabilities, rule trigger flag, SHAP driver text, daily rank, and difficulty class.
- `reports/report.txt`: Narrative report answering all EDA questions, describing the scoring approach, and highlighting operational recommendations.
- `artifacts/figures/*.png`: Visuals referenced in the report (now includes SHAP interaction plots).
- `artifacts/tables/*.csv|json`: Supporting tables including model metrics, SHAP feature importances, cost-benefit analysis, EDA summaries, and DuckDB query outputs.
- `visualisation/*.png`: Calibration curve, confusion matrix, top feature bars, and daily cost comparison to accompany the executive summary.

## Extending the solution

- Add new engineered features in `src/feature_engineering.py` and rerun the pipeline.
- Swap the classifier or hyperparameters in `src/scoring.py` to test alternative scoring strategies.
- Place additional SQL queries inside `sql/` and they will be executed automatically on the scored feature table.
- Use the processed Parquet tables (`data/processed/`) to explore ad-hoc analytics or to build dashboards.

## Troubleshooting

- If the pipeline fails because of missing dependencies, rerun the install command listed above.
- DuckDB results are written to `data/interim/skyhack.duckdb`. Remove the file to force a clean rebuild of SQL tables.
- For large dataset experiments, consider enabling lazy reading via DuckDB to reduce memory pressure.
