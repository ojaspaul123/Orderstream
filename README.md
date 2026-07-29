# ☕ BrewBite Outlet Analytics

An end-to-end data analytics project for a multi-outlet food & beverage chain in Bangalore — combining **data engineering feature pipelines**, **machine learning models**, and an **interactive Streamlit dashboard** to turn raw outlet-level financial data into decisions someone can actually act on.

> Built as a portfolio / learning project to practice the full stack: data cleaning → feature engineering → statistical analysis → ML modeling → interactive BI dashboard.

---

## 🖼️ Preview

| Overview | Risk Detection | Segmentation |
|---|---|---|
| Revenue/profit trends, zone & format breakdowns | ML-flagged at-risk outlets with driver explanations | Outlets clustered into performance tiers |

<img width="958" height="449" alt="image" src="https://github.com/user-attachments/assets/b2d45306-a26e-4bb9-8e62-33c58071f15c" />


---

## 🧾 What This Project Does

Starting from a single raw CSV of monthly outlet performance (footfall, orders, revenue, costs, profit margins across 50 outlets over 20 months), this project:

1. **Cleans & validates the data** with automated quality checks (nulls, duplicates, logical inconsistencies like orders exceeding footfall)
2. **Engineers features** — unit economics (revenue/profit per order), cost ratios (rent-to-revenue, salary-to-revenue), momentum features (month-over-month growth, rolling averages), and risk labels
3. **Runs statistical tests** (ANOVA, t-tests) to check whether differences across store formats/zones are real or just noise
4. **Trains ML models**:
   - Random Forest / XGBoost regression to predict outlet revenue
   - Random Forest classification to flag outlets at risk of falling below a healthy profit margin
   - KMeans + PCA to segment outlets into performance tiers (Struggling / Stable / High Performer)
   - ARIMA time-series forecasting for revenue trends
   - SHAP for model explainability (why a specific outlet is predicted to underperform)
5. **Serves it all through a live Streamlit dashboard** — filterable by zone, store format, and date range, with 5 tabs: Overview, Outlet Deep Dive, Risk Detection, Segmentation, and Forecast

---

## 📁 Project Structure

```
brewbite-outlet-analytics/
│
├── app.py                          # Streamlit dashboard (main entry point)
├── brewbite_analytics_notebook.ipynb   # Full analysis notebook (EDA, stats, ML, SHAP, forecasting)
├── S.csv                           # Raw outlet-level monthly data
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── docs/                           # (optional) screenshots, architecture diagram
```

**Note:** `app.py` currently reads `S.csv` from the same folder it's run in (resolved via `BASE_DIR = os.path.dirname(os.path.abspath(__file__))`), so the CSV must sit next to `app.py`.

---

## 🚀 How to Run

```bash
git clone https://github.com/<your-username>/brewbite-outlet-analytics.git
cd brewbite-outlet-analytics

pip install -r requirements.txt

python -m streamlit run app.py
```

Then open the URL Streamlit prints — usually `https://orderstream-vj6bejbnn4j22pnkikvufp.streamlit.app/`.

To explore the full analysis (statistical tests, model comparisons, SHAP plots, forecasting) instead of just the dashboard, open `brewbite_analytics_notebook.ipynb` in Jupyter.

---

## 👥 Who This Is For

- **Outlet / operations managers** — spot which outlets are underperforming *before* month-end reports catch it, and see *why* (via feature importance and SHAP)
- **Marketing teams** — check Marketing ROI and conversion rate by outlet/zone to reallocate spend
- **Business/finance analysts** — segment outlets into performance tiers to prioritize interventions
- **Recruiters / interviewers** — as a portfolio piece demonstrating the full data science + data engineering lifecycle, not just a Jupyter notebook of charts

---

## 💡 Real-World Impact

In a real deployment, a dashboard like this would let a small F&B chain:

- **Catch underperforming outlets early** — the risk classifier flags low-margin outlets before they show up as a bad quarterly number, using precision/recall rather than gut feel
- **Justify marketing spend allocation** — Marketing ROI and conversion rate broken out by outlet/zone show where spend is/isn't working
- **Standardize expansion decisions** — the "High Performer" vs "Struggling" segmentation gives a repeatable, data-backed way to decide which outlet profile to replicate when opening new locations
- **Reduce manual reporting time** — replaces a recurring manual Excel/PowerPoint exercise with a live, filterable dashboard anyone on the team can self-serve

*(This project uses synthetic/sample data — treat the specific numbers as illustrative, not real business figures.)*

---

## 🐛 Issues Faced & How They Were Fixed

Building and running this project surfaced a handful of very common (and very fixable) real-world setup issues — documented here because they're useful for anyone reusing this project:

| Issue | Root Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'plotly'` despite a successful `pip install` | Installed the wrong package (`px`) by mistake — pip installs exactly the name you type, not what you meant | `pip uninstall px -y` then `pip install plotly`, or just `pip install -r requirements.txt` |
| `FileNotFoundError: [Errno 2] No such file or directory: 'S.csv'` | `pd.read_csv("S.csv")` used a relative path resolved against the terminal's **current working directory**, not the script's location | Anchored the path to the script itself: `BASE_DIR = os.path.dirname(os.path.abspath(__file__))` and load `S.csv` from there |
| `streamlit : command not recognized` | Python's `Scripts` folder (where `streamlit.exe` lives) wasn't on the Windows PATH | Ran via the Python interpreter directly: `python -m streamlit run app.py` (works regardless of PATH), or added the Scripts folder to PATH permanently |
| `cd : Cannot find path '...\outstream'` | Typed path didn't match the actual nested folder structure (e.g. missing an intermediate `ML_Projects` folder) | Used VS Code's Explorer → right-click file → *Copy Path* / *Open in Integrated Terminal* instead of retyping paths by hand |
| Metric cards showed **white text on a white background** (invisible until manually highlighted) | Streamlit's theme applies its own text color on top of custom CSS backgrounds; targeting specific child elements (`label p`, `stMetricValue`) didn't match the actual rendered DOM structure in this Streamlit version | Used a wildcard selector `div[data-testid="stMetric"] *  { color: ... !important; }` to force text color on every nested element regardless of Streamlit's internal markup, and switched the card style to match the dashboard's dark theme |

**Takeaway:** most of these weren't logic bugs — they were environment/path/CSS-specificity issues, which is realistic for any first deployment. Documenting them here doubles as a troubleshooting guide for anyone else running into the same errors.

---

## 🔭 Future Improvements

**Data**
- Replace/augment the outlet-financials CSV with real order-level delivery data (customer id, delivery time, GPS, delivery partner) to make this a genuine food-*delivery* analytics project, not just outlet financials
- Merge in external data — weather, local holidays/festivals, footfall from nearby events — to see if it improves forecast accuracy

**Modeling**
- Hyperparameter tuning (`GridSearchCV` / `Optuna`) instead of default model params
- `TimeSeriesSplit` cross-validation instead of a single random train/test split, since this is monthly panel data
- Per-outlet forecasting (Prophet/SARIMAX with exogenous regressors) instead of one aggregate forecast
- Model ensembling/stacking across Linear, Random Forest, and XGBoost

**MLOps**
- Experiment tracking with MLflow instead of ad-hoc printed metrics
- Serve the risk classifier via a FastAPI endpoint instead of only inside the notebook/dashboard
- Add data/model drift monitoring for when new monthly data lands

**Data Engineering**
- Automate the feature pipeline (currently notebook/dashboard cells) into a scheduled **Airflow DAG**: ingest → validate → transform → score → load to warehouse
- Add **dbt** models on top of a real warehouse (Postgres/Snowflake) so BI and ML share one source of truth
- Add CI (GitHub Actions) to run data quality checks and a smoke test of the pipeline on every push
- Unit tests for feature engineering functions

**Product**
- Deploy the dashboard publicly (Streamlit Community Cloud) instead of local-only
- Add outlet-level drill-down maps (Bangalore zones) using geospatial libraries
- Role-based views (ops manager vs. marketing vs. finance see different default tabs)

---

## 🛠️ Tech Stack

`Python` · `Pandas` / `NumPy` · `scikit-learn` · `XGBoost` · `SHAP` · `statsmodels` (ARIMA) · `Streamlit` · `Plotly`

---

## 📄 License

Add a license of your choice (MIT is common for portfolio projects) — create a `LICENSE` file in the repo root.
