
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, r2_score, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Page config & style
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="BrewBite Outlet Analytics",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#7c3aed"
GREEN = "#16a34a"
RED = "#dc2626"
AMBER = "#f59e0b"
BLUE = "#2563eb"

st.markdown("""
<style>
    .block-container {padding-top: 2rem;}
    div[data-testid="stMetric"] {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px 16px;
    }
    h1, h2, h3 { font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# Data loading & feature engineering (cached)

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data(path=os.path.join(BASE_DIR, "S.csv")):
    df = pd.read_csv(path)
    df["month"] = pd.to_datetime(df["month"])
    df["Year"] = df["month"].dt.year
    df = df.sort_values(["outlet_id", "month"]).reset_index(drop=True)

    df["Revenue_per_Order"] = df["revenue"] / df["orders"]
    df["Profit_per_Order"] = df["profit"] / df["orders"]
    df["Marketing_ROI"] = df["revenue"] / df["marketing_spend"]
    df["Conversion_Rate"] = df["orders"] / df["footfall"]
    df["Rent_to_Revenue_pct"] = df["rent"] / df["revenue"] * 100
    df["Salary_to_Revenue_pct"] = df["salaries"] / df["revenue"] * 100
    df["Outlet_Age_Years"] = df["Year"] - df["opening_year"]
    df["Revenue_MoM_Growth_pct"] = df.groupby("outlet_id")["revenue"].pct_change() * 100
    df["Revenue_Roll3_Mean"] = df.groupby("outlet_id")["revenue"].transform(lambda s: s.rolling(3).mean())
    df["Is_Profitable"] = (df["profit"] > 0).astype(int)
    df["Low_Margin_Risk"] = (df["profit_margin_pct"] < 20).astype(int)
    return df


@st.cache_resource
def train_risk_classifier(df):
    feature_cols = ["footfall", "orders", "avg_order_value", "marketing_spend",
                     "rent", "salaries", "utilities", "store_size_sqft", "Outlet_Age_Years"]
    cat_cols = ["area", "zone", "store_format"]
    clf_df = df.dropna(subset=feature_cols + cat_cols + ["Low_Margin_Risk"]).copy()

    X = clf_df[feature_cols + cat_cols]
    y = clf_df["Low_Margin_Risk"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), feature_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    pipe = Pipeline([("prep", preprocessor),
                      ("model", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42))])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    # Score full dataset for dashboard use
    clf_df["Predicted_Risk_Prob"] = pipe.predict_proba(clf_df[feature_cols + cat_cols])[:, 1]

    feat_names = feature_cols + list(
        pipe.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(cat_cols)
    )
    importances = pipe.named_steps["model"].feature_importances_
    imp_df = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values(
        "importance", ascending=False
    )

    return pipe, report, cm, clf_df, imp_df


@st.cache_resource
def train_revenue_model(df):
    feature_cols = ["footfall", "avg_order_value", "marketing_spend", "rent",
                     "store_size_sqft", "Outlet_Age_Years", "salaries", "utilities"]
    cat_cols = ["area", "zone", "store_format"]
    model_df = df.dropna(subset=feature_cols + cat_cols + ["revenue"]).copy()

    X = model_df[feature_cols + cat_cols]
    y = model_df["revenue"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), feature_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    pipe = Pipeline([("prep", preprocessor),
                      ("model", RandomForestRegressor(n_estimators=300, random_state=42))])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    metrics = {"R2": r2_score(y_test, pred), "MAE": mean_absolute_error(y_test, pred)}
    return pipe, metrics, y_test, pred


@st.cache_data
def segment_outlets(df):
    outlet_agg = df.groupby("outlet_name").agg(
        avg_revenue=("revenue", "mean"),
        avg_profit=("profit", "mean"),
        avg_profit_margin=("profit_margin_pct", "mean"),
        avg_footfall=("footfall", "mean"),
        avg_marketing_roi=("Marketing_ROI", "mean"),
        avg_conversion=("Conversion_Rate", "mean"),
        zone=("zone", "first"),
        area=("area", "first"),
        store_format=("store_format", "first"),
    ).reset_index()

    cluster_features = ["avg_revenue", "avg_profit", "avg_profit_margin",
                         "avg_footfall", "avg_marketing_roi", "avg_conversion"]
    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(outlet_agg[cluster_features])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    outlet_agg["cluster"] = kmeans.fit_predict(X_cluster)

    cluster_order = outlet_agg.groupby("cluster")["avg_profit_margin"].mean().sort_values().index
    label_map = {cluster_order[0]: "Struggling", cluster_order[1]: "Stable", cluster_order[2]: "High Performer"}
    outlet_agg["segment"] = outlet_agg["cluster"].map(label_map)

    pca = PCA(n_components=2)
    comp = pca.fit_transform(X_cluster)
    outlet_agg["pca1"], outlet_agg["pca2"] = comp[:, 0], comp[:, 1]

    return outlet_agg

# Load data

df = load_data()

# Sidebar filters

st.sidebar.title("☕ BrewBite Analytics")
st.sidebar.caption("Bangalore multi-outlet performance dashboard")

zones = st.sidebar.multiselect("Zone", sorted(df["zone"].unique()), default=sorted(df["zone"].unique()))
formats = st.sidebar.multiselect("Store Format", sorted(df["store_format"].unique()), default=sorted(df["store_format"].unique()))
date_range = st.sidebar.date_input(
    "Date range",
    value=(df["month"].min().date(), df["month"].max().date()),
    min_value=df["month"].min().date(),
    max_value=df["month"].max().date(),
)

if len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = df["month"].min().date(), df["month"].max().date()

mask = (
    df["zone"].isin(zones)
    & df["store_format"].isin(formats)
    & (df["month"].dt.date >= start_date)
    & (df["month"].dt.date <= end_date)
)
fdf = df[mask].copy()

st.sidebar.markdown("---")
st.sidebar.caption(f"{fdf['outlet_id'].nunique()} outlets · {fdf.shape[0]} outlet-months in view")

# Header + KPIs

st.title("BrewBite Outlet Analytics Dashboard")
st.caption("Revenue, profitability, risk detection & outlet segmentation across Bangalore outlets")

if fdf.empty:
    st.warning("No data matches the current filters. Adjust filters in the sidebar.")
    st.stop()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Revenue", f"₹{fdf['revenue'].sum()/1e5:,.1f}L")
k2.metric("Total Profit", f"₹{fdf['profit'].sum()/1e5:,.1f}L")
k3.metric("Avg Profit Margin", f"{fdf['profit_margin_pct'].mean():.1f}%")
k4.metric("Avg Marketing ROI", f"{fdf['Marketing_ROI'].mean():.2f}x")
k5.metric("At-Risk Outlet-Months", f"{fdf['Low_Margin_Risk'].sum():,} ({fdf['Low_Margin_Risk'].mean():.0%})")

st.markdown("---")


# Tabs

tab_overview, tab_outlets, tab_risk, tab_segments, tab_forecast = st.tabs(
    ["📊 Overview", "🏪 Outlet Deep Dive", "⚠️ Risk Detection", "🧩 Segmentation", "📈 Forecast"]
)

#  OVERVIEW 
with tab_overview:
    col1, col2 = st.columns(2)

    monthly = fdf.groupby("month").agg(revenue=("revenue", "sum"), profit=("profit", "sum")).reset_index()
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=monthly["month"], y=monthly["revenue"], name="Revenue",
                               line=dict(color=BLUE, width=3), mode="lines+markers"))
    fig1.add_trace(go.Scatter(x=monthly["month"], y=monthly["profit"], name="Profit",
                               line=dict(color=GREEN, width=3), mode="lines+markers"))
    fig1.update_layout(title="Monthly Revenue vs Profit", height=400, legend=dict(orientation="h", y=1.1))
    col1.plotly_chart(fig1, use_container_width=True)

    zone_rev = fdf.groupby("zone")["revenue"].sum().reset_index()
    fig2 = px.pie(zone_rev, names="zone", values="revenue", title="Revenue Share by Zone",
                  color_discrete_sequence=px.colors.sequential.Purples_r)
    fig2.update_layout(height=400)
    col2.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    fmt_margin = fdf.groupby("store_format")["profit_margin_pct"].mean().reset_index().sort_values("profit_margin_pct")
    fig3 = px.bar(fmt_margin, x="store_format", y="profit_margin_pct", title="Avg Profit Margin % by Store Format",
                  color="profit_margin_pct", color_continuous_scale="Viridis")
    fig3.update_layout(height=400)
    col3.plotly_chart(fig3, use_container_width=True)

    corr_cols = ["footfall", "orders", "revenue", "marketing_spend", "rent", "salaries", "profit_margin_pct"]
    corr = fdf[corr_cols].corr()
    fig4 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                     title="Correlation Matrix")
    fig4.update_layout(height=400)
    col4.plotly_chart(fig4, use_container_width=True)

#  OUTLET DEEP DIVE 
with tab_outlets:
    outlet_totals = fdf.groupby("outlet_name")["profit"].sum().sort_values()
    fig5 = px.bar(outlet_totals, orientation="h", title="Total Profit by Outlet",
                  color=outlet_totals.values, color_continuous_scale="RdYlGn")
    fig5.update_layout(height=900, showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig5, use_container_width=True)

    selected_outlet = st.selectbox("Select an outlet to inspect trend", sorted(fdf["outlet_name"].unique()))
    odf = fdf[fdf["outlet_name"] == selected_outlet].sort_values("month")

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Monthly Revenue", f"₹{odf['revenue'].mean():,.0f}")
    c2.metric("Avg Profit Margin", f"{odf['profit_margin_pct'].mean():.1f}%")
    c3.metric("Avg Marketing ROI", f"{odf['Marketing_ROI'].mean():.2f}x")

    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=odf["month"], y=odf["revenue"], name="Revenue", line=dict(color=BLUE)))
    fig6.add_trace(go.Scatter(x=odf["month"], y=odf["profit"], name="Profit", line=dict(color=GREEN)))
    fig6.add_trace(go.Scatter(x=odf["month"], y=odf["Revenue_Roll3_Mean"], name="Revenue (3M Rolling Avg)",
                               line=dict(color=AMBER, dash="dash")))
    fig6.update_layout(title=f"{selected_outlet} — Revenue & Profit Trend", height=450)
    st.plotly_chart(fig6, use_container_width=True)

#RISK DETECTION 
with tab_risk:
    st.subheader("Outlet Risk Classifier (Random Forest)")
    st.caption("Predicts probability that an outlet-month falls below 20% profit margin.")

    clf_pipe, report, cm, scored_df, imp_df = train_risk_classifier(df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Precision (At Risk)", f"{report['1']['precision']:.2f}")
    c2.metric("Recall (At Risk)", f"{report['1']['recall']:.2f}")
    c3.metric("F1-score (At Risk)", f"{report['1']['f1-score']:.2f}")

    col1, col2 = st.columns(2)
    cm_fig = px.imshow(cm, text_auto=True, x=["Healthy", "At Risk"], y=["Healthy", "At Risk"],
                       color_continuous_scale="Blues", title="Confusion Matrix")
    col1.plotly_chart(cm_fig, use_container_width=True)

    imp_fig = px.bar(imp_df.head(12).sort_values("importance"), x="importance", y="feature",
                     orientation="h", title="Top Risk Drivers (Feature Importance)",
                     color="importance", color_continuous_scale="Purples")
    col2.plotly_chart(imp_fig, use_container_width=True)

    st.markdown("#### Highest Risk Outlet-Months (latest month in view)")
    latest_month = fdf["month"].max()
    latest_scored = scored_df[scored_df["outlet_id"].isin(fdf["outlet_id"].unique()) & (scored_df["month"] == latest_month)]
    risk_table = latest_scored[["outlet_name", "zone", "store_format", "profit_margin_pct", "Predicted_Risk_Prob"]] \
        .sort_values("Predicted_Risk_Prob", ascending=False).head(10)
    risk_table["Predicted_Risk_Prob"] = (risk_table["Predicted_Risk_Prob"] * 100).round(1).astype(str) + "%"
    st.dataframe(risk_table, use_container_width=True, hide_index=True)

# SEGMENTATION -
with tab_segments:
    st.subheader("Outlet Segmentation (KMeans + PCA)")
    outlet_agg = segment_outlets(fdf)

    palette = {"Struggling": RED, "Stable": AMBER, "High Performer": GREEN}
    fig7 = px.scatter(outlet_agg, x="pca1", y="pca2", color="segment", hover_name="outlet_name",
                      color_discrete_map=palette, title="Outlet Segments (PCA-projected)",
                      size="avg_revenue", size_max=25)
    fig7.update_layout(height=500)
    st.plotly_chart(fig7, use_container_width=True)

    seg_counts = outlet_agg["segment"].value_counts().reset_index()
    seg_counts.columns = ["segment", "count"]
    col1, col2 = st.columns([1, 2])
    fig8 = px.bar(seg_counts, x="segment", y="count", color="segment", color_discrete_map=palette,
                 title="Outlets per Segment")
    col1.plotly_chart(fig8, use_container_width=True)

    col2.markdown("#### Segment Details")
    col2.dataframe(
        outlet_agg[["outlet_name", "zone", "store_format", "avg_revenue", "avg_profit_margin", "segment"]]
        .sort_values("avg_profit_margin"),
        use_container_width=True, hide_index=True, height=380
    )


with tab_forecast:
    st.subheader("Revenue Forecast (Random Forest Regressor)")
    st.caption("Compares model predictions vs actual on a held-out test set.")

    rev_pipe, rev_metrics, y_test, y_pred = train_revenue_model(df)
    c1, c2 = st.columns(2)
    c1.metric("R² (test set)", f"{rev_metrics['R2']:.3f}")
    c2.metric("MAE (test set)", f"₹{rev_metrics['MAE']:,.0f}")

    fig9 = go.Figure()
    fig9.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers", marker=dict(color=BLUE, opacity=0.6),
                               name="Predictions"))
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    fig9.add_trace(go.Scatter(x=lims, y=lims, mode="lines", line=dict(color=RED, dash="dash"), name="Perfect Fit"))
    fig9.update_layout(title="Actual vs Predicted Revenue", xaxis_title="Actual", yaxis_title="Predicted", height=450)
    st.plotly_chart(fig9, use_container_width=True)

    st.markdown("#### Total Revenue Trend (All Outlets in View)")
    monthly_total = fdf.groupby("month")["revenue"].sum().reset_index()
    fig10 = px.line(monthly_total, x="month", y="revenue", markers=True, title="Total Monthly Revenue")
    fig10.update_traces(line_color=PRIMARY)
    st.plotly_chart(fig10, use_container_width=True)

st.markdown("---")
st.caption("Built with Streamlit · Data: BrewBite outlet performance (Bangalore) · Models cached for speed")