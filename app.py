import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import hashlib

st.set_page_config(layout="wide")
st.title("Nairobi County Healthcare Analytics Dashboard (PCA + OLAP + Anonymization)")

# --------------------------------------------------------------------
# LOAD DATA (optimized for larger dataset)
# --------------------------------------------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/MutuaNdunda/CSA_806_PCA/refs/heads/main/data/healthcare_simulated_data.csv"
    return pd.read_csv(url, parse_dates=["Date"])

df = load_data()

# Ensure predictable PatientID for anonymization; keep the column if not present
np.random.seed(42)
if "PatientID" not in df.columns:
    df["PatientID"] = np.random.randint(10000, 999999, df.shape[0])

# --------------------------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------------------------
st.sidebar.header("Filters")

clinics = sorted(df["Clinic"].unique())
ailments = sorted(df["Ailment"].unique())

clinic_filter = st.sidebar.multiselect("Select Clinic(s)", clinics, default=clinics)
ailment_filter = st.sidebar.multiselect("Select Ailment(s)", ailments, default=ailments)

# date_input expects a tuple (start, end)
date_min = df["Date"].min().date()
date_max = df["Date"].max().date()
date_range = st.sidebar.date_input("Select Date Range", value=(date_min, date_max))

performance_mode = st.sidebar.checkbox("Enable Performance Mode (sample large data)", value=True)
sample_size = st.sidebar.number_input("Sample size when performance mode enabled", value=30000, step=1000)

# --------------------------------------------------------------------
# FILTER DATA
# --------------------------------------------------------------------
# Guard if user picks invalid tuple
start_date = pd.to_datetime(date_range[0])
end_date = pd.to_datetime(date_range[1])
if end_date < start_date:
    st.sidebar.error("End date must be after start date.")
    end_date = start_date

filtered_df = df[
    (df["Clinic"].isin(clinic_filter)) &
    (df["Ailment"].isin(ailment_filter)) &
    (df["Date"].between(start_date, end_date))
].reset_index(drop=True)

# Performance sampling
if performance_mode and len(filtered_df) > sample_size:
    filtered_df = filtered_df.sample(sample_size, random_state=42).reset_index(drop=True)

# --------------------------------------------------------------------
# PCA COMPUTATION (done once so both PCA & OLAP tabs share it)
# --------------------------------------------------------------------
pca_table = pd.DataFrame()  # empty default

if len(filtered_df) == 0:
    st.warning("No records match the selected filters. Please adjust filters to see analyses.")
else:
    numeric_cols = ["Attendance", "MedicationUsed", "MedicationStock"]
    # Ensure numeric cols exist
    if not all(col in filtered_df.columns for col in numeric_cols):
        st.error(f"Missing expected numeric columns: {numeric_cols}")
    else:
        X = filtered_df[numeric_cols].fillna(0).values

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply PCA (2 components)
        pca = PCA(n_components=2)
        components = pca.fit_transform(X_scaled)

        # Build pca_table with aligned metadata
        pca_table = pd.DataFrame({
            "PC1": components[:, 0],
            "PC2": components[:, 1],
            "Clinic": filtered_df["Clinic"].reset_index(drop=True),
            "Ailment": filtered_df["Ailment"].reset_index(drop=True),
            "Date": filtered_df["Date"].reset_index(drop=True)
        })

# --------------------------------------------------------------------
# TAB LAYOUT  (FIXED — Added "Team Members")
# --------------------------------------------------------------------
tab_overview, tab_pca, tab_olap, tab_stock, tab_anonymize, tab_team = st.tabs(
    ["Overview", "PCA (Dimensionality Reduction)", "OLAP Trends", "Stock Adequacy", "Anonymize & Download", "Team Members"]
)

# --------------------------------------------------------------------
# TAB 1: OVERVIEW
# --------------------------------------------------------------------
with tab_overview:
    st.subheader("Filtered Dataset Preview")
    st.dataframe(filtered_df.head(20))

    # Compute summary metrics
    total_attendance = filtered_df["Attendance"].sum()
    total_medication_used = filtered_df["MedicationUsed"].sum()
    avg_stock = filtered_df["MedicationStock"].mean()

    # Format numbers with commas and 2 decimal places
    total_attendance_fmt = f"{total_attendance:,.0f}"
    total_medication_fmt = f"{total_medication_used:,.0f}"
    avg_stock_fmt = f"{avg_stock:,.2f}"

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Attendance", total_attendance_fmt)
    col2.metric("Total Medication Used", total_medication_fmt)
    col3.metric("Avg Medication Stock", avg_stock_fmt)


# --------------------------------------------------------------------
# TAB 2: PCA ANALYSIS
# --------------------------------------------------------------------
with tab_pca:
    st.subheader("Principal Component Analysis (PCA)")

    if pca_table.empty:
        st.info("PCA not available (no data after filtering). Adjust filters to compute PCA.")
    else:
        st.write("""
        PCA reduces high-dimensional data into fewer components.
        We compress **Attendance, MedicationUsed, MedicationStock** → 2 PCA components.
        """)

        # Explained variance
        evr = pca.explained_variance_ratio_ if "pca" in locals() else None
        if evr is not None:
            st.write(f"**Explained Variance:** PC1 = {evr[0]:.2%}, PC2 = {evr[1]:.2%}")

        # PCA scatter
        fig = px.scatter(
            pca_table,
            x="PC1",
            y="PC2",
            color="Ailment",
            symbol="Clinic",
            hover_data=["Clinic", "Ailment", "Date"],
            title="PCA Scatter: Clinic & Ailment Patterns"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Show PCA table (first N rows)
        max_rows = st.slider("Rows to show in PCA table", min_value=10, max_value=500, value=50, step=10)
        st.subheader("Full PCA Table (sample)")
        st.dataframe(pca_table.head(max_rows))

        # Download PCA table
        csv_pca = pca_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download PCA Table (CSV)",
            data=csv_pca,
            file_name="pca_table_nairobi_healthcare.csv",
            mime="text/csv"
        )

# --------------------------------------------------------------------
# TAB 3: OLAP ANALYSIS
# --------------------------------------------------------------------
with tab_olap:
    st.subheader("OLAP-Style Trends (Based on PCA Data)")

    if pca_table.empty:
        st.info("OLAP trends unavailable — no PCA data. Adjust filters.")
    else:
        # Add Month column for roll-up
        monthly = pca_table.copy()
        monthly["Month"] = monthly["Date"].dt.to_period("M").astype(str)
        monthly_pca = monthly.groupby("Month", as_index=False)[["PC1", "PC2"]].mean()

        fig1 = px.line(monthly_pca, x="Month", y="PC1", markers=True, title="PC1 Trend (Monthly)")
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.line(monthly_pca, x="Month", y="PC2", markers=True, title="PC2 Trend (Monthly)")
        st.plotly_chart(fig2, use_container_width=True)

        st.write("""
        **Interpretation:**  
        - **PC1** correlates with overall activity (attendance + medication use).  
        - **PC2** correlates with stock variability.  
        - Rising PC1 → seasonal disease spikes.  
        - Rising PC2 → supply pressure / stock reliability issues.
        """)

        # Dice example: Ailment vs PC1
        ailment_agg = pca_table.groupby("Ailment", as_index=False)[["PC1", "PC2"]].mean().sort_values("PC1", ascending=False)
        st.write("### Ailment-level PCA summary (PC1 descending)")
        st.dataframe(ailment_agg)

# --------------------------------------------------------------------
# TAB 4: STOCK ADEQUACY
# --------------------------------------------------------------------
with tab_stock:
    st.subheader("Stock Adequacy Overview")

    if len(filtered_df) == 0:
        st.info("No data to show stock adequacy.")
    else:
        adequacy_counts = filtered_df["StockAdequacy"].value_counts().reset_index()
        adequacy_counts.columns = ["StockAdequacy", "Count"]

        fig_stock = px.pie(
            adequacy_counts,
            names="StockAdequacy",
            values="Count",
            title="Stock Adequacy Distribution"
        )
        st.plotly_chart(fig_stock, use_container_width=True)

        clinic_stock = filtered_df.groupby("Clinic")["StockAdequacy"].value_counts().unstack().fillna(0)
        st.write("### Stock Adequacy by Clinic")
        st.dataframe(clinic_stock)

# --------------------------------------------------------------------
# TAB 5: ANONYMIZATION
# --------------------------------------------------------------------
with tab_anonymize:
    st.subheader("Anonymize Sensitive Patient Data")

    if len(filtered_df) == 0:
        st.info("No data to anonymize.")
    else:
        anon_df = filtered_df.copy()
        anon_df["PatientID_Hash"] = anon_df["PatientID"].astype(str).apply(lambda x: hashlib.sha256(x.encode()).hexdigest())
        anon_df = anon_df.drop(columns=["PatientID"])

        st.write("### Preview of Anonymized Data")
        st.dataframe(anon_df.head(20))

        csv = anon_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Anonymized CSV",
            data=csv,
            file_name="anonymized_healthcare_data.csv",
            mime="text/csv"
        )

# --------------------------------------------------------------------
# TAB 6: TEAM MEMBERS 
# --------------------------------------------------------------------
with tab_team:
    st.subheader("Project Team Members")

    team_data = {
        "Registration Number": [
            "ST62/80168/2024",
            "ST62/80313/2024",
            "ST62/80195/2024",
            "ST62/80774/2024",
            "ST62/80472/2024"
        ],
        "Name": [
            "Gabriel Ndunda",
            "Donsy Ombura",
            "Leonard Kiti",
            "Josephat Motonu",
            "Tabitha Kiarie"
        ]
    }

    df_team = pd.DataFrame(team_data)

    st.table(df_team)
