import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Page config
st.set_page_config(page_title="Car Price Prediction", layout="wide", page_icon="🚗")
sns.set_style("whitegrid")
st.title(" Car Price Prediction using Regression")
st.caption(
    "Multiple Linear Regression on the Car Price Assignment dataset, "
    "with a Ridge / Lasso regularization self-learning comparison."
)
CYLINDER_MAP = {"two": 2, "three": 3, "four": 4, "five": 5,
                 "six": 6, "eight": 8, "twelve": 12}
BRAND_FIX = {"maxda": "mazda", "porcshce": "porsche", "toyouta": "toyota",
             "vokswagen": "volkswagen", "vw": "volkswagen"}
# Data loading & preprocessing (cached)
@st.cache_data
def load_data(path_or_buffer):
    return pd.read_excel(path_or_buffer)
@st.cache_data
def preprocess(df_raw: pd.DataFrame):
    df_clean = df_raw.drop(columns=["car_ID"])
    df_clean["carbrand"] = df_clean["CarName"].str.split().str[0].str.lower()
    df_clean["carbrand"] = df_clean["carbrand"].replace(BRAND_FIX)
    df_clean = df_clean.drop(columns=["CarName"])
    df_encoded = df_clean.copy()
    binary_cols = ["fueltype", "aspiration", "doornumber", "enginelocation"]
    encoders = {}
    for col in binary_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        encoders[col] = le
    df_encoded["cylindernumber"] = df_encoded["cylindernumber"].map(CYLINDER_MAP)
    nominal_cols = ["carbody", "drivewheel", "enginetype", "fuelsystem", "carbrand"]
    df_encoded = pd.get_dummies(df_encoded, columns=nominal_cols, drop_first=True)
    X = df_encoded.drop(columns=["price"])
    y = df_encoded["price"]
    return df_clean, df_encoded, X, y, encoders, nominal_cols
@st.cache_resource
def train_models(X_train, X_test, y_train, y_test, alpha_ridge, alpha_lasso):
    lr = LinearRegression().fit(X_train, y_train)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    lr_scaled = LinearRegression().fit(X_train_s, y_train)
    ridge = Ridge(alpha=alpha_ridge, random_state=42).fit(X_train_s, y_train)
    lasso = Lasso(alpha=alpha_lasso, random_state=42, max_iter=10000).fit(X_train_s, y_train)
    return lr, scaler, lr_scaled, ridge, lasso
def metrics_row(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2 Score": r2_score(y_true, y_pred),
    }
# Sidebar — data source + controls
st.sidebar.header("⚙️ Settings")
uploaded_file = st.sidebar.file_uploader("Upload CarPrice_Assignment.xlsx", type=["xlsx"])
default_path = "CarPrice_Assignment.xlsx"
if uploaded_file is not None:
    df_raw = load_data(uploaded_file)
    st.sidebar.success("Using uploaded file.")
else:
    try:
        df_raw = load_data(default_path)
        st.sidebar.info("Using bundled sample dataset.")
    except FileNotFoundError:
        st.sidebar.warning("Upload the dataset to get started.")
        st.stop()
test_size = st.sidebar.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", value=42, step=1)
st.sidebar.markdown("**Regularization strength (self-learning section)**")
alpha_ridge = st.sidebar.slider("Ridge alpha", 0.1, 50.0, 10.0, 0.1)
alpha_lasso = st.sidebar.slider("Lasso alpha", 0.1, 50.0, 10.0, 0.1)
df_clean, df_encoded, X, y, encoders, nominal_cols = preprocess(df_raw)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=int(random_state)
)
lr_model, scaler, lr_scaled, ridge_model, lasso_model = train_models(
    X_train, X_test, y_train, y_test, alpha_ridge, alpha_lasso
)
y_pred = lr_model.predict(X_test)
numerical_cols = df_raw.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = df_raw.select_dtypes(include=["object"]).columns.tolist()
tabs = st.tabs([
    " Data Overview",
    " Preprocessing",
    " EDA",
    " Model & Evaluation",
    " Predictions",
    " Predict a New Car",
    " Self Learning: Ridge & Lasso",
])
# TAB 1 — Data Overview
with tabs[0]:
    st.subheader("Dataset Overview")
    c1, c2 = st.columns(2)
    c1.metric("Rows", df_raw.shape[0])
    c2.metric("Columns", df_raw.shape[1])
    st.markdown("**First five records**")
    st.dataframe(df_raw.head())
    st.markdown("**Last five records**")
    st.dataframe(df_raw.tail())
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Numerical attributes ({len(numerical_cols)})**")
        st.write(numerical_cols)
    with c2:
        st.markdown(f"**Categorical attributes ({len(categorical_cols)})**")
        st.write(categorical_cols)
    c1, c2 = st.columns(2)
    c1.metric("Missing values (total)", int(df_raw.isnull().sum().sum()))
    c2.metric("Duplicate records", int(df_raw.duplicated().sum()))
    st.markdown("**Descriptive statistics (numerical attributes)**")
    st.dataframe(df_raw.describe())
# TAB 2 — Preprocessing
with tabs[1]:
    st.subheader("Preprocessing Steps Applied")
    st.markdown("""
    - Dropped `car_ID` (identifier, no predictive value)
    - Extracted **brand** from `CarName` (first word), fixed typos (`maxda`→`mazda`, etc.), dropped raw `CarName`
    - Label-encoded binary columns: `fueltype`, `aspiration`, `doornumber`, `enginelocation`
    - Mapped `cylindernumber` word→integer (`four`→4, `six`→6, ...)
    - One-hot encoded nominal columns: `carbody`, `drivewheel`, `enginetype`, `fuelsystem`, `carbrand`
    """)
    st.markdown(f"**Shape after encoding:** {df_encoded.shape}")
    st.dataframe(df_encoded.head())
    st.markdown(f"**X shape:** {X.shape}  |  **y shape:** {y.shape}")
# TAB 3 — EDA
with tabs[2]:
    st.subheader("Exploratory Data Analysis")
    st.markdown("**Distribution of Car Prices**")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df_raw["price"], kde=True, bins=30, ax=axes[0], color="steelblue")
    axes[0].set_title("Price Distribution")
    sns.boxplot(x=df_raw["price"], ax=axes[1], color="lightcoral")
    axes[1].set_title("Price Boxplot")
    st.pyplot(fig)
    st.caption(f"Skewness: {df_raw['price'].skew():.2f} (right-skewed — a long tail of premium cars)")
    col1, col2, col3 = st.columns(3)
    for col, ax_col, feature in zip(
        (col1, col2, col3), (col1, col2, col3), ["horsepower", "enginesize", "curbweight"]
    ):
        with col:
            st.markdown(f"**{feature} vs Price**")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.regplot(x=feature, y="price", data=df_raw, ax=ax,
                        scatter_kws={"alpha": 0.6}, line_kws={"color": "red"})
            st.pyplot(fig)
            st.caption(f"Correlation: {df_raw[feature].corr(df_raw['price']):.3f}")
    st.markdown("**Correlation Heatmap (numerical attributes)**")
    fig, ax = plt.subplots(figsize=(11, 8))
    corr_matrix = df_raw[numerical_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                linewidths=0.5, ax=ax)
    st.pyplot(fig)
    st.markdown("**Features most correlated with price**")
    price_corr = corr_matrix["price"].drop("price").sort_values(key=abs, ascending=False)
    st.dataframe(price_corr.to_frame("Correlation with price"))
# TAB 4 — Model & Evaluation
with tabs[3]:
    st.subheader("Multiple Linear Regression — Training & Evaluation")
    c1, c2 = st.columns(2)
    c1.metric("Training records", X_train.shape[0])
    c2.metric("Testing records", X_test.shape[0])
    m = metrics_row(y_test, y_pred)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", f"{m['MAE']:,.2f}")
    c2.metric("MSE", f"{m['MSE']:,.2f}")
    c3.metric("RMSE", f"{m['RMSE']:,.2f}")
    c4.metric("R² Score", f"{m['R2 Score']:.4f}")
    st.info(
        f"An R² of {m['R2 Score']:.2f} means the model explains about "
        f"{m['R2 Score']*100:.1f}% of the variance in car price on unseen test data. "
        "RMSE gives the typical prediction error in the same units as price."
    )
# TAB 5 — Predictions visualization
with tabs[4]:
    st.subheader("Visualization of Predictions")
    st.markdown("**Actual vs Predicted Price**")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, alpha=0.7, color="teal", edgecolor="k")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", label="Perfect Prediction (y = x)")
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.legend()
    st.pyplot(fig)
    st.markdown("**Residuals**")
    residuals = y_test - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(y_pred, residuals, alpha=0.7, color="darkorange", edgecolor="k")
    axes[0].axhline(y=0, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted Price")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residual Plot")
    sns.histplot(residuals, kde=True, ax=axes[1], color="mediumpurple")
    axes[1].set_title("Distribution of Residuals")
    st.pyplot(fig)
    st.caption(f"Mean residual: {residuals.mean():.2f} (should be close to 0)")
    st.markdown("**Sample predictions**")
    st.dataframe(pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred}).head(15))
# TAB 6 — Predict a new car (interactive form)
with tabs[5]:
    st.subheader("Predict the Price of a New Car")
    st.caption("Fill in the specifications below and get an instant price prediction.")
    with st.form("new_car_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            fueltype = st.selectbox("Fuel type", ["gas", "diesel"])
            aspiration = st.selectbox("Aspiration", ["std", "turbo"])
            doornumber = st.selectbox("Doors", ["two", "four"])
            enginelocation = st.selectbox("Engine location", ["front", "rear"])
            carbody = st.selectbox("Body style", sorted(df_clean["carbody"].unique()))
        with c2:
            drivewheel = st.selectbox("Drive wheel", sorted(df_clean["drivewheel"].unique()))
            enginetype = st.selectbox("Engine type", sorted(df_clean["enginetype"].unique()))
            fuelsystem = st.selectbox("Fuel system", sorted(df_clean["fuelsystem"].unique()))
            cylindernumber = st.selectbox("Cylinders", list(CYLINDER_MAP.keys()), index=2)
            carbrand = st.selectbox("Brand", sorted(df_clean["carbrand"].unique()))
        with c3:
            wheelbase = st.number_input("Wheelbase", value=99.0)
            carlength = st.number_input("Car length", value=175.0)
            carwidth = st.number_input("Car width", value=66.0)
            carheight = st.number_input("Car height", value=54.0)
            curbweight = st.number_input("Curb weight", value=2500)
        c4, c5, c6 = st.columns(3)
        with c4:
            enginesize = st.number_input("Engine size", value=140)
            boreratio = st.number_input("Bore ratio", value=3.3)
        with c5:
            stroke = st.number_input("Stroke", value=3.3)
            compressionratio = st.number_input("Compression ratio", value=9.0)
        with c6:
            horsepower = st.number_input("Horsepower", value=110)
            peakrpm = st.number_input("Peak RPM", value=5200)
        c7, c8, c9 = st.columns(3)
        with c7:
            citympg = st.number_input("City MPG", value=27)
        with c8:
            highwaympg = st.number_input("Highway MPG", value=33)
        with c9:
            symboling = st.number_input("Symboling (risk rating)", value=0, step=1)
        submitted = st.form_submit_button("🔮 Predict Price")
    if submitted:
        row = {
            "symboling": symboling,
            "fueltype": fueltype,
            "aspiration": aspiration,
            "doornumber": doornumber,
            "carbody": carbody,
            "drivewheel": drivewheel,
            "enginelocation": enginelocation,
            "wheelbase": wheelbase,
            "carlength": carlength,
            "carwidth": carwidth,
            "carheight": carheight,
            "curbweight": curbweight,
            "enginetype": enginetype,
            "cylindernumber": cylindernumber,
            "enginesize": enginesize,
            "fuelsystem": fuelsystem,
            "boreratio": boreratio,
            "stroke": stroke,
            "compressionratio": compressionratio,
            "horsepower": horsepower,
            "peakrpm": peakrpm,
            "citympg": citympg,
            "highwaympg": highwaympg,
            "carbrand": carbrand,
        }
        new_df = pd.DataFrame([row])
        for col in ["fueltype", "aspiration", "doornumber", "enginelocation"]:
            new_df[col] = encoders[col].transform(new_df[col])
        new_df["cylindernumber"] = new_df["cylindernumber"].map(CYLINDER_MAP)
        new_df = pd.get_dummies(new_df, columns=nominal_cols)
        new_df = new_df.reindex(columns=X.columns, fill_value=0)
        predicted_price = lr_model.predict(new_df)[0]
        st.success(f"### Predicted Price: ${predicted_price:,.2f}")
# TAB 7 — Self learning: Ridge & Lasso
with tabs[6]:
    st.subheader("🧪 Self Learning: Ridge & Lasso Regularization")
    st.markdown("""
    With ~205 rows and 40+ one-hot encoded features, plain OLS Linear Regression is prone to
    **overfitting** and unstable coefficients from **multicollinearity** (e.g. `enginesize`,
    `curbweight`, `horsepower` are all strongly correlated). This section compares plain Linear
    Regression against **Ridge (L2)** and **Lasso (L1)** regularization on the same
    train/test split, using standardized features. Tune the alpha (regularization strength)
    sliders in the sidebar to see the effect live.
    """)
    rows = []
    scaler_local = scaler
    X_test_s = scaler_local.transform(X_test)
    for name, model in [("Linear Regression", lr_scaled),
                         ("Ridge Regression", ridge_model),
                         ("Lasso Regression", lasso_model)]:
        preds = model.predict(X_test_s)
        rows.append({"Model": name, **metrics_row(y_test, preds)})
    results_df = pd.DataFrame(rows).set_index("Model")
    st.dataframe(results_df.round(3))
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        results_df[["MAE", "RMSE"]].plot(kind="bar", ax=ax, color=["steelblue", "salmon"])
        ax.set_title("Error Comparison")
        plt.xticks(rotation=0)
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(6, 4))
        results_df[["R2 Score"]].plot(kind="bar", ax=ax, color="seagreen", legend=False)
        ax.set_title("R² Score Comparison")
        plt.xticks(rotation=0)
        st.pyplot(fig)
    st.markdown("**Which features did Lasso eliminate?**")
    lasso_coefs = pd.Series(lasso_model.coef_, index=X.columns)
    zeroed = lasso_coefs[lasso_coefs == 0]
    kept = lasso_coefs[lasso_coefs != 0].sort_values(key=abs, ascending=False)
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Features zeroed out by Lasso", f"{len(zeroed)} / {len(lasso_coefs)}")
        st.write(list(zeroed.index))
    with c2:
        st.markdown("Top features Lasso kept as most important:")
        st.dataframe(kept.head(10).to_frame("Coefficient"))
    st.info(
        "If Ridge/Lasso match or beat plain OLS on test R² with smaller coefficients, "
        "it confirms OLS was overfitting to noise from the many one-hot columns. "
        "Lasso's zeroed features add little unique signal once core specs are accounted for."
    )
