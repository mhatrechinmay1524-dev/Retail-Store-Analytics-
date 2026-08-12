import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Retail Sales Analytics",
    page_icon="Retail",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #0f172a 0%,
        #111827 50%,
        #172554 100%
    );
    color: white;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}


/* Main title */

.main-title {
    font-size: 40px;
    font-weight: 800;
    text-align: center;
    color: white;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 16px;
    margin-bottom: 30px;
}


/* =========================================================
   KPI CARDS
   ========================================================= */

.metric-card {
    border-radius: 18px;
    padding: 22px 12px;
    text-align: center;
    min-height: 125px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.30);
    color: white;
}

.card-1 {
    background: linear-gradient(
        135deg,
        #2563eb,
        #1e40af
    );
}

.card-2 {
    background: linear-gradient(
        135deg,
        #16a34a,
        #166534
    );
}

.card-3 {
    background: linear-gradient(
        135deg,
        #9333ea,
        #6b21a8
    );
}

.card-4 {
    background: linear-gradient(
        135deg,
        #ea580c,
        #9a3412
    );
}

.metric-title {
    color: #ffffff;
    font-size: 15px;
    font-weight: 500;
    margin-bottom: 8px;
}

.metric-value {
    color: white;
    font-size: 27px;
    font-weight: 800;
    white-space: nowrap;
}


/* =========================================================
   SECTION HEADINGS
   ========================================================= */

.section-title {
    color: white;
    font-size: 25px;
    font-weight: 700;
    margin-top: 30px;
    margin-bottom: 15px;
}


/* =========================================================
   PREDICTION BOX
   ========================================================= */

.prediction-box {
    background: linear-gradient(
        135deg,
        #4f46e5,
        #7c3aed
    );
    padding: 28px;
    border-radius: 20px;
    text-align: center;
    margin-top: 20px;
    box-shadow: 0px 10px 30px rgba(99,102,241,0.35);
}

.prediction-label {
    font-size: 17px;
    color: #e0e7ff;
}

.prediction-value {
    font-size: 38px;
    font-weight: 800;
    color: white;
}


/* =========================================================
   SIDEBAR
   ========================================================= */

section[data-testid="stSidebar"] {
    background: #020617;
}

section[data-testid="stSidebar"] h2 {
    color: white;
}


/* =========================================================
   BUTTON
   ========================================================= */

.stButton > button {
    width: 100%;
    border-radius: 12px;
    border: none;
    background: linear-gradient(
        135deg,
        #2563eb,
        #7c3aed
    );
    color: white;
    font-weight: 700;
    padding: 12px;
}

.stButton > button:hover {
    background: linear-gradient(
        135deg,
        #1d4ed8,
        #6d28d9
    );
    color: white;
}


/* =========================================================
   DATAFRAME
   ========================================================= */

[data-testid="stDataFrame"] {
    border-radius: 12px;
}


/* =========================================================
   MOBILE
   ========================================================= */

@media only screen and (max-width: 768px) {

    .main-title {
        font-size: 30px;
    }

    .subtitle {
        font-size: 14px;
    }

    .metric-card {
        min-height: 110px;
        padding: 15px 5px;
    }

    .metric-value {
        font-size: 21px;
    }

    .metric-title {
        font-size: 12px;
    }

}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    data = pd.read_csv("retail_sales_dataset.csv")

    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="coerce"
    )

    return data


# =========================================================
# TRAIN MODEL
# =========================================================

def train_model(data):

    X = data[
        [
            "Gender",
            "Age",
            "Product Category",
            "Quantity",
            "Price per Unit"
        ]
    ]

    y = data["Total Amount"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                [
                    "Gender",
                    "Product Category"
                ]
            ),
            (
                "numerical",
                "passthrough",
                [
                    "Age",
                    "Quantity",
                    "Price per Unit"
                ]
            )
        ]
    )

    random_forest = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                random_forest
            )
        ]
    )

    pipeline.fit(X, y)

    return pipeline


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model(data):

    try:

        with open(
            "model.pkl",
            "rb"
        ) as file:

            model = pickle.load(file)

        return model

    except Exception:

        model = train_model(data)

        try:

            with open(
                "model.pkl",
                "wb"
            ) as file:

                pickle.dump(
                    model,
                    file,
                    protocol=4
                )

        except Exception:
            pass

        return model


# =========================================================
# LOAD DATA + MODEL
# =========================================================

df = load_data()

model = load_model(df)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">Retail Sales Analytics</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Interactive Retail Sales Dashboard and Machine Learning Prediction'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown(
    "## Retail Analytics"
)

st.sidebar.markdown(
    "Explore sales performance, customer insights "
    "and predict total sales using Machine Learning."
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Sales Prediction",
        "Dataset"
    ]
)


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    # =====================================================
    # KPI CALCULATIONS
    # =====================================================

    total_transactions = df[
        "Transaction ID"
    ].nunique()

    total_revenue = df[
        "Total Amount"
    ].sum()

    total_customers = df[
        "Customer ID"
    ].nunique()

    average_sale = df[
        "Total Amount"
    ].mean()


    # =====================================================
    # KPI CARDS
    # =====================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            f"""
            <div class="metric-card card-1">
                <div class="metric-title">
                    Transactions
                </div>

                <div class="metric-value">
                    {total_transactions:,}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            f"""
            <div class="metric-card card-2">
                <div class="metric-title">
                    Total Revenue
                </div>

                <div class="metric-value">
                    ₹{total_revenue:,.0f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with col3:

        st.markdown(
            f"
            <div class="metric-card card-3">
                <div class="metric-title">
                    Customers
                </div>

               
