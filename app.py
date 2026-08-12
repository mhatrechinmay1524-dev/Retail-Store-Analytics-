import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Retail Sales Analytics",
    page_icon="🛍️",
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
        #0f172a,
        #111827,
        #172554
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


/* TITLE */

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


/* KPI CARDS */

.metric-card {
    border-radius: 18px;
    padding: 22px 10px;
    text-align: center;
    min-height: 120px;
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
    color: white;
    font-size: 15px;
    margin-bottom: 10px;
}

.metric-value {
    color: white;
    font-size: 27px;
    font-weight: 800;
}


/* SECTION TITLE */

.section-title {
    color: white;
    font-size: 25px;
    font-weight: 700;
    margin-top: 30px;
    margin-bottom: 15px;
}


/* PREDICTION */

.prediction-box {
    background: linear-gradient(
        135deg,
        #4f46e5,
        #7c3aed
    );
    padding: 30px;
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


/* SIDEBAR */

section[data-testid="stSidebar"] {
    background: #020617;
}


/* BUTTON */

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

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    data = pd.read_csv(
        "retail_sales_dataset.csv"
    )

    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="coerce"
    )

    return data


df = load_data()


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

    categorical_columns = [
        "Gender",
        "Product Category"
    ]

    numeric_columns = [
        "Age",
        "Quantity",
        "Price per Unit"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_columns
            ),
            (
                "numeric",
                "passthrough",
                numeric_columns
            )
        ]
    )

    random_forest = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    pipeline = Pipeline(
        [
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

        return model


model = load_model(df)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">'
    'Retail Sales Analytics'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Interactive Retail Sales Dashboard and '
    'Machine Learning Prediction'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown(
    "## Retail Analytics"
)

st.sidebar.write(
    "Explore sales performance, customer insights "
    "and machine learning predictions."
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

    # -----------------------------------------------------
    # KPI VALUES
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # KPI CARDS
    # -----------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        html = (
            '<div class="metric-card card-1">'
            '<div class="metric-title">'
            'Transactions'
            '</div>'
            '<div class="metric-value">'
            + str(total_transactions)
            + '</div>'
            '</div>'
        )

        st.markdown(
            html,
            unsafe_allow_html=True
        )


    with col2:

        revenue_text = (
            "₹"
            + format(
                total_revenue,
                ",.0f"
            )
        )

        html = (
            '<div class="metric-card card-2">'
            '<div class="metric-title">'
            'Total Revenue'
            '</div>'
            '<div class="metric-value">'
            + revenue_text
            + '</div>'
            '</div>'
        )

        st.markdown(
            html,
            unsafe_allow_html=True
        )


    with col3:

        html = (
            '<div class="metric-card card-3">'
            '<div class="metric-title">'
            'Customers'
            '</div>'
            '<div class="metric-value">'
            + str(total_customers)
            + '</div>'
            '</div>'
        )

        st.markdown(
            html,
            unsafe_allow_html=True
        )


    with col4:

        average_text = (
            "₹"
            + format(
                average_sale,
                ",.0f"
            )
        )

        html = (
            '<div class="metric-card card-4">'
            '<div class="metric-title">'
            'Average Sale'
            '</div>'
            '<div class="metric-value">'
            + average_text
            + '</div>'
            '</div>'
        )

        st.markdown(
            html,
            unsafe_allow_html=True
        )


    # =====================================================
    # SALES PERFORMANCE
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        'Sales Performance'
        '</div>',
        unsafe_allow_html=True
    )


    category_sales = (
        df.groupby(
            "Product Category"
        )["Total Amount"]
        .sum()
        .reset_index()
        .sort_values(
            "Total Amount",
            ascending=False
        )
    )


    col1, col2 = st.columns(2)


    # BAR CHART

    with col1:

        fig_bar = px.bar(
            category_sales,
            x="Product Category",
            y="Total Amount",
            text="Total Amount",
            title="Sales by Product Category",
            template="plotly_dark"
        )

        fig_bar.update_traces(
            texttemplate="₹%{text:,.0f}",
            textposition="outside"
        )

        fig_bar.update_layout(
            height=450,
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig_bar,
            use_container_width=True
        )


    # PIE CHART

    with col2:

        fig_pie = px.pie(
            category_sales,
            names="Product Category",
            values="Total Amount",
            hole=0.55,
            title="Sales Distribution",
            template="plotly_dark"
        )

        fig_pie.update_layout(
            height=450,
            paper_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig_pie,
            use_container_width=True
        )


    # =====================================================
    # SALES TREND
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        'Sales Trend'
        '</div>',
        unsafe_allow_html=True
    )


    daily_sales = (
        df.groupby(
            "Date"
        )["Total Amount"]
        .sum()
        .reset_index()
        .sort_values("Date")
    )


    fig_line = px.line(
        daily_sales,
        x="Date",
        y="Total Amount",
        title="Retail Sales Over Time",
        markers=True,
        template="plotly_dark"
    )

    fig_line.update_layout(
        height=450,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(
        fig_line,
        use_container_width=True
    )


    # =====================================================
    # HISTOGRAM AND BOX PLOT
    # =====================================================

    col1, col2 = st.columns(2)


    with col1:

        fig_hist = px.histogram(
            df,
            x="Total Amount",
            nbins=30,
            title="Distribution of Total Amount",
            template="plotly_dark"
        )

        fig_hist.update_layout(
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig_hist,
            use_container_width=True
        )


    with col2:

        fig_box = px.box(
            df,
            y="Total Amount",
            title="Sales Amount Box Plot",
            template="plotly_dark"
        )

        fig_box.update_layout(
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig_box,
            use_container_width=True
        )


    # =====================================================
    # PRODUCT ANALYSIS
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        'Product Analysis'
        '</div>',
        unsafe_allow_html=True
    )


    fig_scatter = px.scatter(
        df,
        x="Quantity",
        y="Price per Unit",
        size="Total Amount",
        color="Product Category",
        hover_data=[
            "Transaction ID",
            "Customer ID",
            "Gender"
        ],
        title="Quantity vs Price per Unit",
        template="plotly_dark"
    )


    fig_scatter.update_layout(
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )


    st.plotly_chart(
        fig_scatter,
        use_container_width=True
    )


# =========================================================
# SALES PREDICTION
# =========================================================

elif page == "Sales Prediction":

    st.markdown(
        '<div class="section-title">'
        'Sales Amount Prediction'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter customer and product information "
        "to predict the expected Total Amount."
    )


    col1, col2 = st.columns(2)


    with col1:

        gender = st.selectbox(
            "Gender",
            [
                "Male",
                "Female"
            ]
        )

        age = st.slider(
            "Age",
            18,
            70,
            30
        )

        category = st.selectbox(
            "Product Category",
            sorted(
                df[
                    "Product Category"
                ].unique()
            )
        )


    with col2:

        quantity = st.slider(
            "Quantity",
            1,
            10,
            2
        )

        price = st.number_input(
            "Price per Unit",
            min_value=1.0,
            max_value=5000.0,
            value=100.0,
            step=10.0
        )


    st.markdown("---")


    if st.button(
        "Predict Total Amount",
        use_container_width=True
    ):

        input_data = pd.DataFrame(
            {
                "Gender": [gender],
                "Age": [age],
                "Product Category": [category],
                "Quantity": [quantity],
                "Price per Unit": [price]
            }
        )


        prediction = model.predict(
            input_data
        )[0]


        prediction_text = (
            "₹"
            + format(
                prediction,
                ",.2f"
            )
        )


        html = (
            '<div class="prediction-box">'
            '<div class="prediction-label">'
            'Predicted Total Amount'
            '</div>'
            '<div class="prediction-value">'
            + prediction_text
            + '</div>'
            '</div>'
        )


        st.markdown(
            html,
            unsafe_allow_html=True
        )

        st.success(
            "Prediction completed successfully."
        )


# =========================================================
# DATASET
# =========================================================

elif page == "Dataset":

    st.markdown(
        '<div class="section-title">'
        'Retail Sales Dataset'
        '</div>',
        unsafe_allow_html=True
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Rows",
            df.shape[0]
        )


    with col2:

        st.metric(
            "Columns",
            df.shape[1]
        )


    with col3:

        st.metric(
            "Missing Values",
            int(
                df.isnull()
                .sum()
                .sum()
            )
        )


    st.markdown("---")


    st.dataframe(
        df,
        use_container_width=True,
        height=600
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    '<div style="text-align:center;'
    'color:#94a3b8;'
    'padding:20px;">'
    '<b>Retail Sales Analytics</b><br>'
    'Built with Python, Pandas, Plotly, '
    'Scikit-learn and Streamlit'
    '</div>',
    unsafe_allow_html=True
    )
