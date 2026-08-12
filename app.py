import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go


# =========================================================
# PAGE CONFIGURATION
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

    /* Main background */
    .stApp {
        background: linear-gradient(
            135deg,
            #0f172a 0%,
            #111827 50%,
            #172554 100%
        );
        color: white;
    }

    /* Hide Streamlit branding */
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
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
        color: white;
    }

    .subtitle {
        text-align: center;
        color: #cbd5e1;
        font-size: 17px;
        margin-bottom: 30px;
    }

    /* Cards */
    .metric-card {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 18px;
        padding: 22px;
        text-align: center;
        box-shadow: 0px 8px 25px rgba(0,0,0,0.25);
    }

    .metric-title {
        color: #94a3b8;
        font-size: 15px;
    }

    .metric-value {
        color: white;
        font-size: 28px;
        font-weight: 700;
        margin-top: 7px;
    }

    /* Section headings */
    .section-title {
        color: white;
        font-size: 25px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    /* Prediction box */
    .prediction-box {
        background: linear-gradient(
            135deg,
            #6366f1,
            #8b5cf6
        );
        padding: 25px;
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

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #020617;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    data = pd.read_csv("retail_sales_dataset.csv")

    data["Date"] = pd.to_datetime(data["Date"])

    return data


@st.cache_resource
def load_model():

    with open("model.pkl", "rb") as file:
        model = pickle.load(file)

    return model


df = load_data()
model = load_model()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🛍️ Retail Sales Analytics</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Interactive Retail Sales Dashboard & Machine Learning Prediction'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown("## 🛍️ Retail Analytics")

st.sidebar.markdown(
    "Explore sales performance, customer insights and "
    "predict total sales using Machine Learning."
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Dashboard",
        "🤖 Sales Prediction",
        "📋 Dataset"
    ]
)


# =========================================================
# DASHBOARD
# =========================================================

if page == "📊 Dashboard":

    # -----------------------------------------------------
    # KPI CARDS
    # -----------------------------------------------------

    total_transactions = df["Transaction ID"].nunique()
    total_revenue = df["Total Amount"].sum()
    total_customers = df["Customer ID"].nunique()
    average_sale = df["Total Amount"].mean()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">🧾 Transactions</div>
                <div class="metric-value">{total_transactions:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">💰 Total Revenue</div>
                <div class="metric-value">₹{total_revenue:,.0f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">👥 Customers</div>
                <div class="metric-value">{total_customers:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">📈 Avg. Sale</div>
                <div class="metric-value">₹{average_sale:,.0f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # -----------------------------------------------------
    # PRODUCT CATEGORY SALES
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">📊 Sales Performance</div>',
        unsafe_allow_html=True
    )

    category_sales = (
        df.groupby("Product Category")["Total Amount"]
        .sum()
        .reset_index()
        .sort_values("Total Amount", ascending=False)
    )

    col1, col2 = st.columns(2)

    with col1:

        fig_bar = px.bar(
            category_sales,
            x="Product Category",
            y="Total Amount",
            text="Total Amount",
            title="💰 Sales by Product Category",
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


    # -----------------------------------------------------
    # PIE CHART
    # -----------------------------------------------------

    with col2:

        fig_pie = px.pie(
            category_sales,
            names="Product Category",
            values="Total Amount",
            hole=0.55,
            title="🥧 Sales Distribution"
        )

        fig_pie.update_layout(
            height=450,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig_pie,
            use_container_width=True
        )


    # -----------------------------------------------------
    # SALES TREND
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">📅 Sales Trend</div>',
        unsafe_allow_html=True
    )

    daily_sales = (
        df.groupby("Date")["Total Amount"]
        .sum()
        .reset_index()
    )

    fig_line = px.line(
        daily_sales,
        x="Date",
        y="Total Amount",
        title="📈 Retail Sales Over Time",
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


    # -----------------------------------------------------
    # HISTOGRAM + BOX PLOT
    # -----------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        fig_hist = px.histogram(
            df,
            x="Total Amount",
            nbins=30,
            title="📊 Distribution of Total Amount",
            template="plotly_dark"
        )

        fig_hist.update_layout(
            height=400,
            paper_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig_hist,
            use_container_width=True
        )


    with col2:

        fig_box = px.box(
            df,
            y="Total Amount",
            title="📦 Sales Amount Box Plot",
            template="plotly_dark"
        )

        fig_box.update_layout(
            height=400,
            paper_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig_box,
            use_container_width=True
        )


    # -----------------------------------------------------
    # QUANTITY VS PRICE
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">🔍 Product Analysis</div>',
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
        title="🛒 Quantity vs Price per Unit",
        template="plotly_dark"
    )

    fig_scatter.update_layout(
        height=500,
        paper_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(
        fig_scatter,
        use_container_width=True
    )


# =========================================================
# SALES PREDICTION
# =========================================================

elif page == "🤖 Sales Prediction":

    st.markdown(
        '<div class="section-title">🤖 Sales Amount Prediction</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter customer and product information below "
        "to predict the expected Total Amount."
    )

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "👤 Gender",
            ["Male", "Female"]
        )

        age = st.slider(
            "🎂 Age",
            min_value=18,
            max_value=70,
            value=30
        )

        category = st.selectbox(
            "🛍️ Product Category",
            sorted(df["Product Category"].unique())
        )


    with col2:

        quantity = st.slider(
            "📦 Quantity",
            min_value=1,
            max_value=10,
            value=2
        )

        price = st.number_input(
            "💵 Price per Unit",
            min_value=1.0,
            max_value=5000.0,
            value=100.0,
            step=10.0
        )


    st.markdown("---")

    if st.button(
        "🚀 Predict Total Amount",
        use_container_width=True
    ):

        input_data = pd.DataFrame({
            "Gender": [gender],
            "Age": [age],
            "Product Category": [category],
            "Quantity": [quantity],
            "Price per Unit": [price]
        })

        prediction = model.predict(input_data)[0]

        st.markdown(
            f"""
            <div class="prediction-box">
                <div class="prediction-label">
                    Predicted Total Amount
                </div>

                <div class="prediction-value">
                    ₹{prediction:,.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.success(
            f"Prediction completed for {category} category."
        )


# =========================================================
# DATASET PAGE
# =========================================================

elif page == "📋 Dataset":

    st.markdown(
        '<div class="section-title">📋 Retail Sales Dataset</div>',
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
            int(df.isnull().sum().sum())
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
    """
    <div style="text-align:center; color:#94a3b8; padding:20px;">
        🛍️ <b>Retail Sales Analytics</b><br>
        Built with Python • Pandas • Plotly • Scikit-learn • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)