import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Housing Dashboard", layout="wide")

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    try:
        return pd.read_csv("Housing.csv")
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return pd.DataFrame()

df = load_data()

# ------------------ SIDEBAR ------------------
st.sidebar.title("🏠 Housing App")
page = st.sidebar.radio("Go To", ["Home", "Explore", "Predict"])

# ------------------ HOME PAGE ------------------
def show_home():
    st.title("🏠 Housing Price Prediction Dashboard")

    if df.empty:
        st.error("No data available")
        return

    st.markdown("""
    Welcome to the **Housing Price Prediction System**  
    Analyze housing data and predict prices.
    """)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", len(df))
    col2.metric("Average Price", f"₹ {df['price'].mean():,.0f}")
    col3.metric("Maximum Price", f"₹ {df['price'].max():,.0f}")
    col4.metric("Most Common Bedrooms", int(df["bedrooms"].mode()[0]))

    st.markdown("---")
    st.subheader("📂 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

# ------------------ EXPLORE PAGE ------------------
def show_explore():
    st.title("📊 Explore Housing Dataset")

    if df.empty:
        st.error("No data available")
        return

    col1, col2 = st.columns(2)

    # Bar Chart
    with col1:
        st.subheader("Bedrooms vs Price")
        fig1, ax1 = plt.subplots()
        df.groupby("bedrooms")["price"].mean().plot(kind="bar", ax=ax1)
        st.pyplot(fig1)

    # Pie Chart
    with col2:
        st.subheader("Air Conditioning Distribution")
        fig2, ax2 = plt.subplots()
        df["airconditioning"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax2)
        ax2.set_ylabel("")
        st.pyplot(fig2)

    # Line Chart
    st.subheader("Area vs Price")
    fig3, ax3 = plt.subplots()
    ax3.plot(df["area"], df["price"], "o")
    st.pyplot(fig3)

    # Box Plot
    st.subheader("Bathrooms vs Price")
    fig4, ax4 = plt.subplots()
    sns.boxplot(x="bathrooms", y="price", data=df, ax=ax4)
    st.pyplot(fig4)

    # Distribution
    st.subheader("Price Distribution")
    fig5, ax5 = plt.subplots()
    sns.histplot(df["price"], kde=True, ax=ax5)
    st.pyplot(fig5)

# ------------------ PREDICTION PAGE ------------------
def show_prediction():
    st.title("🔮 House Price Prediction")

    if df.empty:
        st.error("No data available")
        return

    col1, col2 = st.columns(2)

    with col1:
        area = st.number_input("Area", 500, 10000, 1000)
        bedrooms = st.selectbox("Bedrooms", sorted(df["bedrooms"].unique()))
        bathrooms = st.selectbox("Bathrooms", sorted(df["bathrooms"].unique()))

    with col2:
        stories = st.selectbox("Stories", sorted(df["stories"].unique()))
        parking = st.selectbox("Parking", sorted(df["parking"].unique()))
        furnishing = st.selectbox("Furnishing", df["furnishingstatus"].unique())

    df_model = df.copy()

    df_model["furnishingstatus"] = df_model["furnishingstatus"].map({
        "furnished": 2,
        "semi-furnished": 1,
        "unfurnished": 0
    })

    X = df_model[["area", "bedrooms", "bathrooms", "stories", "parking", "furnishingstatus"]]
    y = df_model["price"]

    model = LinearRegression()
    model.fit(X, y)

    furnishing_map = {
        "furnished": 2,
        "semi-furnished": 1,
        "unfurnished": 0
    }

    input_data = [[
        area,
        bedrooms,
        bathrooms,
        stories,
        parking,
        furnishing_map[furnishing]
    ]]

    if st.button("Predict Price 💰"):
        prediction = model.predict(input_data)[0]
        st.success(f"Estimated Price: ₹ {prediction:,.0f}")

# ------------------ NAVIGATION ------------------
if page == "Home":
    show_home()

elif page == "Explore":
    show_explore()

elif page == "Predict":
    show_prediction()