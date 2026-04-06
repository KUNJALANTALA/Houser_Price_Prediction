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
page = st.sidebar.radio("Go To", ["Home", "Explore", "Predict", "scanner"])


# ------------------ HOME PAGE ------------------
def show_home():
    st.title("🏠 Housing Price Prediction Dashboard")

    if df.empty:
        st.error("No data available")
        return

    st.markdown(
        """
    Welcome to the **Housing Price Prediction System**  
    Analyze housing data and predict prices.
    """
    )

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

    df_model["furnishingstatus"] = df_model["furnishingstatus"].map(
        {"furnished": 2, "semi-furnished": 1, "unfurnished": 0}
    )

    X = df_model[
        ["area", "bedrooms", "bathrooms", "stories", "parking", "furnishingstatus"]
    ]
    y = df_model["price"]

    model = LinearRegression()
    model.fit(X, y)

    furnishing_map = {"furnished": 2, "semi-furnished": 1, "unfurnished": 0}

    input_data = [
        [area, bedrooms, bathrooms, stories, parking, furnishing_map[furnishing]]
    ]

    if st.button("Predict Price 💰"):
        prediction = model.predict(input_data)[0]
        st.success(f"Estimated Price: ₹ {prediction:,.0f}")


# ------------------ BULK SCANNER PAGE ------------------
def show_bulkscanner():
    st.title("📂 Bulk House Price Prediction")

    if df.empty:
        st.error("No data available")
        return

    # ------------------ SAMPLE FILE DOWNLOADS ------------------
    st.subheader("📥 Download Sample Files")

    sample_df = pd.DataFrame({
        "area": [1200, 1500],
        "bedrooms": [2, 3],
        "bathrooms": [2, 2],
        "stories": [1, 2],
        "parking": [1, 2],
        "furnishingstatus": ["furnished", "semi-furnished"]
    })

    col1, col2, col3 = st.columns(3)

    # CSV
    with col1:
        csv_data = sample_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇ Download CSV file",
            data=csv_data,
            file_name="sample_house_data.csv",
            mime="text/csv"
        )

    # Excel
    with col2:
        import io
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            sample_df.to_excel(writer, index=False, sheet_name="SampleData")
        excel_data = excel_buffer.getvalue()

        st.download_button(
            label="⬇ Download Excel file",
            data=excel_data,
            file_name="sample_house_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # JSON
    with col3:
        json_data = sample_df.to_json(orient="records", indent=2)
        st.download_button(
            label="⬇ Download JSON file",
            data=json_data,
            file_name="sample_house_data.json",
            mime="application/json"
        )

    st.markdown("---")

    # ------------------ FILE UPLOAD ------------------
    uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        bulk_df = pd.read_csv(uploaded_file)

        st.subheader("📊 Uploaded Data Preview")
        st.dataframe(bulk_df.head())

        expected_columns = [
            "area", "bedrooms", "bathrooms",
            "stories", "parking", "furnishingstatus"
        ]

        if all(col in bulk_df.columns for col in expected_columns):

            st.success("✅ File format is correct!")

            furnishing_map = {
                "furnished": 2,
                "semi-furnished": 1,
                "unfurnished": 0
            }

            bulk_df["furnishingstatus"] = bulk_df["furnishingstatus"].map(furnishing_map)

            # Train model
            df_model = df.copy()
            df_model["furnishingstatus"] = df_model["furnishingstatus"].map(furnishing_map)

            X = df_model[[
                "area", "bedrooms", "bathrooms",
                "stories", "parking", "furnishingstatus"
            ]]
            y = df_model["price"]

            model = LinearRegression()
            model.fit(X, y)

            if st.button("🚀 Predict Bulk Prices"):
                try:
                    predictions = model.predict(bulk_df[X.columns])
                    bulk_df["Predicted_Price"] = predictions

                    st.subheader("📈 Prediction Results")
                    st.dataframe(bulk_df)

                    # Download results
                    result_csv = bulk_df.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        label="⬇ Download Results",
                        data=result_csv,
                        file_name="bulk_predictions.csv",
                        mime="text/csv"
                    )

                except Exception as e:
                    st.error(f"❌ Error: {e}")

        else:
            st.error("❌ CSV must contain:")
            st.write(expected_columns)            
# ------------------ NAVIGATION ------------------
if page == "Home":
    show_home()

elif page == "Explore":
    show_explore()

elif page == "Predict":
    show_prediction()

elif page == "scanner":
    show_bulkscanner()
