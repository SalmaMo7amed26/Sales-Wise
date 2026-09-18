import streamlit as st
import joblib
import pandas as pd
import sqlite3
from datetime import date

st.set_page_config(page_title="SalesWise", page_icon="📦", layout="wide", initial_sidebar_state="expanded")

model = joblib.load("Data/demand_model.pkl")
model_columns = joblib.load("Data/model_columns.pkl")

sub_categories = ["Chairs", "Phones", "Binders", "Storage", "Tables",
                   "Furnishings", "Paper", "Accessories"]

def get_latest_data(sub_category):
    conn = sqlite3.connect("Data/saleswise.db")
    query = """
        SELECT Total_Quantity FROM monthly_demand
        WHERE "Sub-Category" = ?
        ORDER BY Year DESC, Month DESC
        LIMIT 3
    """
    df = pd.read_sql_query(query, conn, params=(sub_category,))
    conn.close()
    if len(df) == 0:
        return 0, 0
    lag_1 = float(df.iloc[0]['Total_Quantity'])
    rolling_3 = float(df['Total_Quantity'].mean())
    return lag_1, rolling_3

today_str = date.today().strftime("%Y-%m-%d")
st.write(f"Welcome, Inventory Manager | Date: {today_str}")

if "order_list" not in st.session_state:
    st.session_state.order_list = []

with st.sidebar:
    st.markdown("## 📦 SalesWise")
    sub_category = st.selectbox("Sub-Category", sub_categories)
    lag_1, rolling_3 = get_latest_data(sub_category)
    st.info(f"Last recorded month: {lag_1:.0f} units\n\nAverage of last 3 months: {rolling_3:.1f} units")

    col_a, col_b = st.columns(2)
    with col_a:
        month = st.selectbox("Month", list(range(1, 13)), index=0)
    with col_b:
        year = st.selectbox("Year", [2018, 2019], index=0)

    had_discount_choice = st.radio("Planning a discount?", ["No", "Yes"], horizontal=True)
    had_discount = 1 if had_discount_choice == "Yes" else 0

    if had_discount == 1:
        discount_percent = st.number_input("Enter discount percentage (%)", min_value=0, max_value=100, value=10, step=1)
        avg_discount = discount_percent / 100
    else:
        avg_discount = 0.0

    predict_btn = st.button("🔍 Forecast Quantity Now")

st.title("Inventory Demand Forecasting Dashboard")
st.write("Select the category and future period from the sidebar to get an instant forecast of the required quantity.")

c1, c2, c3 = st.columns(3)
c1.metric("Selected Category", sub_category)
c2.metric("Last Recorded Sales", f"{lag_1:.0f}")
c3.metric("3-Month Average", f"{rolling_3:.1f}")

if predict_btn:
    input_dict = {col: 0 for col in model_columns}
    input_dict['Year'] = year
    input_dict['Month'] = month
    input_dict['Avg_Discount'] = avg_discount
    input_dict['Had_Discount'] = had_discount
    input_dict['Lag_1'] = lag_1
    input_dict['Rolling_3'] = rolling_3
    col_name = f"Sub-Category_{sub_category}"
    if col_name in input_dict:
        input_dict[col_name] = 1
    input_df = pd.DataFrame([input_dict])[model_columns]
    prediction = round(float(model.predict(input_df)[0]))

    if prediction < 0:
        prediction = 0

    st.session_state.last_prediction = {
        "sub_category": sub_category,
        "month": month,
        "year": year,
        "prediction": prediction,
        "lag_1": lag_1,
        "rolling_3": rolling_3
    }

if "last_prediction" in st.session_state:
    p = st.session_state.last_prediction
    st.success(f"Forecasted quantity for {p['sub_category']} — Month {p['month']}/{p['year']}: {p['prediction']} units")

    if p['lag_1'] > 0:
        change_ratio = (p['prediction'] - p['lag_1']) / p['lag_1']
    else:
        change_ratio = 0

    if change_ratio >= 0.3:
        st.warning("⚠️ Forecasted demand is clearly higher than the last recorded month. It is recommended to review current stock before confirming the order quantity.")
    elif change_ratio <= -0.3:
        st.info("💡 Forecasted demand is clearly lower than the last recorded month. It is recommended to be cautious about ordering a large quantity to avoid excess stock.")
    else:
        st.success("✅ Forecasted demand is close to the usual pattern.")

    st.caption("📊 Model accuracy: average error of approximately ±14 units. This is an estimated figure used to support the ordering decision.")

    st.subheader("Comparison Between Forecast and Historical Data")
    chart_df = pd.DataFrame({
        "Quantity": [p['lag_1'], p['rolling_3'], p['prediction']]
    }, index=["Last Recorded Month", "3-Month Average", "Upcoming Forecast"])
    st.bar_chart(chart_df)

    if st.button("➕ Add this category to today's order"):
        st.session_state.order_list.append({
            "Category": p['sub_category'],
            "Month/Year": f"{p['month']}/{p['year']}",
            "Requested Quantity": p['prediction']
        })
        st.success(f"{p['sub_category']} has been added to the order ✅")

else:
    st.info("👈 Select the data from the sidebar, then click (Forecast Quantity Now)")

st.write("---")
st.subheader("🧾 Today's Order")

if len(st.session_state.order_list) == 0:
    st.write("No items in the order yet.")
else:
    order_df = pd.DataFrame(st.session_state.order_list)
    st.table(order_df)

    remove_index = st.selectbox(
        "Select an item to remove from the order (optional):",
        options=list(range(len(st.session_state.order_list))),
        format_func=lambda i: f"{st.session_state.order_list[i]['Category']} - {st.session_state.order_list[i]['Requested Quantity']} units"
    )
    if st.button("🗑️ Remove this item"):
        st.session_state.order_list.pop(remove_index)
        st.rerun()

    if st.button("🧹 Clear entire order"):
        st.session_state.order_list = []
        st.rerun()

    report_lines = [f"Today's Order Report - SalesWise", f"Date: {today_str}", "=" * 30]
    for item in st.session_state.order_list:
        report_lines.append(f"- {item['Category']} | {item['Month/Year']} | Quantity: {item['Requested Quantity']} units")
    report_text = "\n".join(report_lines)

    st.download_button("⬇️ Download Full Order Report", data=report_text, file_name="todays_order_report.txt")