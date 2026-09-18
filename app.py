from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("Data/demand_model.pkl")
model_columns = joblib.load("Data/model_columns.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    sub_category = data.get('sub_category')
    avg_discount = data.get('avg_discount', 0)
    had_discount = data.get('had_discount', 0)
    lag_1 = data.get('lag_1')
    rolling_3 = data.get('rolling_3')
    year = data.get('year')
    month = data.get('month')

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

    prediction = model.predict(input_df)[0]

    return jsonify({
        "sub_category": sub_category,
        "predicted_quantity": round(float(prediction), 2)
    })

if __name__ == '__main__':
    app.run(debug=True)
    