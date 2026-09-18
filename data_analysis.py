import pandas as pd

df = pd.read_excel("Data/Dataset.xls")

print(df.head())
print(df.info())
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month

# هل فيه خصم أو لا (كعمود مساعد)
df['Has_Discount'] = (df['Discount'] > 0).astype(int)

print(df[['Product Name', 'Category', 'Sub-Category']].nunique())
monthly = df.groupby(['Sub-Category', 'Year', 'Month']).agg(
    Total_Quantity=('Quantity', 'sum'),
    Avg_Discount=('Discount', 'mean'),
    Had_Discount=('Has_Discount', 'max')
).reset_index()

print(monthly.head(15))
print(monthly.shape)
import sqlite3

conn = sqlite3.connect("Data/saleswise.db")

# نحفظ الداتا الخام المنضفة
df.to_sql("raw_sales", conn, if_exists="replace", index=False)

# نحفظ الداتا المجمعة شهرياً (اللي هنستخدمها في الموديل)
monthly.to_sql("monthly_demand", conn, if_exists="replace", index=False)

conn.close()
print("Database created successfully ✅")
monthly = monthly.sort_values(['Sub-Category', 'Year', 'Month'])

# الكمية في الشهر اللي فات لنفس الفئة
monthly['Lag_1'] = monthly.groupby('Sub-Category')['Total_Quantity'].shift(1)

# متوسط آخر 3 شهور
monthly['Rolling_3'] = monthly.groupby('Sub-Category')['Total_Quantity'].shift(1).rolling(3).mean()

# نشيل أول صفوف كل فئة (مفيهاش lag كافي)
monthly = monthly.dropna()

print(monthly.head(10))
print(monthly.shape)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# نحول Sub-Category لأرقام (عشان الموديل يفهمها)
monthly_encoded = pd.get_dummies(monthly, columns=['Sub-Category'])

# X = كل الأعمدة إلا الهدف (Total_Quantity)
X = monthly_encoded.drop(columns=['Total_Quantity'])
y = monthly_encoded['Total_Quantity']

# تقسيم Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تدريب الموديل
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# التقييم
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
import joblib

joblib.dump(model, "Data/demand_model.pkl")
joblib.dump(X.columns.tolist(), "Data/model_columns.pkl")

print("Model saved successfully ✅")