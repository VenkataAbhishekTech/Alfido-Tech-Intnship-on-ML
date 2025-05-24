import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load
df = pd.read_csv('data_house.csv')

# 2. Inspect columns
print("Columns:", df.columns.tolist())

# 3. Target
y = df['price']

# 4. Optional: derive date features
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['year']  = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day']   = df['date'].dt.day

# 5. Build X by selecting numeric cols and dropping price
X = df.select_dtypes(include=[np.number]).drop(['price'], axis=1)

# 6. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Fit
model = LinearRegression()
model.fit(X_train, y_train)

# 8. Predict & evaluate
y_pred = model.predict(X_test)
print('MSE:', mean_squared_error(y_test, y_pred))
print('R²:',   r2_score(y_test, y_pred))
