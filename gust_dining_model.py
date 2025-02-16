from pymongo import MongoClient
import pandas as pd
import os
from together import Together
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss
import matplotlib.pyplot as plt

# Load dataset
dataset = pd.read_excel("/mnt/data/dining_info.xlsx")

# Initialize Together AI client
client = Together(api_key='5492062267d446ef604e77b4495550013e28a71c18c2547d4cf72e00bc1fa6d4')  # Replace with actual API key

response = client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    messages=[{"role": "user", "content": "What are some fun things to do in New York?"}],
)
print(response.choices[0].message.content)

# MongoDB connection
client = MongoClient("mongodb+srv://sonyeda601:buodR9tHY0aIjd4A@cluster0.gbw0x.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")  # Replace with actual connection string
db = client["hotel_guests"]
collection = db["dining_info"]
df_from_mongo = pd.DataFrame(list(collection.find()))
df = df_from_mongo.copy()

# Convert dates
for col in ['check_in_date', 'check_out_date', 'order_time']:
    df[col] = pd.to_datetime(df[col])

df['check_in_day'] = df['check_in_date'].dt.dayofweek  # Monday=0, Sunday=6
df['check_out_day'] = df['check_out_date'].dt.dayofweek
df['check_in_month'] = df['check_in_date'].dt.month
df['check_out_month'] = df['check_out_date'].dt.month
df['stay_duration'] = (df['check_out_date'] - df['check_in_date']).dt.days

# Split data
features_df = df[df['order_time'] < '2024-01-01']
train_df = df[(df['order_time'] >= '2024-01-01') & (df['order_time'] <= '2024-10-01')]
test_df = df[df['order_time'] > '2024-10-01']

# Feature engineering
customer_features = features_df.groupby('customer_id').agg(
    total_orders_per_customer=('transaction_id', 'count'),
    avg_spend_per_customer=('price_for_1', 'mean')
).reset_index()

customer_dish = features_df.groupby('customer_id')['dish'].agg(lambda x: x.mode()[0]).reset_index()
cuisine_features = features_df.groupby('Preferred Cusine').agg(
    total_orders_per_cuisine=('transaction_id', 'count')
).reset_index()

cuisine_popular_dish = features_df.groupby('Preferred Cusine')['dish'].agg(lambda x: x.mode()[0]).reset_index()
cuisine_popular_dish.rename(columns={'dish': 'popular_dish_for_this_cuisine'}, inplace=True)

# Merge features
train_df = train_df.merge(customer_features, on='customer_id', how='left')
train_df = train_df.merge(customer_dish.rename(columns={'dish': 'fav_dish_per_customer'}), on='customer_id', how='left')
train_df = train_df.merge(cuisine_features, on='Preferred Cusine', how='left')
train_df = train_df.merge(cuisine_popular_dish, on='Preferred Cusine', how='left')

train_df.drop(['_id', 'transaction_id', 'customer_id', 'price_for_1', 'Qty', 'order_time', 'check_in_date', 'check_out_date'], axis=1, inplace=True)

# One-hot encoding
categorical_cols = ['Preferred Cusine', 'fav_dish_per_customer', 'popular_dish_for_this_cuisine']
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoded_df = pd.DataFrame(encoder.fit_transform(train_df[categorical_cols]), columns=encoder.get_feature_names_out(categorical_cols))
train_df = pd.concat([train_df.drop(columns=categorical_cols), encoded_df], axis=1)

# Process test set
test_df = test_df.merge(customer_features, on='customer_id', how='left')
test_df = test_df.merge(customer_dish.rename(columns={'dish': 'fav_dish_per_customer'}), on='customer_id', how='left')
test_df = test_df.merge(cuisine_features, on='Preferred Cusine', how='left')
test_df = test_df.merge(cuisine_popular_dish, on='Preferred Cusine', how='left')
test_df.drop(['_id', 'transaction_id', 'customer_id', 'price_for_1', 'Qty', 'order_time', 'check_in_date', 'check_out_date'], axis=1, inplace=True)
encoded_test_df = pd.DataFrame(encoder.transform(test_df[categorical_cols]), columns=encoder.get_feature_names_out(categorical_cols))
test_df = pd.concat([test_df.drop(columns=categorical_cols), encoded_test_df], axis=1)

# Encode target
train_df = train_df.dropna(subset=['dish'])
label_encoder = LabelEncoder()
train_df['dish'] = label_encoder.fit_transform(train_df['dish'])
X_train, y_train = train_df.drop(columns=['dish']), train_df['dish']
test_df = test_df.dropna(subset=['dish'])
test_df['dish'] = label_encoder.transform(test_df['dish'])
X_test, y_test = test_df.drop(columns=['dish']), test_df['dish']

# Train XGBoost model
xgb_model = xgb.XGBClassifier(
    objective="multi:softmax",
    eval_metric="mlogloss",
    learning_rate=0.1,  # Adjust this
    max_depth=6,  # Adjust this
    n_estimators=100,  # Adjust this
    subsample=0.8,  # Adjust this
    colsample_bytree=0.8,  # Adjust this
    random_state=42
)
xgb_model.fit(X_train, y_train)

# Evaluate model
y_pred = xgb_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
y_pred_prob = xgb_model.predict_proba(X_test)
print("Log Loss:", log_loss(y_test, y_pred_prob))

# Feature importance
importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': xgb_model.feature_importances_}).sort_values(by='Importance', ascending=False)
xgb.plot_importance(xgb_model, max_num_features=5)
plt.show()
