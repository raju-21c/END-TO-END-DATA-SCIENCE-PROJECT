import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import joblib # For saving the model

# --- 1. Simulate Data Collection ---
np.random.seed(42) # for reproducibility

num_samples = 1000

data = {
    'SquareFootage': np.random.normal(2000, 500, num_samples).astype(int),
    'Bedrooms': np.random.randint(2, 6, num_samples),
    'Bathrooms': np.random.choice([1, 1.5, 2, 2.5, 3], num_samples),
    'YearBuilt': np.random.randint(1950, 2020, num_samples),
    'Location': np.random.choice(['Suburb', 'Urban', 'Rural'], num_samples, p=[0.5, 0.3, 0.2])
}

df = pd.DataFrame(data)

# Generate 'Price' based on features with some noise
df['Price'] = (
    df['SquareFootage'] * 150 +
    df['Bedrooms'] * 15000 +
    df['Bathrooms'] * 10000 +
    (2025 - df['YearBuilt']) * 500 + # Older homes slightly less, new slightly more
    df['Location'].apply(lambda x: {'Suburb': 50000, 'Urban': 100000, 'Rural': -20000}.get(x, 0)) +
    np.random.normal(0, 30000, num_samples) # Add some random noise
).astype(int)

# Ensure prices are not negative (though highly unlikely with this formula)
df['Price'] = df['Price'].apply(lambda x: max(100000, x))

print("Simulated Dataset Head:")
print(df.head())
print("\nDataset Info:")
df.info()
# --- 2. Data Preprocessing ---

# Define features (X) and target (y)
X = df.drop('Price', axis=1)
y = df['Price']

# Identify numerical and categorical features
numerical_features = ['SquareFootage', 'Bedrooms', 'Bathrooms', 'YearBuilt']
categorical_features = ['Location']

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore') # handle_unknown for new categories during inference

# Create a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nShape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)

# --- 3. Model Training ---

# Create the full pipeline: preprocessor + model
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('regressor', LinearRegression())])

# Train the model
model_pipeline.fit(X_train, y_train)

print("\nModel training complete.")
from sklearn.metrics import mean_absolute_error, r2_score

# --- 4. Model Evaluation ---

y_pred = model_pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Evaluation:")
print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
print(f"R-squared (R2): {r2:.4f}")

# Save the trained model
model_filename = 'house_price_model.joblib'
joblib.dump(model_pipeline, model_filename)
print(f"\nModel saved as {model_filename}")
# Filename: app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model
# Ensure the model file is in the same directory or provide the full path
MODEL_PATH = 'house_price_model.joblib'

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file '{MODEL_PATH}' not found. Please run the training script first.")
    # Exit or handle the error appropriately
    exit()

try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

@app.route('/')
def home():
    """Simple home route to check if the API is running."""
    return "House Price Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predicts house price based on input features.
    Expects a JSON payload like:
    {
        "SquareFootage": 2100,
        "Bedrooms": 4,
        "Bathrooms": 2.5,
        "YearBuilt": 2005,
        "Location": "Suburb"
    }
    """
    try:
        data = request.get_json(force=True)
        print(f"Received data for prediction: {data}")

        # Convert input data to DataFrame, ensuring correct order and types
        # It's crucial that the column order matches the training data
        input_df = pd.DataFrame([data])

        # Make prediction
        prediction = model.predict(input_df)[0]
        # Format the prediction to 2 decimal places and return as a string with currency
        formatted_prediction = f"${prediction:,.2f}"

        return jsonify({'predicted_price': formatted_prediction})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # You can specify host='0.0.0.0' to make it accessible from other machines in your network
    # and debug=True for development (do not use in production)
    app.run(host='0.0.0.0', port=5000, debug=True)
