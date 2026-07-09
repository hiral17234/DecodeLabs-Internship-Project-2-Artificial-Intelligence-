import joblib
import numpy as np

# Load Saved Model
model = joblib.load("../models/iris_model.pkl")

# Load Saved Scaler
scaler = joblib.load("../models/scaler.pkl")

# Flower Names
flower_names = [
    "🌸 Iris Setosa",
    "🌺 Iris Versicolor",
    "🌼 Iris Virginica"
]

print("=" * 50)
print("      IRIS FLOWER PREDICTION SYSTEM")
print("=" * 50)

try:

    sepal_length = float(input("Enter Sepal Length (cm): "))
    sepal_width = float(input("Enter Sepal Width (cm): "))
    petal_length = float(input("Enter Petal Length (cm): "))
    petal_width = float(input("Enter Petal Width (cm): "))

    # Create Input Array
    sample = np.array([
        [
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]
    ])

    # Scale Input
    sample = scaler.transform(sample)

    # Predict
    prediction = model.predict(sample)

    print("\n" + "=" * 50)
    print("Prediction Completed Successfully!")
    print("=" * 50)

    print(f"\nPredicted Flower : {flower_names[prediction[0]]}")

except ValueError:
    print("\nPlease enter valid numeric values only!")

print("\nThank you for using the Iris Flower Prediction System.")
