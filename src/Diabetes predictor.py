import pickle
import pandas as pd
import tkinter as tk
from tkinter import messagebox

# Load trained models
with open('../models/random_forest.pkl', 'rb') as f:
    model_rf = pickle.load(f)

with open('../models/xgb_classifier.pkl', 'rb') as f:
    model_xgb = pickle.load(f)

# Define the prediction function
valid_ranges = {
    "HighBP": [0, 1],
    "HighChol": [0, 1],
    "Stroke": [0, 1],
    "HeartDiseaseorAttack": [0, 1],
    "PhysActivity": [0, 1],
    "HvyAlcoholConsump": [0, 1],
    "AnyHealthcare": [0, 1],
    "DiffWalk": [0, 1],
    "Sex": [0, 1],
    "GenHlth": [1, 5],
    "PhysHlth": [0, 30],
    "Age": [1, 13],
    "Education": [1, 6],
    "Income": [1, 8]
}


# Define the prediction function with validation
def predict_diabetes():
    try:
        feature_dict = {}

        for var_name, entry_widget in entries.items():
            value = entry_widget.get()

            # Validate numerical values
            if var_name == "BMI":
                value = float(value)
                if value <= 0:
                    raise ValueError(f"{var_name} must be a positive number.")
            else:
                value = int(value)
                if var_name in valid_ranges and value not in range(valid_ranges[var_name][0],
                                                                   valid_ranges[var_name][1] + 1):
                    raise ValueError(
                        f"{var_name} must be in range {valid_ranges[var_name][0]} to {valid_ranges[var_name][1]}.")

            feature_dict[var_name] = value

        # Convert dictionary to DataFrame
        features_df = pd.DataFrame([feature_dict])

        # Select model
        selected_model = model_var.get()

        if selected_model == "Random Forest":
            prediction = model_rf.predict(features_df)[0]
        else:
            prediction = model_xgb.predict(features_df)[0]

        # Generate result and suggestions
        if prediction == 1:
            result = "Diabetic"
            suggestions = (
                "To manage diabetes:\n"
                "- Eat a balanced, low-sugar diet.\n"
                "- Exercise regularly.\n"
                "- Monitor blood sugar levels.\n"
                "- Avoid smoking and limit alcohol."
            )
        else:
            result = "Non-Diabetic"
            suggestions = (
                "To maintain a healthy lifestyle:\n"
                "- Continue a balanced diet.\n"
                "- Engage in regular exercise.\n"
                "- Maintain a healthy weight.\n"
                "- Avoid smoking and excessive alcohol."
            )

        # Show result
        messagebox.showinfo("Prediction Result", f"The prediction is: {result}\n\n{suggestions}")

    except ValueError as e:
        messagebox.showerror("Input Error", str(e))


# Create Tkinter window
root = tk.Tk()
root.title("Diabetes Prediction")
root.geometry("450x650")

# Create labels and entry fields
features = [
    ("HighBP (0 or 1)", "HighBP"),
    ("HighChol (0 or 1)", "HighChol"),
    ("BMI (Positive Value)", "BMI"),
    ("Stroke (0 or 1)", "Stroke"),
    ("HeartDiseaseorAttack (0 or 1)", "HeartDiseaseorAttack"),
    ("PhysActivity (0 or 1)", "PhysActivity"),
    ("HvyAlcoholConsump (0 or 1)", "HvyAlcoholConsump"),
    ("AnyHealthcare (0 or 1)", "AnyHealthcare"),
    ("GenHlth (1-5)", "GenHlth"),
    ("PhysHlth (0-30)", "PhysHlth"),
    ("DiffWalk (0 or 1)", "DiffWalk"),
    ("Sex (0 for Female, 1 for Male)", "Sex"),
    ("Age (1-13)", "Age"),
    ("Education (1-6)", "Education"),
    ("Income (1-8)", "Income")
]

# Dictionary for entry widgets
entries = {}

# Create entry fields dynamically
for i, (label_text, var_name) in enumerate(features):
    label = tk.Label(root, text=label_text)
    label.grid(row=i, column=0, padx=10, pady=5, sticky="w")
    entry = tk.Entry(root)
    entry.grid(row=i, column=1, padx=10, pady=5)
    entries[var_name] = entry

# Model selection dropdown
model_var = tk.StringVar(root)
model_var.set("Random Forest")  # Default selection

model_label = tk.Label(root, text="Select Model:")
model_label.grid(row=len(features), column=0, padx=10, pady=5, sticky="w")

model_dropdown = tk.OptionMenu(root, model_var, "Random Forest", "XGBoost")
model_dropdown.grid(row=len(features), column=1, padx=10, pady=5)

# Prediction button
predict_button = tk.Button(root, text="Predict", command=predict_diabetes)
predict_button.grid(row=len(features) + 1, column=0, columnspan=2, pady=20)

# Run Tkinter main loop
root.mainloop()