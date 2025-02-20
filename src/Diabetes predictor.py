import pickle
import pandas as pd
import tkinter as tk
from tkinter import messagebox

# Load trained models
with open('models/random_forest.pkl', 'rb') as f:
    model_rf = pickle.load(f)

with open('models/xgb_classifier.pkl', 'rb') as f:
    model_xgb = pickle.load(f)

# Define the prediction function
def predict_diabetes():
    try:
        # Gather input from entry fields
        feature_dict = {
            'HighBP': int(entry_highbp.get()),
            'HighChol': int(entry_highchol.get()),
            'BMI': float(entry_bmi.get()),
            'Stroke': int(entry_stroke.get()),
            'HeartDiseaseorAttack': int(entry_heart.get()),
            'PhysActivity': int(entry_physact.get()),
            'HvyAlcoholConsump': int(entry_alcohol.get()),
            'AnyHealthcare': int(entry_healthcare.get()),
            'GenHlth': int(entry_genhealth.get()),
            'PhysHlth': int(entry_physhealth.get()),
            'DiffWalk': int(entry_diffwalk.get()),
            'Sex': int(entry_sex.get()),
            'Age': int(entry_age.get()),
            'Education': int(entry_education.get()),
            'Income': int(entry_income.get())
        }

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

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid input values.")

# Create Tkinter window
root = tk.Tk()
root.title("Diabetes Prediction")
root.geometry("450x650")

# Create labels and entry fields
features = [
    ("HighBP (0 or 1)", "highbp"),
    ("HighChol (0 or 1)", "highchol"),
    ("BMI", "bmi"),
    ("Stroke (0 or 1)", "stroke"),
    ("HeartDiseaseorAttack (0 or 1)", "heart"),
    ("PhysActivity (0 or 1)", "physact"),
    ("HvyAlcoholConsump (0 or 1)", "alcohol"),
    ("AnyHealthcare (0 or 1)", "healthcare"),
    ("GenHlth (1-5)", "genhealth"),
    ("PhysHlth (0-30)", "physhealth"),
    ("DiffWalk (0 or 1)", "diffwalk"),
    ("Sex (0 for Female, 1 for Male)", "sex"),
    ("Age (1-13)", "age"),
    ("Education (1-6)", "education"),
    ("Income (1-8)", "income")
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

# Assign entries to variables
entry_highbp = entries["highbp"]
entry_highchol = entries["highchol"]
entry_bmi = entries["bmi"]
entry_stroke = entries["stroke"]
entry_heart = entries["heart"]
entry_physact = entries["physact"]
entry_alcohol = entries["alcohol"]
entry_healthcare = entries["healthcare"]
entry_genhealth = entries["genhealth"]
entry_physhealth = entries["physhealth"]
entry_diffwalk = entries["diffwalk"]
entry_sex = entries["sex"]
entry_age = entries["age"]
entry_education = entries["education"]
entry_income = entries["income"]

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
