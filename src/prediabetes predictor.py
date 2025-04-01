import tkinter as tk
from tkinter import ttk, messagebox
import pickle
import pandas as pd
import os

# Load the default model
model_path = "../models/random_forest.pkl"
if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
else:
    messagebox.showerror("Error", f"Model file {model_path} not found.")

# Define mappings
education_mapping = {
    "Never attended school or only kindergarten": 1,
    "Grades 1 through 8 (Elementary)": 2,
    "Grades 9 through 11 (Some high school)": 3,
    "Grade 12 or GED (High school graduate)": 4,
    "College 1 year to 3 years (Some college or technical school)": 5,
    "College 4 years or more (College graduate)": 6
}

gen_hlth_mapping = {"Excellent": 1, "Very Good": 2, "Good": 3, "Fair": 4, "Poor": 5}

def safe_int(value, field_name):
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"{field_name} must be a valid integer.")

def safe_float(value, field_name):
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"{field_name} must be a valid number.")

def predict():
    try:
        age = safe_int(Age.get(), "Age")
        income = safe_int(Income.get(), "Income")
        education = education_mapping.get(Education.get(), 3)
        height = safe_float(Height.get(), "Height") / 100  # Convert cm to meters
        weight = safe_float(Weight.get(), "Weight")
        BMI = weight / (height ** 2)
        phys_hlth = safe_int(PhysHlth.get(), "Days with poor physical health")
        gen_hlth = gen_hlth_mapping.get(GenHlth.get(), 3)

        inputs = [
            HighBP.get(), HighChol.get(), BMI, Stroke.get(),
            HeartDiseaseorAttack.get(), PhysActivity.get(), HvyAlcoholConsump.get(),
            AnyHealthcare.get(), gen_hlth, phys_hlth,
            DiffWalk.get(), Sex.get(), age, education, income
        ]

        feature_names = [
            "HighBP", "HighChol", "BMI", "Stroke", "HeartDiseaseorAttack", "PhysActivity",
            "HvyAlcoholConsump", "AnyHealthcare", "GenHlth", "PhysHlth", "DiffWalk", "Sex",
            "Age", "Education", "Income"
        ]
        inputs_df = pd.DataFrame([inputs], columns=feature_names)

        prediction = model.predict(inputs_df)[0]

        result, suggestions = (
            ("Diabetic", "To manage diabetes:\n- Eat a balanced, low-sugar diet.\n- Exercise regularly.\n- Monitor blood sugar levels.\n- Avoid smoking and limit alcohol.")
            if prediction == 2 else
            ("Pre-Diabetic", "To prevent diabetes:\n- Maintain a healthy diet with fiber-rich foods.\n- Exercise regularly to maintain a healthy weight.\n- Monitor blood sugar levels.\n- Reduce stress and get regular checkups.")
            if prediction == 1 else
            ("Healthy", "To maintain good health:\n- Continue a balanced diet.\n- Engage in regular physical activity.\n- Maintain a healthy weight.\n- Avoid smoking and excessive alcohol.")
        )

        messagebox.showinfo("Prediction Result", f"The prediction is: {result}\n\n{suggestions}")
    except ValueError as ve:
        messagebox.showerror("Input Error", f"Invalid input: {ve}")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")

root = tk.Tk()
root.title("Diabetes Prediction")

Age, Income, Height, Weight, PhysHlth = (tk.StringVar() for _ in range(5))
HighBP, HighChol, Stroke, HeartDiseaseorAttack, PhysActivity = (tk.IntVar(value=0) for _ in range(5))
HvyAlcoholConsump, AnyHealthcare, DiffWalk, Sex = (tk.IntVar(value=0) for _ in range(4))
GenHlth, Education = (tk.StringVar(value=v) for v in ["Excellent", "Never attended school or only kindergarten"])

fields = ["Age", "Income(USD $)", "Height (cm)", "Weight (kg)", "Days with poor physical health in last 30 days"]
inp_vars = [Age, Income, Height, Weight, PhysHlth]
for i, (label, var) in enumerate(zip(fields, inp_vars)):
    tk.Label(root, text=label+":").grid(row=i, column=0, sticky='w')
    tk.Entry(root, textvariable=var).grid(row=i, column=1)

tk.Label(root, text="General Health:").grid(row=5, column=0, sticky='w')
ttk.Combobox(root, textvariable=GenHlth, values=list(gen_hlth_mapping.keys())).grid(row=5, column=1)

questions = [
    ("Do you have high blood pressure?", HighBP),
    ("Do you have high cholesterol?", HighChol),
    ("Have you ever had a stroke?", Stroke),
    ("Have you ever had a heart attack or heart disease?", HeartDiseaseorAttack),
    ("Did you do any physical activity in the past 30 days?", PhysActivity),
    ("Are you a heavy drinker?", HvyAlcoholConsump),
    ("Do you have healthcare coverage?", AnyHealthcare),
    ("Do you have difficulty walking?", DiffWalk)
]

for i, (text, var) in enumerate(questions, start=6):
    tk.Label(root, text=text).grid(row=i, column=0, sticky='w')
    tk.Radiobutton(root, text="No", variable=var, value=0).grid(row=i, column=1)
    tk.Radiobutton(root, text="Yes", variable=var, value=1).grid(row=i, column=2)

tk.Label(root, text="Education Level:").grid(row=14, column=0, sticky='w')
ttk.Combobox(root, textvariable=Education, values=list(education_mapping.keys())).grid(row=14, column=1)

tk.Button(root, text="Predict", command=predict).grid(row=15, column=0, columnspan=2)

root.mainloop()