from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
model, all_symptoms, main_df = pickle.load(open("medicine_model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    medicines = []
    precautions = []
    emergency_signs = []
    lifestyle_tips = []
    
    if request.method == "POST":
        selected_symptoms = request.form.getlist("symptoms")
        input_data = [1 if sym in selected_symptoms else 0 for sym in all_symptoms]
        predicted_disease = model.predict([input_data])[0]
        prediction = predicted_disease

        # Extract recommendations
        record = main_df[main_df["Disease"] == predicted_disease].iloc[0]
        medicines = record["Medicines"].split(",") if pd.notna(record["Medicines"]) else []
        precautions = record["Precautions"].split(",") if pd.notna(record["Precautions"]) else []
        emergency_signs = record["Emergency_Signs"].split(",") if pd.notna(record["Emergency_Signs"]) else []
        lifestyle_tips = record["Lifestyle_Tips"].split(",") if pd.notna(record["Lifestyle_Tips"]) else []

    return render_template("index.html", symptoms=all_symptoms, prediction=prediction, 
                           medicines=medicines, precautions=precautions,
                           emergency_signs=emergency_signs, lifestyle_tips=lifestyle_tips)

if __name__ == "__main__":
    app.run(debug=True)