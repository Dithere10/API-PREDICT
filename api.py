from flask import Flask, request, jsonify
import pandas as pd
import joblib
from flask_cors import CORS
# Load model + encoders
bundle = joblib.load("match_prediction.joblib")
model = bundle["model"]
encoders = bundle["encoders"]

app = Flask(__name__)
CORS(app)
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        # Prepare input for prediction
        input_data = pd.DataFrame([{
            "team1": encoders["team1"].transform([data["team1"]])[0],
            "team2": encoders["team2"].transform([data["team2"]])[0],
            "venue": encoders["venue"].transform([data["venue"]])[0],
            "toss_winner": encoders["toss_winner"].transform([data["toss_winner"]])[0],
            "toss_decision": encoders["toss_decision"].transform([data["toss_decision"]])[0]
        }])

        # Predict
        winner_encoded = model.predict(input_data)[0]
        predicted_winner = encoders["winner"].inverse_transform([winner_encoded])[0]

        return jsonify({"predicted_winner": predicted_winner})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
