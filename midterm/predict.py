import pickle as pkl
from flask import Flask
from flask import request
from flask import jsonify
import numpy as np


model_file = "model_xgboost.bin"

with open(model_file, 'rb') as f_in:
    model = pkl.load(f_in)

app = Flask('smoking')


@app.route('/predict', methods=['POST'])
def predict():
    test = request.get_json()
    test["bmi"] = test["weight(kg)"] / (test["height(cm)"]/100)**2

    reshaped_data = np.array(list(test.values()))
    reshaped_data = reshaped_data.reshape(1, -1)
    X = reshaped_data
    y_pred = model.predict_proba(X)[0, 1]
    smoking = model.predict(X)
    result = {
        "smoking_probability": float(y_pred),
        "smoking": bool(smoking)
    }
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)


# waitress-serve --listen=0.0.0.0:9696 predict:app
# docker build -t [containerName] .
# docker run -it --rm -p 9696:9696 [containerName]
