import pickle as pkl
from flask import Flask
from flask import request
from flask import jsonify


model_file = "midterm/model_xgboost.bin"

with open(model_file,'rb') as f_in:
    model = pkl.load(f_in)
    
app = Flask('smoking')
@app.route('/predict', methods=['POST'])

def predict():
    test = request.get_json()
    
    X = test
    y_pred = model.predict_proba(X)[0,1]
    smoking = y_pred >= 0.5

    result = {
        'smoking_probability': float(y_pred),
        'smoking': bool(smoking)
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=9696)