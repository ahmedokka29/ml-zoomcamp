import requests
url = 'http://127.0.0.1:9696/predict'


test = {
    "age": 40, "height(cm)": 175,
    "weight(kg)": 85, "waist(cm)": 88,
    "eyesight(left)": 1.2, "eyesight(right)": 1,
    "hearing(left)": 1, "hearing(right)": 1,
    "systolic": 100, "relaxation": 70,
    "fasting blood sugar": 112, "Cholesterol": 295,
    "triglyceride": 184, "HDL": 51,
    "LDL": 209, "hemoglobin": 15.2,
    "Urine protein": 1, "serum creatinine": 0.9,
    "AST": 17, "ALT": 13, "Gtp": 55,
    "dental caries": 0
}

try:
    response = requests.post(url, json=test).json()
    if "smoking_probability" in response and "smoking" in response:
        print(response["smoking_probability"])
        if response["smoking"] == True:
            print("smoking")
        else:
            print("not smoking")
    else:
        print("Response does not contain the expected keys.")
except requests.exceptions.JSONDecodeError as e:
    print("Error decoding JSON response:", e)
