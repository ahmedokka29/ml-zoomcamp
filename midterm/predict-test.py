import requests
url = 'http://127.0.0.1:9696/predict'


test = {'age': 55,'height(cm)': 175,
        'weight(kg)': 60,'waist(cm)': 81.1,
        'eyesight(left)': 1.0,'eyesight(right)': 1.0,
        'hearing(left)': 1,'hearing(right)': 1,
        'systolic': 114,'relaxation': 66,
        'fasting blood sugar': 86,
        'Cholesterol': 212,
        'triglyceride': 57,
        'HDL': 64,'LDL': 137,
        'hemoglobin': 13.9,
        'Urine protein': 1,
        'serum creatinine': 1.0,
        'AST': 18,'ALT': 12,
        'Gtp': 16,'dental caries': 0,
        'bmi': 19.5918}
# smoking:1

test_2 = {"age": 40, "height(cm)": 160,
          "weight(kg)": 55, "waist(cm)": 75.0,
          "eyesight(left)": 1.5, "eyesight(right)": 1.5,
          "hearing(left)": 1, "hearing(right)": 1,
          "systolic": 95, "relaxation": 69,
          "fasting blood sugar": 102, "Cholesterol": 206,
          "triglyceride": 48, "HDL": 79,
          "LDL": 116, "hemoglobin": 12.0,
          "Urine protein": 1, "serum creatinine": 0.6,
          "AST": 24, "ALT": 20, "Gtp": 17,
          "dental caries": 0,
          "bmi": 21.484375}
# smoking:1

response = requests.post(url, json=test).json()
print(response['smoking_probability'])
if response['smoking'] == True:
    print('smoking')
else:
    print('not smoking')
