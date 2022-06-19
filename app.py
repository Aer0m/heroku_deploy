import pickle
import numpy as np
import sys
from sklearn.neighbors import KNeighborsClassifier

from flask import Flask, request, url_for, redirect, render_template

with open('model.pkl', 'rb') as model_pkl:
   knn = pickle.load(model_pkl)

app = Flask(__name__, template_folder='templates')

@app.route('/predict', methods=['POST'])
def predict_bill():
   age = request.form.get('age')
   sex = request.form.get('sex')
   bmi = request.form.get('bmi')
   children = request.form.get('children')
   smoker = request.form.get('smoker')
   region = request.form.get('region')

   if sex == "female":
      sex = 0
   elif sex == "male":
      sex = 1
   if smoker == "yes":
      smoker = 0
   elif smoker == "no":
      smoker = 1
   if region == "southwest":
      region = 0
   elif region == "southeast":
      region = 1
   elif region == "northwest":
      region = 2
   elif region == "northeast":
      region = 3

   # Используем метод модели predict для
   # получения прогноза для неизвестных данных
   unseen = np.array([[age, sex, bmi, children, smoker, region]], dtype=float)
   # unseen = np.nan_to_num(unseen)
   result = knn.predict(unseen)
   # возвращаем результат
   return render_template("index.html", pred='Примерная сумма оплаты {}'.format(result))

@app.route('/predict_api')
def predict_bills():
   age = request.args.get('age')
   sex = request.args.get('sex')
   bmi = request.args.get('bmi')
   children = request.args.get('children')
   smoker = request.args.get('smoker')
   region = request.args.get('region')

# Используем метод модели predict для
# получения прогноза для неизвестных данных
   unseen = np.array([[age, sex, bmi, children, smoker, region]], dtype=float)
   #unseen = np.nan_to_num(unseen)
   result = knn.predict(unseen)
  # возвращаем результат
   return str(result)

@app.route('/')
def home():
    return render_template("index.html")


if __name__ == '__main__':
   app.run(host='0.0.0.0', debug=True, port=5000)