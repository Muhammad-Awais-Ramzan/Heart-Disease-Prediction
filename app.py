
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

knn_model = pickle.load(open('/models/knn_trained_model.pkl', 'rb'))
svc_model = pickle.load(open('/models/svc_trained_model.pkl', 'rb'))
dt_model = pickle.load(open('/models/dt_trained_model.pkl', 'rb'))
cnb_model = pickle.load(open('/models/cnb_trained_model.pkl', 'rb'))
lr_model = pickle.load(open('/models/lr_trained_model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('Home.html')


@app.route('/predict', methods=['POST'])
def predict():

    inputs = []
    inputs.append(request.form['sex'])
    inputs.append(request.form['cp'])
    inputs.append(request.form['restBP'])
    inputs.append(request.form['chol'])
    inputs.append(request.form['fbs'])
    inputs.append(request.form['restecg'])
    inputs.append(request.form['thalach'])
    inputs.append(request.form['exang'])
    inputs.append(request.form['oldpeak'])
    inputs.append(request.form['slope'])
    inputs.append(request.form['ca'])
    inputs.append(request.form['thal'])

    inputs = [int(float(items)) for items in inputs]

    sex = request.form['sex']
    cp = request.form['cp']
    restBP = request.form['restBP']
    chol = request.form['chol']
    fbs = request.form['fbs']
    restecg = request.form['restecg']
    thalach = request.form['thalach']
    exang = request.form['exang']
    oldpeak = request.form['oldpeak']
    slope = request.form['slope']
    ca = request.form['ca']
    thal = request.form['thal']

    final_inputs = [np.array(inputs)]

    prediction_knn = knn_model.predict(final_inputs)
    if prediction_knn[0] == 1:
        knn_result = "Defective Heart"
    if prediction_knn[0] == 0:
        knn_result = "Healthy Heart"

    prediction_svc = svc_model.predict(final_inputs)
    if prediction_svc[0] == 1:
        svc_result = "Defective Heart"
    if prediction_svc[0] == 0:
        svc_result = "Healthy Heart"

    prediction_dt = dt_model.predict(final_inputs)
    if prediction_dt[0] == 1:
        dt_result = "Defective Heart"
    if prediction_dt[0] == 0:
        dt_result = "Healthy Heart"

    prediction_lr = lr_model.predict(final_inputs)
    if prediction_lr[0] == 1:
        lr_result = "Defective Heart"
    if prediction_lr[0] == 0:
        lr_result = "Healthy Heart"

    inputs1 = [int(float(items)) for items in inputs]
    final_inputs1 = [np.array(inputs1)]

    prediction_cnb = cnb_model.predict(final_inputs1)
    if prediction_cnb[0] == 1:
        cnb_result = "Defective Heart"
    if prediction_cnb[0] == 0:
        cnb_result = "Healthy Heart"

    if sex == '1':
        sex = "Male"
    if sex == '0':
        sex = "Female"

    if cp == '0':
        cp = 'ASY'
    if cp == '1':
        cp = 'ATA'
    if cp == '2':
        cp = 'NAP'
    if cp == '3':
        cp = 'TA'

    if fbs == '1':
        fbs = '> 120'

    if restecg == '0':
        restecg = 'LVH'
    if restecg == '1':
        restecg = 'Normal'
    if restecg == '2':
        restecg = 'ST'

    if exang == '0':
        exang = 'No'
    if exang == '1':
        exang = 'Yes'

    if slope == '0':
        slope = 'Down'
    if slope == '1':
        slope = 'Flat'
    if slope == '2':
        slope = 'Up'

    return render_template('Home.html', pred_knn=knn_result, pred_svc=svc_result, pred_dt=dt_result, pred_gnb=cnb_result, pred_lr=lr_result, sex=sex, cp=cp, restBP=restBP, chol=chol, fbs=fbs, restecg=restecg, thalach=thalach, exang=exang, oldpeak=oldpeak, slope=slope, ca=ca, thal=thal)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)
