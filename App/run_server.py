import dill
import pandas as pd

dill._dill._reverse_typemap['ClassType'] = type
# import cloudpickle
from flask import Flask, request, jsonify

# initialize our Flask application and the model
app = Flask(__name__)
model = None


def load_model(model_path):
    # load the pre-trained model
    global model
    with open(model_path, 'rb') as f:
        model = dill.load(f)


@app.route("/", methods=["GET"])
def general():
    return "Welcome to fraudelent prediction process"


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}
    # ensure an image was properly uploaded to our endpoint
    if request.method == "POST":
        Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, \
        DiabetesPedigreeFunction, Age = 0, 0, 0, 0, 0, 0, 0, 0
        request_json = request.get_json()

        if request_json['Pregnancies']:
            Pregnancies = request_json['Pregnancies']

        if request_json['Glucose']:
            Glucose = request_json['Glucose']

        if request_json['BloodPressure']:
            BloodPressure = request_json['BloodPressure']

        if request_json['SkinThickness']:
            SkinThickness = request_json['SkinThickness']

        if request_json['Insulin']:
            Insulin = request_json['Insulin']

        if request_json['BMI']:
            BMI = request_json['BMI']

        if request_json['DiabetesPedigreeFunction']:
            DiabetesPedigreeFunction = request_json['DiabetesPedigreeFunction']

        if request_json['Age']:
            Age = request_json['Age']

        preds = model.predict_proba(pd.DataFrame({'Pregnancies': [Pregnancies],
                                                  'Glucose': [Glucose],
                                                  'BloodPressure': [BloodPressure],
                                                  'SkinThickness': [SkinThickness],
                                                  'Insulin': [Insulin],
                                                  'BMI': [BMI],
                                                  'DiabetesPedigreeFunction': [DiabetesPedigreeFunction],
                                                  'Age': [Age]}))
        data['predictions'] = preds[:, 1][0]
        data['success'] = True

    # return the data dictionary as a JSON response
    return jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading the model and Flask starting server..."
           "please wait until server has fully started"))
    modelpath = "./models/randfor_pipeline.dill"
    load_model(modelpath)
    app.run()
