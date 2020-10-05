import boto3
import numpy as np
import pickle
import json
import pandas as pd
from io import BytesIO
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier

def get_model():
    bucket = boto3.resource("s3").Bucket("srvlss-lambda")
    with BytesIO() as model_fileobject:
        bucket.download_fileobj(Key="model/mlp.pkl", Fileobj=model_fileobject)
        model = joblib.load(model_fileobject)
    return model


def predict(event):
   # print(event)
    body = event.get('body')
    data = json.loads(body)
    arr = data.get('symptoms')
    data = np.array([arr])
    
    #record = data.transpose()
    #print(data)

    clf = get_model()
    labels = np.reshape(clf.classes_, (1,801))[0]
    probability_lists = clf.predict_proba(data)
    for resp in probability_lists:

            dict = {}
            for l, p in zip(labels, resp):
                dict[l] = p

            sorted_predictions = sorted(dict.items(), key=lambda x: x[1], reverse=True)

            first = sorted_predictions[0]
            second = sorted_predictions[1]
            third = sorted_predictions[2]
            fourth = sorted_predictions[3]
            fifth = sorted_predictions[4]

            return {"results": [{"condition": first[0], "probability": first[1]}, {"condition": second[0], "probability": second[1]}, {"condition": third[0], "probability": third[1]}, {"condition": fourth[0], "probability": fourth[1]}, {"condition": fifth[0], "probability": fifth[1]}]}



def lambda_handler(event, context):
    # We need to predict something from the event - i.e our payload
    predictions = predict(event)
    result = json.dumps(predictions)
    return {"statusCode": 200,
            "body": result}
