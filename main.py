import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import io
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

from flask import Flask, request, jsonify

def preprocessingData(frame):
    frame = frame.drop(columns=['MiscFeature','Fence','PoolQC','Alley'])
    frame_object = frame.select_dtypes(include='object').copy()
    frame_number = frame.select_dtypes(exclude='object').copy()

    frame_object = pd.get_dummies(frame_object)
    frame_number = frame_number.fillna(frame_number.median())

    frame_new = pd.concat([frame_object,frame_number],axis=1)

    return frame_new

model = keras.models.load_model("homeprice.h5")
df = pd.read_csv("train.csv")
df_new = preprocessingData(df)

data = pd.read_csv("Data Corelation Result.csv",index_col="Unnamed: 0")
df_use = df_new.loc[:, data.index]

normZ_Score = {}
for d in df_use.columns:
    normZ_Score[d] = {'mean':[]}
    normZ_Score[d]['mean']=df_use[d].mean()
    normZ_Score[d]['std'] =df_use[d].mean()

def actual_Price(x):
    meanX = normZ_Score["SalePrice"]['mean']
    stdX  = normZ_Score["SalePrice"]['std']
    return (x*stdX) + meanX

def predict(x):
    return actual_Price(model.predict(x))

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        dataAdd = df.drop(columns=['SalePrice'])

        dt = pd.read_json(request.json, orient ='index')

        if dt.shape[1]==1:
            dt = dt.T

        dt = dt.loc[:,dataAdd.select_dtypes(exclude='object').columns].apply(pd.to_numeric)
        
        dt = pd.concat([dataAdd,dt],axis=0).reset_index(drop=True)
        dt_new = preprocessingData(dt).iloc[len(dataAdd):,:]

        dataPred = [x for x in data.index if x!="SalePrice"]
        dt_use = dt_new.loc[:, dataPred]
        
        print(dt_use.columns)
        print(df_use.columns)
        dt_use
        for d in dt_use.columns:
            dt_use[d] = (dt_use[d]-normZ_Score[d]['mean']) / normZ_Score[d]['std']

        pred = actual_Price(model.predict(dt_use.values)).reshape(dt_use.shape[0])
        return jsonify(pred.tolist())

    return "OK"


if __name__ == "__main__":
    app.run(debug=True)