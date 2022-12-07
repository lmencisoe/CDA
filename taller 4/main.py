import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay, precision_score, recall_score, f1_score, auc
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import warnings
import re
from fastapi import FastAPI


df_2 = pd.read_json('https://raw.githubusercontent.com/lmencisoe/CDA/main/taller%204/DataSet_Entrenamiento_v2.json')
json_modelo = joblib.load('xgb_best_hp.pkl')
mejor_modelo = joblib.load('pipeline_seleccionado.pkl')

def entrenar(data):
	with open("version.txt", "r") as version:
		model_version = version.read()
	data_f = data.copy()
	data_f['target'] = np.where(data_f['Churn']!= 'Yes', 1, 0)
	Y_total = data_f['target'].astype(float)
	mejor_modelo = joblib.load('pipeline_seleccionado.pkl')
	data_f['TotalCharges'] = data_f['TotalCharges'].replace("", 0)
	data_f['TotalCharges'] = data_f['TotalCharges'].astype(float)
	X_total = data_f.drop(['Churn', 'customerID','target'], axis=1)
	X_train, X_test, y_train, y_test = train_test_split(X_total, Y_total, test_size=0.2, random_state=2022)
	cat_features = X_total.select_dtypes(exclude=["float64", "int64"]).columns.to_list()
	numeric_features = X_total.select_dtypes(["float64", "int64"]).columns.to_list()

	categorical_transformer = Pipeline(
			steps=[
				('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
				('encoder', OneHotEncoder())
			]
		)    

	numeric_transformer = Pipeline(
			steps=[
				('imputer', SimpleImputer(strategy='most_frequent')),
				('scaler', StandardScaler())
			]
		)

	preprocessor = ColumnTransformer(
			transformers=[
				('numerical', numeric_transformer, numeric_features),
				('categorical', categorical_transformer, cat_features)
			]
		)
	xgb_pipeline = Pipeline(
			[
				("preprocessor", preprocessor),
				(
					"classifier", XGBClassifier(**json_modelo),
				),
			]
		)
	fit_pipe = xgb_pipeline.fit(X_train, y_train)
	resul1 = " ".join(['AUC anterior modelo train:' , mejor_modelo.score(X_train, y_train).astype(str)])
	resul2 = " ".join(['AUC anterior modelo test:' , mejor_modelo.score(X_test, y_test).astype(str)])
	resul3 = " ".join(['AUC nuevo modelo train:' , fit_pipe.score(X_train, y_train).astype(str)])
	resul4 = " ".join(['AUC nuevo modelo test:' , fit_pipe.score(X_test, y_test).astype(str)])
	joblib.dump(fit_pipe, f'pipeline_seleccionado_{model_version}.pkl')
	with open("version.txt", "w") as version:
		version.write(str(int(model_version)+1))

	return resul1, resul2, resul3, resul4  


with open("version.txt", "r") as version:
	model_version = version.read()

last_model = joblib.load(f'pipeline_seleccionado_{model_version}.pkl')
df_prediccion = pd.read_json('https://raw.githubusercontent.com/lmencisoe/CDA/main/taller%204/DataSet_Prediccion.json')

def predict(data):
	data_f = data.copy()
	data_f['TotalCharges'] = data_f['TotalCharges'].replace("", 0)
	data_f['TotalCharges'] = data_f['TotalCharges'].astype(float)
	X_total = data_f.drop(['customerID'], axis=1)

	predicciones_proba = last_model.predict_proba(X_total)
	predicciones = last_model.predict(X_total)

	return pd.DataFrame(
		{
			"Prob_0":predicciones_proba[:,0],
			"Prob_1":predicciones_proba[:,1],
			"Predicción":predicciones
		}
	)

	# return predicciones
app = FastAPI()

@app.post("/train")
def train(json):
	df_2 = pd.read_json(json)
	re = entrenar(df_2)
	return {"message": "Training complete {re}".format(re)}

@app.post("/predict")
def predict(json):
	df_prediccion = pd.read_json(json)
	result_predict = predict(df_prediccion)
	return {"message": "Prediction complete {result_predict}".format(result_predict)}

