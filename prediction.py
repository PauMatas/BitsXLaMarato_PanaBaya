import pandas as pd
import numpy as np
import data_cleaner as dc
import model_training as mt

def data_predict(file_name):
	X_val = dc.clean_data(file_name, False, False)
	random_forest = mt.get_model()
	feat = mt.get_features()
	X_val = X_val.loc[:,feat]

	y_pred = random_forest.predict(X_val)
	predictions = pd.DataFrame(y_pred)
	predictions.to_csv("predictions.csv" ,header = ['predicted_final_diagnosis_code'], index = False)

	print("Prediction raised no errors.")
	print(y_pred)

data_predict("COPEDICATClinicSympt_DATA_2020-12-17_1642.csv")