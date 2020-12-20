import pandas as pd
import numpy as np
import data_cleaner
from joblib import dump, load

def data_predict(file_name):
	X_val = clean_data(file_name, False, False)
	random_forest = load('model.joblib')
	
	y_pred = random_forest.predict(X_val)