import pandas as pd
import numpy as np
import math
import itertools
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,8)
from fbprophet import Prophet
import pmdarima as pm
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Ridge
import xgboost as xgb
from math import sqrt
import warnings
warnings.filterwarnings('ignore')
import os


def mean_absolute_percentage_error(y_true, y_pred):
	y_true, y_pred = np.array(y_true), np.array(y_pred)
	return (np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)) * 100	### MAE%


class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


def get_best_params_fb_prophet(train, test, time_col, target):
	df = pd.DataFrame()
	df['ds'] = train[time_col]
	df['y'] = train[target]

	param_grid = {  
	'changepoint_prior_scale': [x for x in np.arange(0.01,0.11,0.01)],
	'seasonality_mode': ['multiplicative']
	}
	# Generate all combinations of parameters
	all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
	error = []  # Store the error for each params here

	# Evaluate all parameters
	for params in all_params:
		m = Prophet(**params)
		with suppress_stdout_stderr():
			m.fit(df)
		future = pd.DataFrame()
		future['ds'] = test[time_col]
		forecast = m.predict(future)
		#print(test, forecast, target)
		mape = mean_absolute_percentage_error(test[target], forecast['yhat'])
		#("MAPE for %s %s" % (target, mape))
		error.append(mape)

	# Find the best parameters
	best_params = all_params[np.argmin(error)]
	return best_params


def gridsearch_rf(train, target):
	from sklearn.model_selection import GridSearchCV
	# Create the parameter grid based on the results of random search 
	param_grid = {
	    'bootstrap': [True],
	    'max_depth': [3,4,5,6],
	    'max_features': ['auto', 'sqrt'],
	    # 'min_samples_leaf': [3, 4, 5],
	    # 'min_samples_split': [8, 10, 12],
	    'n_estimators': [20, 30, 40, 50]
	}
	# Create a based model
	rf = RandomForestRegressor(random_state=42)
	# Instantiate the grid search model
	grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
	                          cv = 3, n_jobs = -1, verbose = 0)
	# Fit the grid search to the data
	grid_search.fit(train, target)
	params = grid_search.best_params_
	#print(params)
	return params


def gridsearch_xgb(train, target):
	from sklearn.model_selection import GridSearchCV
	# Create the parameter grid based on the results of random search 
	param_grid = {
	    'max_depth': [3,4,5,6],
	    'n_estimators': [20, 30, 40, 50]
	}
	# Create a based model
	rf = xgb.XGBRegressor(random_state=42)
	# Instantiate the grid search model
	grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
	                          cv = 3, n_jobs = -1, verbose = 0)

	# Fit the grid search to the data
	grid_search.fit(train, target)
	params = grid_search.best_params_
	#print(params)
	return params


def forecast_using_prophet(train, test, time_col, target):
  train_x, test_x = np.split(train, [int(0.8*len(train))])
  df = pd.DataFrame()
  df['ds'] = train[time_col]
  df['y'] = train[target]
  best_params = get_best_params_fb_prophet(train_x,test_x,time_col,target)
  m = Prophet(**best_params)
  with suppress_stdout_stderr():
    m.fit(df)
  future = pd.DataFrame()
  future['ds'] = test[time_col]
  forecast = m.predict(future)
  result_prophet = pd.DataFrame()
  result_prophet['ds'] = forecast['ds']
  result_prophet['y'] = forecast['yhat']
  return df.append(result_prophet)

def forecast_using_linear_regression(train, test, time_col, target_col):
	feature_names = train.columns.tolist()
	try:
		feature_names.remove(time_col)
		feature_names.remove(target_col)
	except ValueError:
		pass
	target = train[target_col]

	regr = LinearRegression()
	regr.fit(train[feature_names], target)
	predict = regr.predict(test[feature_names])
	test[target_col] = predict
	return train.append(test).reset_index(drop=True)


def forecast_using_ridge_regression(train, test, time_col, target_col):
	feature_names = train.columns.tolist()
	try:
		feature_names.remove(time_col)
		feature_names.remove(target_col)
	except ValueError:
		pass
	target = train[target_col]

	regr = Ridge()
	regr.fit(train[feature_names], target)
	predict = regr.predict(test[feature_names])
	test[target_col] = predict
	return train.append(test).reset_index(drop=True)

def feature_transformation(data, transformation='log'):
	if transformation == 'log':
		data = np.log(data + 1)
	elif transformation == 'exp':
		data = np.exp(data)
	
	return data

def forecast_using_RF(train, test, time_col, target_col):
  feature_names = train.columns.tolist()

  try:
    feature_names.remove(time_col)
    feature_names.remove(target_col)
  except ValueError:
    pass

  target = train[target_col]

  params = gridsearch_rf(train[feature_names].copy(), target)
  regr = RandomForestRegressor(random_state=42, **params)
  regr.fit(train[feature_names], target)
  feats = {} # a dict to hold feature_name: feature_importance
  for feature, importance in zip(feature_names, regr.feature_importances_):
    feats[feature] = importance #add the name/value pair 

  importances_df = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'}).reset_index().rename(columns={'index':'Variable Name'})
  predict = regr.predict(test[feature_names])
  test[target_col] = predict
  return train.append(test).reset_index(drop=True), importances_df


def forecast_using_XGB(train, test, time_col, target_col):
	feature_names = train.columns.tolist()
	try:
		feature_names.remove(time_col)
		feature_names.remove(target_col)
	except ValueError:
		pass
	target = train[target_col]

	params = gridsearch_xgb(train[feature_names].copy(), target)
	regr = xgb.XGBRegressor(random_state=42, **params)
	regr.fit(train[feature_names], target)
	predict = regr.predict(test[feature_names])
	test[target_col] = predict
	return train.append(test).reset_index(drop=True)


def forecast_using_arima(train, test, time_col, target_col):
	# Fit auto_arima function to the dataset 
	model = auto_arima(train[target_col], start_p = 1, start_q = 1, 
							max_p = 3, max_q = 3, m = 12, 
							start_P = 0, seasonal = True, 
							d = None, D = 1, trace = True, 
							error_action ='ignore',   # we don't want to know if an order does not work 
							suppress_warnings = True,  # we don't want convergence warnings 
							stepwise = True)           # set to stepwise 
	
	# Predictions for test dataset
	try:
		predictions = model.predict(n_periods=len(test))
	except Exception as e:
		predictions = [0] * len(test)
	
	test[target_col] = predictions
	return train.append(test).reset_index(drop=True)

def forecast_using_VAR(train, test, time_col, target_col):
  train = train.loc[:, (train != train.iloc[0]).any()]
  feature_names = train.columns.tolist()
  try:
    feature_names.remove(time_col)
  except ValueError:
    pass

  try:
    target_index = feature_names.index(target_col)
    model = VAR(endog=train[feature_names])
    model_fit = model.fit()
    prediction = model_fit.forecast(model_fit.y, steps=len(test))
    target_prediction = []
    for i in prediction:
      target_prediction.append(i[target_index])

  except ValueError:
    test[target_col] = 0
    return train.append(test).reset_index(drop=True)

  test[target_col] = target_prediction
  return train.append(test).reset_index(drop=True)


def model_validation(train, test, time_col, target_col, threshold):
	validation_df = pd.DataFrame()
	y_predicted, feature_importance = forecast_using_RF(train, test, time_col, target_col)
	validation_df[time_col] = y_predicted[time_col]
	validation_df[target_col+'_actual'] = data[data[time_col] <= threshold][target_col]
	validation_df[target_col+'_RF'] = y_predicted[target_col]
	rf_mape = mean_absolute_percentage_error(data[data[time_col] <= threshold][target_col], y_predicted[target_col])

	y_predicted = forecast_using_XGB(data[data[time_col] <= threshold][features].copy(), year, target_col) 
	validation_df[target_col+'_XGB'] = y_predicted[target_col]
	xgb_mape = mean_absolute_percentage_error(data[data[time_col] <= threshold][target_col], y_predicted[target_col])

	print('RF_Validation_mape: ', rf_mape)
	print('XGB_Validation_mape: ', xgb_mape)
	return validation_df, feature_importance


def calculate_model_accuracy(y_true, y_pred, type='regression'):
	if type == 'regression':
		mae = mean_absolute_error(y_true, y_pred)
		mse = mean_squared_error(y_true, y_pred)
		r2 = r2_score(y_true, y_pred)
		mape = mean_absolute_percentage_error(y_true, y_pred)

		return {'MAE': mae, 'MSE': mse, 'R2 Score': r2, 'MAPE': mape}

	elif type == 'classification':
		accuracy = accuracy_score(y_true, y_pred)
		precision = precision_score(y_true, y_pred)
		recall = recall_score(y_true, y_pred)
		f1 = f1_score(y_true, y_pred)
		roc = roc_auc_score(y_true, y_pred)

		return {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1, 'ROC-AUC Score': roc}


# print(calculate_model_accuracy([1,2,1], [1,2,2], type='classification'))