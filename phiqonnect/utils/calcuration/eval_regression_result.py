import math
import numpy as np

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import explained_variance_score

def eval_score(y_true, y_pred):
	"""正解データと予測データからスコアを算出する
        
	Parameters
	----------
	y_true: array
		正解データ
	y_pred: array
		予測データ

	Returns
	-------
	result: dict
		各スコア(r2, mae, rmse, evs)
	"""
	r2 = r2_score(y_true, y_pred)
	mae = mean_absolute_error(y_true, y_pred)
	rmse = np.sqrt(mean_squared_error(y_true, y_pred))
	evs = explained_variance_score(y_true, y_pred)
	
	
	eval_dict = {
		"r2" : r2, 
		"mae" : mae, 
		"rmse" : rmse, 
		"evs": evs,
	}

	return eval_dict