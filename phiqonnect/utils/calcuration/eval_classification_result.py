import math
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

def eval_score(y_true, y_pred, pos_label=None):
	"""正解データと予測データからスコアを算出する
        
	Parameters
	----------
	y_true: array
		正解データ
	y_pred: array
		予測データ
	pos_label: array
		ラベルを更新する場合に使用

	Returns
	-------
	result: dict
		各スコア(accuracy, precision, recall, specificity, f-measure, confusion_matrix)
	"""
	confusion = confusion_matrix(y_true, y_pred)
	accuracy = accuracy_score(y_true, y_pred)
	
	target = np.unique(y_true)
	if len(np.unique(y_true)) == 2:
		if pos_label is None:
			if target[0] == 0:
				pos_label = target[1]
			else:
				pos_label = target[0]
		precision = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
		recall = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
		if sum([y != 0 for y in y_pred]) == 0:
			specificity = np.nan 
		else:
			specificity = sum([x == y for x, y in zip(y_true, y_pred) if y != 0]) / sum([y != 0 for y in y_pred])
		f = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
	else:
		precision = np.nan
		recall = np.nan
		specificity = np.nan
		f = np.nan
		
	eval_dict = {"accuracy" : accuracy, "precision" : precision, "recall" : recall, "specificity" : specificity, "f-measure" : f, "confusion_matrix": confusion}

	return eval_dict
	
def multi_class_eval_score(y_true, y_pred):
	"""正解データと予測データからスコアを算出する（他クラス分類）
        
	Parameters
	----------
	y_true: array
		正解データ
	y_pred: array
		予測データ

	Returns
	-------
	result: dict
		各スコア(accuracy, precision, recall, specificity, f-measure, confusion_matrix)
	"""
	confusion = confusion_matrix(y_true, y_pred)
	accuracy = accuracy_score(y_true, y_pred)
	precision = precision_score(y_true, y_pred, average=None, zero_division=0)
	recall = recall_score(y_true, y_pred, average=None, zero_division=0)
	# specificity = sum([x == y for x, y in zip(y_true, y_pred) if y == 1]) / sum([y == 1 for y in y_pred])
	f = f1_score(y_true, y_pred, average=None, zero_division=0)
	eval_dict = {"accuracy" : accuracy, "precision" : precision, "recall" : recall, "f-measure" : f, "confusion_matrix": confusion}

	return eval_dict