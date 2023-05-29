from matplotlib.ticker import MultipleLocator
import seaborn as sns
import matplotlib.pyplot as plt

from ..calcuration.pca import *
from ..calcuration.eval_regression_result import *
from ..dataset_utils.load_regression_datasets import *

def plot_kernel_matrix(kernel_matrix, title=None, save_file=None, figsize=(5, 5), **args):
	"""カーネル行列をプロットする
        
	Parameters
	----------
	kernel_matrix: array
		カーネル行列
	title: string
		タイトル
	save_file: string
		ファイルとして保存する場合はファイルパス
	figsize: (int, int)
		Figureのサイズ
	
	Returns
	-------
	None
	"""
	fig, axes = plt.subplots(nrows=1, ncols=1, figsize=figsize)
	axes.xaxis.set_major_locator(MultipleLocator(5))
	axes.yaxis.set_major_locator(MultipleLocator(5))
	im = axes.imshow(np.asmatrix(kernel_matrix), interpolation='nearest', origin='upper', cmap='bwr', vmin=0, vmax=1)
	fig.colorbar(im, shrink=0.85)
	if title is not None: axes.set_title(f'kernel: {title}')
	if save_file is not None: plt.savefig(f'{save_file}')
	plt.show()
	
def plot_predict(predicted_value, correct_value=None, title=None, save_file=None, figsize=(7, 5), target_scaler=None, **args):
	"""予測値をプロットする
        
	Parameters
	----------
	predicted_value: array
		予測値
	correct_value: array
		正解値
	title: string
		タイトル
	save_file: string
		ファイルとして保存する場合はファイルパス
	figsize: (int, int)
		Figureのサイズ
	target_scaler: scaler
		スケーラー
	
	Returns
	-------
	None
	"""
	if target_scaler is not None:
		predicted_value = original_target(predicted_value, target_scaler)
		if correct_value is not None: correct_value = original_target(correct_value, target_scaler)
		
	fig, axes = plt.subplots(nrows=1, ncols=1, figsize=figsize)
	if correct_value is not None: axes.plot(list(range(0, len(correct_value))), correct_value, label="y_true", color="r")
	axes.plot(list(range(0, len(predicted_value))), predicted_value, label="y_pred", color="b")
	plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1,), borderaxespad=0, ncol=2)
	
	if title is not None: axes.set_title(f'predicted: {title}')
	if save_file is not None: plt.savefig(f'{save_file}')
	plt.show()
	
def plot_parity_plot(predicted_value, correct_value, title=None, save_file=None, figsize=(5, 4), **args):
	"""予測値を散布図でプロットする
        
	Parameters
	----------
	predicted_value: array
		予測値
	correct_value: array
		正解値
	title: string
		タイトル
	save_file: string
		ファイルとして保存する場合はファイルパス
	figsize: (int, int)
		Figureのサイズ
	target_scaler: scaler
		スケーラー
	
	Returns
	-------
	None
	"""
	fig, axes = plt.subplots(nrows=1, ncols=1, figsize=figsize)
	axes.scatter(correct_value, predicted_value, color="b", marker="o")
	
	from sklearn.linear_model import LinearRegression
	lr = LinearRegression()
	lr.fit(predicted_value.reshape(-1, 1), correct_value.reshape(-1, 1))
	
	print(f'corrcoef: {np.corrcoef(predicted_value.reshape(1, -1), correct_value.reshape(1, -1))[0, 1]}')
	print(f'y = {lr.coef_[0][0]}x + {lr.intercept_[0]}')
	
	axes.plot(np.arange(-1.0, 1.1, 0.1).reshape(-1, 1), lr.predict(np.arange(-1.0, 1.1, 0.1).reshape(-1, 1)), color='r')
	
	plt.xlabel('y_true')
	plt.ylabel('y_pred') 
	
	if title is not None: axes.set_title(f'predicted: {title}')
	if save_file is not None: plt.savefig(f'{save_file}')
	plt.show()
	
def original_target_result(target_scaler, predicted_value, correct_value, **args):
	"""元のスケールで結果を取得する
        
	Parameters
	----------
	target_scaler: scaler
		スケーラー
	predicted_value: array
		予測値
	correct_value: array
		正解値
	
	Returns
	-------
	None
	"""
	predicted_value = original_target(predicted_value, target_scaler)
	correct_value = original_target(correct_value, target_scaler)
		
	result = eval_score(correct_value, predicted_value)
	return {
		"correct_value" : correct_value,
		"predicted_value" : predicted_value,
		"r2" : result["r2"],
		"mae" : result["mae"],
		"rmse" : result["rmse"],
		"evs" : result["evs"],
	}