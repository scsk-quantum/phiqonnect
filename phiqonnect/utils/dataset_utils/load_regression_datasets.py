from sklearn import datasets
import matplotlib.pyplot as plt
import math
import numpy as np
from phiqonnect.utils.calcuration.pca import pca_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

def load_boston(feature_dim=None, dim_reduction="pca", target_scale="minmax", train_size=20, test_size=10, test_ration=0.25, random_state=100, plot_data=False, shuffle=False):
	"""bostonデータセットを読み込む
        
	Parameters
	----------
	feature_dim: int
		特徴料の次元
	dim_reduction: string
		次元削減手法
	target_scale: string
		ターゲットの変換手法
	train_size: int
		学習データのサイズ
	test_size: int
		テストデータのサイズ
	test_ration: int
		母集団からの学習＆テストデータの振り分けの比率
	random_state: int
		データのランダムシード
	plot_data: boolean
		プロットを表示するか
	shuffle: boolean
		データ順序をシャッフルするかどうか
	
	Returns
	-------
	training_data: array
		学習データの説明変数
	test_data: array
		テストデータの説明変数
	train_target: array
		学習データの目的変数
	test_target: array
		テストデータの目的変数
	data: array
		オリジナルデータの説明変数
	target: array
		オリジナルのデータの目的変数
	target_scaler: sklearn.preprocessing.*
		オリジナルのデータの目的変数のスケーラー
	"""
	data, target = datasets.load_boston(return_X_y=True)
	target = target.reshape(-1, 1)
	
	train_data, test_data, train_target, test_target, sample_train, sample_test, pca_train_data, pca_test_data, sample_target_train, sample_target_test, target_scaler = make_input_data(feature_dim, data, target, train_size, test_size, test_ration, random_state, dim_reduction, target_scale)
	
	if shuffle:
		train_permutaion = np.random.permutation(len(train_target))
		train_data = train_data[train_permutaion]
		train_target = train_target[train_permutaion]
		
		test_permutaion = np.random.permutation(len(test_target))
		test_data = test_data[test_permutaion]
		test_target = test_target[test_permutaion]
		
	if plot_data == True:
		fig = plt.figure(figsize=(8, 8))
		
		make_plot(train_data, test_data, train_target, test_target, fig)
		
		plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1,), borderaxespad=0, ncol=2)
		fig.tight_layout()
		plt.title(f"boston (Mapping by PCA dim: {feature_dim})")
		plt.show()
		
	return train_data, test_data, train_target, test_target, data, target, target_scaler

def load_diabetes(feature_dim=None, dim_reduction="pca", target_scale="minmax", train_size=20, test_size=10, test_ration=0.25, random_state=100, plot_data=False, shuffle=False):
	"""diabetesデータセットを読み込む
        
	Parameters
	----------
	feature_dim: int
		特徴料の次元
	dim_reduction: string
		次元削減手法
	target_scale: string
		ターゲットの変換手法
	train_size: int
		学習データのサイズ
	test_size: int
		テストデータのサイズ
	test_ration: int
		母集団からの学習＆テストデータの振り分けの比率
	random_state: int
		データのランダムシード
	plot_data: boolean
		プロットを表示するか
	shuffle: boolean
		データ順序をシャッフルするかどうか
	
	Returns
	-------
	training_data: array
		学習データの説明変数
	test_data: array
		テストデータの説明変数
	train_target: array
		学習データの目的変数
	test_target: array
		テストデータの目的変数
	data: array
		オリジナルデータの説明変数
	target: array
		オリジナルのデータの目的変数
	target_scaler: sklearn.preprocessing.*
		オリジナルのデータの目的変数のスケーラー
	"""
	data, target = datasets.load_diabetes(return_X_y=True)
	target = target.reshape(-1, 1)
	
	train_data, test_data, train_target, test_target, sample_train, sample_test, pca_train_data, pca_test_data, sample_target_train, sample_target_test, target_scaler = make_input_data(feature_dim, data, target, train_size, test_size, test_ration, random_state, dim_reduction, target_scale)
	
	if shuffle:
		train_permutaion = np.random.permutation(len(train_target))
		train_data = train_data[train_permutaion]
		train_target = train_target[train_permutaion]
		
		test_permutaion = np.random.permutation(len(test_target))
		test_data = test_data[test_permutaion]
		test_target = test_target[test_permutaion]
		
	if plot_data == True:
		fig = plt.figure(figsize=(8, 8))
		
		make_plot(train_data, test_data, train_target, test_target, fig)
		
		plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1,), borderaxespad=0, ncol=2)
		fig.tight_layout()
		plt.title(f"diabetes (Mapping by PCA dim: {feature_dim})")
		plt.show()
		
	return train_data, test_data, train_target, test_target, data, target, target_scaler
	
def load_linnerud(feature_dim=None, dim_reduction="pca", target_scale="minmax", train_size=20, test_size=10, test_ration=0.25, random_state=100, plot_data=False, shuffle=False):
	"""linnerudータセットを読み込む
        
	Parameters
	----------
	feature_dim: int
		特徴料の次元
	dim_reduction: string
		次元削減手法
	target_scale: string
		ターゲットの変換手法
	train_size: int
		学習データのサイズ
	test_size: int
		テストデータのサイズ
	test_ration: int
		母集団からの学習＆テストデータの振り分けの比率
	random_state: int
		データのランダムシード
	plot_data: boolean
		プロットを表示するか
	shuffle: boolean
		データ順序をシャッフルするかどうか
	
	Returns
	-------
	training_data: array
		学習データの説明変数
	test_data: array
		テストデータの説明変数
	train_target: array
		学習データの目的変数
	test_target: array
		テストデータの目的変数
	data: array
		オリジナルデータの説明変数
	target: array
		オリジナルのデータの目的変数
	target_scaler: sklearn.preprocessing.*
		オリジナルのデータの目的変数のスケーラー
	"""
	data, target = datasets.load_linnerud(return_X_y=True)
	target = target.reshape(-1, 1)
	
	train_data, test_data, train_target, test_target, sample_train, sample_test, pca_train_data, pca_test_data, sample_target_train, sample_target_test, target_scaler = make_input_data(feature_dim, data, target, train_size, test_size, test_ration, random_state, dim_reduction, target_scale)
	
	if shuffle:
		train_permutaion = np.random.permutation(len(train_target))
		train_data = train_data[train_permutaion]
		train_target = train_target[train_permutaion]
		
		test_permutaion = np.random.permutation(len(test_target))
		test_data = test_data[test_permutaion]
		test_target = test_target[test_permutaion]
		
	if plot_data == True:
		fig = plt.figure(figsize=(8, 8))
		
		make_plot(train_data, test_data, train_target, test_target, fig)
		
		plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1,), borderaxespad=0, ncol=2)
		fig.tight_layout()
		plt.title(f"linnerud (Mapping by PCA dim: {feature_dim})")
		plt.show()
		
	return train_data, test_data, train_target, test_target, data, target, target_scaler

def make_input_data(feature_dim, data, target, train_size, test_size, test_ration, random_state, dim_reduction, target_scale):
	"""データを加工する
        
	Parameters
	----------
	feature_dim: int
		特徴料の次元
	dim_reduction: string
		次元削減手法
	target_scale: string
		ターゲットの変換手法
	train_size: int
		学習データのサイズ
	test_size: int
		テストデータのサイズ
	test_ration: int
		母集団からの学習＆テストデータの振り分けの比率
	random_state: int
		データのランダムシード
	data: array
		オリジナルデータの説明変数
	target: array
		オリジナルのデータの目的変数
	
	Returns
	-------
	training_input: array
		層化抽出した学習データ
	test_input: array
		層化抽出したテストデータ
	train_target: array
		学習データの目的変数
	test_target: array
		テストデータの目的変数
	sample_train: array
		オリジナルの学習データ
	sample_test: array
		オリジナルのテストデータ
	sample_train: array
		次元削減後の学習データ
	sample_test: array
		次元削減後のテストデータ
	target_train: array
		オリジナルの学習データの目的変数
	target_test: array
		オリジナルのテストデータの目的変数
	target_scaler: sklearn.preprocessing.*
		オリジナルのデータの目的変数のスケーラー
	"""
	sample_train, sample_test, sample_target_train, sample_target_test = train_test_split(data, target, test_size=test_ration, random_state=random_state)
	
	if dim_reduction == "pca":
		pca_train_data, pca_test_data, pca = pca_data(sample_train, sample_test, feature_dim)
	elif dim_reduction == "original":
		pca_train_data, pca_test_data, pca = sample_train[:, :feature_dim], sample_test[:, :feature_dim], None
	else:
		pca_train_data, pca_test_data, pca = sample_train, sample_test, None
	
	if target_scale == "minmax":
		target_scaler = MinMaxScaler((0, 1)).fit(sample_target_train)
		sample_target_train = target_scaler.transform(sample_target_train)
		sample_target_test = target_scaler.transform(sample_target_test)
	
	train_data = pca_train_data[:train_size]
	test_data = pca_test_data[:test_size]
	train_target = sample_target_train[:train_size]
	test_target = sample_target_test[:test_size]
	
	if len(train_data) != train_size or len(test_data) != test_size:
		raise ValueError(f"Try lowering training_size or test_size, or changing the data seed.")
	
	return	train_data, test_data, train_target, test_target, sample_train, sample_test, pca_train_data, pca_test_data, sample_target_train, sample_target_test, target_scaler
	
def make_plot(train_data, test_data, train_target, test_target, fig):
	"""データをプロットする
        
	Parameters
	----------
	train_data: array
		学習データ
	test_data: array
		テストデータ
	train_target: array
		学習データの目的変数
	test_target: array
		テストデータの目的変数
	fig: matplotlib.pyplot.figure.Figure
		書き込み先のFigure
	
	Returns
	-------
	fig: matplotlib.pyplot.figure.Figure
		データをプロットFigure
	"""
	import matplotlib
	color_list = [str(i/10) for i in range(10)]
	# for idx, (color, rgb) in enumerate(matplotlib.colors.BASE_COLORS.items()):
#	color_list.append(color)
	# for idx, (color, rgb) in enumerate(matplotlib.colors.TABLEAU_COLORS.items()):
	# 	color_list.append(color)
		
	plt.plot(list(range(0, len(train_target))), train_target, marker='o', color="b", linestyle = "solid", label="学習データ target")
	for i in range(len(train_data[0])):
		plt.plot(list(range(0, len(train_data[:, i]))), train_data[:, i], marker='o', alpha = 0.5, color=color_list[i%len(color_list)], linestyle = "solid", label="学習データ data[" + str(i) + "]")
	
	plt.plot(list(range(0, len(test_target))), test_target, marker='x', color="r", linestyle = "dotted", label="テストデータ target")	
	for i in range(len(test_data[0])):
		plt.plot(list(range(0, len(test_data[:, i]))), test_data[:, i], marker='x', alpha = 0.5, color=color_list[i%len(color_list)], linestyle = "dotted", label="テストデータ data[" + str(i) + "]")
	
	return fig
	
def original_target(target, scaler):
	"""オリジナルの目的変数を取得する
        
	Parameters
	----------
	target: array
		スケールされた目的変数
	scaler: sklearn.preprocessing.*
		目的変数のスケーラー
	
	Returns
	-------
	target: array
		オリジナルの目的変数
	"""
	return scaler.inverse_transform(target.reshape(-1, 1))