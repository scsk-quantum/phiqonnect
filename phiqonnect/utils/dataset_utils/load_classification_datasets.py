from sklearn import datasets
import matplotlib.pyplot as plt
import math
import numpy as np
from phiqonnect.utils.calcuration.pca import pca_data
from sklearn.model_selection import train_test_split

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

def load_breast_cancer(feature_dim=None, dim_reduction="pca", train_size=20, test_size=10, test_ration=0.25, random_state=100, plot_data=False, shuffle=False, class_label=None, class_num=2, figsize=(8, 8)):
	"""breast_cancerデータセットを読み込む
        
	Parameters
	----------
	feature_dim: int
		特徴料の次元
	dim_reduction: string
		次元削減手法
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
	class_label: array
		クラスラベルの更新をする場合に使用
	class_num: int
		クラス数の指定
	figsize: (int, int)
		プロットを表示する場合にサイズを指定
	
	Returns
	-------
	training_input: array
		学習データ
	test_input: array
		テストデータ
	label_train: array
		学習データの正解ラベル
	label_test: array
		テストデータの正解ラベル
	class_label: array
		更新されたラベル
	data: array
		オリジナルデータ
	target: array
		オリジナルの正解ラベル
	"""
	data, target = datasets.load_breast_cancer(return_X_y=True)
	target = target.astype('int64')
	
	labels = np.array(['malignant', 'benign'])
	training_input, test_input, label_train, label_test, class_label, sample_train, sample_test, pca_train_data, pca_test_data, target_train, target_test = make_input_data(feature_dim, data, target, train_size, test_size, test_ration,random_state, class_num, class_label, dim_reduction)
	
	if shuffle:
		train_permutaion = np.random.permutation(len(label_train))
		training_input = training_input[train_permutaion]
		label_train = label_train[train_permutaion]
		
		test_permutaion = np.random.permutation(len(label_test))
		test_input = test_input[test_permutaion]
		label_test = label_test[test_permutaion]
		
	if plot_data == True:
		fig = plt.figure(figsize=figsize)
		
		make_plot(training_input, test_input, label_train, label_test, class_label, fig, labels)
		
		plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1,), borderaxespad=0, ncol=2)
		fig.tight_layout()
		plt.title(f"breast cancer (Mapping by PCA dim: {feature_dim} -> 2)")
		plt.show()
		
	return training_input, test_input, label_train, label_test, labels[class_label], data, target

def load_wine(feature_dim=None, dim_reduction="pca", train_size=20, test_size=10, test_ration=0.25, random_state=100, plot_data=False, shuffle=False, class_label=None, class_num=2, figsize=(8, 8)):
	"""wineデータセットを読み込む
        
	Parameters
	----------
	feature_dim: int
		特徴料の次元
	dim_reduction: string
		次元削減手法(pca|original))
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
	class_label: array
		クラスラベルの更新をする場合に使用
	class_num: int
		クラス数の指定
	figsize: (int, int)
		プロットを表示する場合にサイズを指定
	
	Returns
	-------
	training_input: array
		学習データ
	test_input: array
		テストデータ
	label_train: array
		学習データの正解ラベル
	label_test: array
		テストデータの正解ラベル
	class_label: array
		更新されたラベル
	data: array
		オリジナルデータ
	target: array
		オリジナルの正解ラベル
	"""
	data, target = datasets.load_wine(return_X_y=True)
	target = target.astype('int64')
	
	labels = np.array(['Wine_A', 'Wine_B', "Wine_C"])
	training_input, test_input, label_train, label_test, class_label, sample_train, sample_test, pca_train_data, pca_test_data, target_train, target_test = make_input_data(feature_dim, data, target, train_size, test_size, test_ration,random_state, class_num, class_label, dim_reduction)
	
	if shuffle:
		train_permutaion = np.random.permutation(len(label_train))
		training_input = training_input[train_permutaion]
		label_train = label_train[train_permutaion]
		
		test_permutaion = np.random.permutation(len(label_test))
		test_input = test_input[test_permutaion]
		label_test = label_test[test_permutaion]
		
	if plot_data == True:
		fig = plt.figure(figsize=figsize)
		
		make_plot(training_input, test_input, label_train, label_test, class_label, fig, labels)
		
		plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1,), borderaxespad=0, ncol=2)
		fig.tight_layout()
		plt.title(f"wine (Mapping by PCA dim: {feature_dim} -> 2)")
		plt.show()
		
	return training_input, test_input, label_train, label_test, labels[class_label], data, target

def load_digits(feature_dim=None, dim_reduction="pca", train_size=20, test_size=10, test_ration=0.25, random_state=100, plot_data=False, shuffle=False, class_label=None, class_num=2, figsize=(8, 8)):
	"""digitsデータセットを読み込む
        
	Parameters
	----------
	feature_dim: int
		特徴料の次元
	dim_reduction: string
		次元削減手法(pca|original))
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
	class_label: array
		クラスラベルの更新をする場合に使用
	class_num: int
		クラス数の指定
	figsize: (int, int)
		プロットを表示する場合にサイズを指定
	
	Returns
	-------
	training_input: array
		学習データ
	test_input: array
		テストデータ
	label_train: array
		学習データの正解ラベル
	label_test: array
		テストデータの正解ラベル
	class_label: array
		更新されたラベル
	data: array
		オリジナルデータ
	target: array
		オリジナルの正解ラベル
	"""
	data, target = datasets.load_digits(return_X_y=True)
	target = target.astype('int64')
	
	labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
	training_input, test_input, label_train, label_test, class_label, sample_train, sample_test, pca_train_data, pca_test_data, target_train, target_test = make_input_data(feature_dim, data, target, train_size, test_size, test_ration,random_state, class_num, class_label, dim_reduction)
	
	if shuffle:
		train_permutaion = np.random.permutation(len(label_train))
		training_input = training_input[train_permutaion]
		label_train = label_train[train_permutaion]
		
		test_permutaion = np.random.permutation(len(label_test))
		test_input = test_input[test_permutaion]
		label_test = label_test[test_permutaion]
		
	if plot_data == True:
		fig = plt.figure(figsize=figsize)
		
		make_plot(training_input, test_input, label_train, label_test, class_label, fig, labels)
		
		plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1,), borderaxespad=0, ncol=2)
		fig.tight_layout()
		plt.title(f"digits (Mapping by PCA dim: {feature_dim} -> 2)")
		plt.show()
		
	return training_input, test_input, label_train, label_test, labels[class_label], data, target
	
def load_iris(feature_dim=None, dim_reduction="pca", train_size=20, test_size=10, test_ration=0.25, random_state=100, plot_data=False, shuffle=False, class_label=None, class_num=2, figsize=(8, 8)):
	"""irisデータセットを読み込む
        
	Parameters
	----------
	feature_dim: int
		特徴料の次元
	dim_reduction: string
		次元削減手法(pca|original))
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
	class_label: array
		クラスラベルの更新をする場合に使用
	class_num: int
		クラス数の指定
	figsize: (int, int)
		プロットを表示する場合にサイズを指定
	
	Returns
	-------
	training_input: array
		学習データ
	test_input: array
		テストデータ
	label_train: array
		学習データの正解ラベル
	label_test: array
		テストデータの正解ラベル
	class_label:
		更新されたラベル
	data: array
		オリジナルデータ
	target: array
		オリジナルの正解ラベル
	"""
	data, target = datasets.load_iris(return_X_y=True)
	target = target.astype('int64')
	
	labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
	training_input, test_input, label_train, label_test, class_label, sample_train, sample_test, pca_train_data, pca_test_data, target_train, target_test = make_input_data(feature_dim, data, target, train_size, test_size, test_ration,random_state, class_num, class_label, dim_reduction)
	
	if shuffle:
		train_permutaion = np.random.permutation(len(label_train))
		training_input = training_input[train_permutaion]
		label_train = label_train[train_permutaion]
		
		test_permutaion = np.random.permutation(len(label_test))
		test_input = test_input[test_permutaion]
		label_test = label_test[test_permutaion]
		
	if plot_data == True:
		fig = plt.figure(figsize=figsize)
		make_plot(training_input, test_input, label_train, label_test, class_label, fig, labels)
		
		plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1,), borderaxespad=0, ncol=2)
		fig.tight_layout()
		plt.title(f"iris (Mapping by PCA dim: {feature_dim} -> 2)")
		plt.show()
		
	return training_input, test_input, label_train, label_test, labels[class_label], data, target

def load_credit_card(feature_dim=None, dim_reduction="original", train_size=20, test_size=10, test_ration=0.25, random_state=100, plot_data=False, shuffle=False, class_label=[0, 1], class_num=2, figsize=(8, 8)):
	"""credit_cardデータセットを読み込む
        
	Parameters
	----------
	feature_dim: int
		特徴料の次元
	dim_reduction: string
		次元削減手法(pca|original))
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
	class_label: array
		クラスラベルの更新をする場合に使用
	class_num: int
		クラス数の指定
	figsize: (int, int)
		プロットを表示する場合にサイズを指定
	
	Returns
	-------
	training_input: array
		学習データ
	test_input: array
		テストデータ
	label_train: array
		学習データの正解ラベル
	label_test: array
		テストデータの正解ラベル
	class_label: array
		更新されたラベル
	data: array
		オリジナルデータ
	target: array
		オリジナルの正解ラベル
	"""
	data = np.loadtxt('creditcard.csv', delimiter=',', skiprows=1, usecols=list(range(1, 29)))
	target_ = np.loadtxt('creditcard.csv', delimiter=',', skiprows=1, usecols=[30], dtype="str").astype('int64')
	
	def fun(e):
		return int(e[1])
	vfunc = np.vectorize(fun)
	target = vfunc(target_)
	
	labels = np.array(['normal', 'fraud'])
	training_input, test_input, label_train, label_test, class_label, sample_train, sample_test, pca_train_data, pca_test_data, target_train, target_test = make_input_data(feature_dim, data, target, train_size, test_size, test_ration,random_state, class_num, class_label, dim_reduction)
	
	if shuffle:
		train_permutaion = np.random.permutation(len(label_train))
		training_input = training_input[train_permutaion]
		label_train = label_train[train_permutaion]
		
		test_permutaion = np.random.permutation(len(label_test))
		test_input = test_input[test_permutaion]
		label_test = label_test[test_permutaion]
		
	if plot_data == True:
		fig = plt.figure(figsize=figsize)
		
		make_plot(training_input, test_input, label_train, label_test, class_label, fig, labels)
		
		plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1,), borderaxespad=0, ncol=2)
		fig.tight_layout()
		plt.title(f"Creditcard Fraud Data (Mapping by PCA dim: {feature_dim} -> 2)")
		plt.show()
		
	return training_input, test_input, label_train, label_test, labels[class_label], data, target

def load_fashion_mnist(feature_dim=None, dim_reduction="pca", train_size=20, test_size=10, test_ration=0.25, random_state=100, plot_data=False, shuffle=False, class_label=None, class_num=2, figsize=(8, 8)):
	"""fashion_mnistデータセットを読み込む
        
	Parameters
	----------
	feature_dim: int
		特徴料の次元
	dim_reduction: string
		次元削減手法(pca|original))
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
	class_label: array
		クラスラベルの更新をする場合に使用
	class_num: int
		クラス数の指定
	figsize: (int, int)
		プロットを表示する場合にサイズを指定
	
	Returns
	-------
	training_input: array
		学習データ
	test_input: array
		テストデータ
	label_train: array
		学習データの正解ラベル
	label_test: array
		テストデータの正解ラベル
	class_label: array
		更新されたラベル
	data: array
		オリジナルデータ
	target: array
		オリジナルの正解ラベル
	"""
	if feature_dim is None:
		feature_dim = 784
	
	from joblib import Memory
	memory = Memory('data/tmp')
	from sklearn import datasets
	fetch_openml_cached = memory.cache(datasets.fetch_openml)
	data, target = fetch_openml_cached('fashion-mnist', data_home="data/src/download/", return_X_y=True)
	data = data.values
	target = target.values.astype('int64')
	
	labels = np.array(["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"])
	training_input, test_input, label_train, label_test, class_label, sample_train, sample_test, pca_train_data, pca_test_data, target_train, target_test = make_input_data(feature_dim, data, target, train_size, test_size, test_ration,random_state, class_num, class_label, dim_reduction)
	
	if shuffle:
		train_permutaion = np.random.permutation(len(label_train))
		training_input = training_input[train_permutaion]
		label_train = label_train[train_permutaion]
		
		test_permutaion = np.random.permutation(len(label_test))
		test_input = test_input[test_permutaion]
		label_test = label_test[test_permutaion]
		
	if plot_data == True:
		fig = plt.figure(figsize=figsize)
		
		make_plot(training_input, test_input, label_train, label_test, class_label, fig, labels)
		
		plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1,), borderaxespad=0, ncol=2)
		fig.tight_layout()
		plt.title(f"Fashion-MNIST (Mapping by PCA dim: {feature_dim} -> 2)")
		plt.show()
		
	return training_input, test_input, label_train, label_test, labels[class_label], data, target

def load_mnist(feature_dim=None, dim_reduction="pca", train_size=20, test_size=10, test_ration=0.25, random_state=100, plot_data=False, shuffle=False, class_label=None, class_num=2, figsize=(8, 8)):
	"""fashion_mnistデータセットを読み込む
        
	Parameters
	----------
	feature_dim: int
		特徴料の次元
	dim_reduction: string
		予測データ
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
	class_label: array
		クラスラベルの更新をする場合に使用
	class_num: int
		クラス数の指定
	figsize: (int, int)
		プロットを表示する場合にサイズを指定
	
	Returns
	-------
	training_input: array
		学習データ
	test_input: array
		テストデータ
	label_train: array
		学習データの正解ラベル
	label_test: array
		テストデータの正解ラベル
	class_label: array
		更新されたラベル
	data: array
		オリジナルデータ
	target: array
		オリジナルの正解ラベル
	"""
	if feature_dim is None:
		feature_dim = 784
	
	from joblib import Memory
	memory = Memory('data/tmp')
	from sklearn import datasets
	fetch_openml_cached = memory.cache(datasets.fetch_openml)
	data, target = fetch_openml_cached('mnist_784', data_home="data/src/download/", return_X_y=True)
	data = data.values
	target = target.values.astype('int64')
	
	training_input, test_input, label_train, label_test, class_label, sample_train, sample_test, pca_train_data, pca_test_data, target_train, target_test = make_input_data(feature_dim, data, target, train_size, test_size, test_ration, random_state, class_num, class_label, dim_reduction)
	
	if shuffle:
		train_permutaion = np.random.permutation(len(label_train))
		training_input = training_input[train_permutaion]
		label_train = label_train[train_permutaion]
		
		test_permutaion = np.random.permutation(len(label_test))
		test_input = test_input[test_permutaion]
		label_test = label_test[test_permutaion]
	
	if plot_data == True:
		fig = plt.figure(figsize=figsize)
		
		
		make_plot(training_input, test_input, label_train, label_test, class_label, fig)
		
		plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1,), borderaxespad=0, ncol=2)
		fig.tight_layout()
		plt.title(f"MNIST784 (Mapping by PCA dim: {feature_dim} -> 2)")
		plt.show()

	return training_input, test_input, label_train, label_test, class_label, data, target

def make_input_data(feature_dim, data, target, train_size, test_size, test_ration, random_state, class_num, class_label, dim_reduction):
	"""データの加工をする
        
	Parameters
	----------
	feature_dim: int
		特徴量の次元
	data: array
		データ
	target: string
		データの正解ラベル
	train_size: int
		学習データのサイズ
	test_size: int
		テストデータのサイズ
	test_ration: int
		母集団からの学習＆テストデータの振り分けの比率
	class_label: array
		クラスラベルの更新をする場合に使用
	class_num: int
		クラス数の指定
	
	Returns
	-------
	training_input: array
		層化抽出した学習データ
	test_input: array
		層化抽出したテストデータ
	label_train: array
		学習データの正解ラベル
	label_test: array
		テストデータの正解ラベル
	class_label: array
		更新されたラベル
	sample_train: array
		オリジナルの学習データ
	sample_test: array
		オリジナルのテストデータ
	sample_train: array
		次元削減後の学習データ
	sample_test: array
		次元削減後のテストデータ
	target_train: array
		オリジナルの学習データの正解ラベル
	target_test: array
		オリジナルのテストデータの正解ラベル
	"""
	target_uniq = np.unique(target)

	if class_label is None:
		class_label = []
		for i in range(class_num):
			class_label.append(target_uniq[i])
		class_label = np.array(class_label)
	else :
		class_num = len(class_label)
		
	sample_train, sample_test, target_train, target_test = train_test_split(data, target, test_size=test_ration, random_state=random_state)
	
	if dim_reduction == "pca":
		pca_train_data, pca_test_data, pca = pca_data(sample_train, sample_test, feature_dim)
	elif dim_reduction == "original":
		pca_train_data, pca_test_data, pca = sample_train[:, :feature_dim], sample_test[:, :feature_dim], None
	else:
		pca_train_data, pca_test_data, pca = sample_train, sample_test, None
	
	_training_input = []
	_test_input = []
	_label_train = []
	_label_test = []
	train_block = [(train_size + i) // class_num for i in range(class_num)]
	test_block = [(test_size + i) // class_num for i in range(class_num)]
	for i in range(class_num):
		_training_input.append(pca_train_data[target_train == class_label[i]][:train_block[i]])
		_test_input.append(pca_test_data[target_test == class_label[i]][:test_block[i]])
		_label_train.append(np.ones(len(_training_input[i])) * class_label[i])
		_label_test.append(np.ones(len(_test_input[i])) * class_label[i])
	
	training_input = np.concatenate(_training_input)
	test_input = np.concatenate(_test_input)
	label_train = np.concatenate(_label_train)
	label_test = np.concatenate(_label_test)
	
	if len(training_input) != train_size or len(test_input) != test_size:
		raise ValueError(f"Try lowering training_size or test_size, or changing the data seed.")
	
	return	training_input, test_input, label_train, label_test, class_label, sample_train, sample_test, pca_train_data, pca_test_data, target_train, target_test
	
def make_plot(train_data, test_data, label_train, label_test, class_label, fig, labels=None):
	"""データをプロットする
        
	Parameters
	----------
	training_input: array
		学習データ
	test_input: array
		テストデータ
	label_train: array
		学習データの正解ラベル
	label_test: array
		テストデータの正解ラベル
	fig: matplotlib.pyplot.figure.Figure
		書き込み先のFigure
	class_label: array
		ラベル
	labels: array
		ラベルの更新に使用する
	
	Returns
	-------
	fig: matplotlib.pyplot.figure.Figure
		データをプロットFigure
	"""
	import matplotlib
	color_list = []
#	for idx, (color, rgb) in enumerate(matplotlib.colors.BASE_COLORS.items()):
#		color_list.append(color)
	for idx, (color, rgb) in enumerate(matplotlib.colors.TABLEAU_COLORS.items()):
		color_list.append(color)
	
	if labels is None:
		labels = class_label
	
	X, y, pca = pca_data(train_data, test_data, 2)
		
	X_axis = 0
	Y_axis = 1
	if np.var(X[:, 0]) <= np.var(X[:, 1]):
		X_axis = 1
		Y_axis = 0
		
	for i in range(len(class_label)):
		plt.scatter(X[label_train == class_label[i], X_axis], X[label_train == class_label[i], Y_axis], color=color_list[i%len(color_list)], marker='.', label=str(labels[class_label[i]])+" 学習データ")
	for i in range(len(class_label)):
		plt.scatter(y[label_test == class_label[i], X_axis], y[label_test == class_label[i], Y_axis], color=color_list[i%len(color_list)], marker='x', label=str(labels[class_label[i]])+" テストデータ")
		
	return fig