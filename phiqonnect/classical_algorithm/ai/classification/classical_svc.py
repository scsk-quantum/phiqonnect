#classical_svc.py

import numpy as np
from sklearn.svm import SVC

from ....utils.calcuration.eval_classification_result import eval_score

class ClassicalSVC():
	"""
	ClassicalSVC(現代のサポートベクター分類)クラス

	Parameters
	----------
	kernel_func: callback function
		カーネル値を返すコールバック関数
	verbose: boolean
		詳細情報を出力する
	probability: boolean
		分類の確率を返すか
	"""

	def __init__(self, kernel_func, verbose=False, probability=False) -> None:
		self.kernel_func = kernel_func
		
		self.verbose = verbose
		self.probability = probability
		
		self.train_kernel_matrix = None
		self.test_kernel_matrix = None
		self.predict_kernel_matrix = None
		
		self.train_data = None
		self.train_labels = None
		
		self.svc = SVC(kernel="precomputed", verbose=self.verbose, probability=self.probability)
	
	def train(self, data, labels, result=False, class_label=None):
		"""学習
		
		Parameters
		----------
		data: array
			学習データ
		labels: array
			正解ラベル
		result: boolean
			結果を取得するかどうか
		class_label: array
			labelsの割り当ての更新

		Returns
		-------
		result: dict
			resultがTrue時に結果のdictを返す
		"""

		self.train_data = data
		self.train_labels = labels
	
		kernel_matrix = self.get_kernel_matrix(data)
		self.train_kernel_matrix = kernel_matrix
		
		if result:
			self._train(kernel_matrix, labels, result=True, class_label=class_label)
			self.support_vectors = self.train_data[self.svc.support_]
			self.yin = self.train_labels[self.svc.support_]
		else:
			self._train(kernel_matrix, labels, class_label=class_label)
			self.support_vectors = self.train_data[self.svc.support_]
			self.yin = self.train_labels[self.svc.support_]
		return self.train_result
	
	def _train(self, kernel_matrix, labels, result=False, class_label=None):
		"""学習
		
		Parameters
		----------
		kernel_matrix: array
			カーネル行列（学習データ）
		labels: array
			正解ラベル
		result: boolean
			結果を取得するかどうか
		class_label: array
			labelsの割り当ての更新

		Returns
		-------
		result: dict
			resultがTrue時に結果のdictを返す
		"""

		self.svc.fit(kernel_matrix, labels)
		
		self.alphas = self.svc.dual_coef_
		self.bias = self.svc.intercept_
		self.support = self.svc.support_
		
		if result:
			self.train_result = self._test(kernel_matrix, labels, class_label=class_label)
			return self.train_result
		else:
			return None
		
	def test(self, data, labels, use_support_vector_index=True, class_label=None):
		"""検証
		
		Parameters
		----------
		data: array
			検証データ
		labels: array
			正解ラベル
		use_support_vector_index: boolean
			サポートベクター以外の演算をするかどうか
		class_label: array
			labelsの割り当ての更新

		Returns
		-------
		result: dict
			結果のdictを返す
		"""

		if use_support_vector_index and self.support_vectors is not None:
			kernel_matrix = np.zeros((len(data), len(self.train_data)))
			kernel_matrix[:, self.support] = self.get_kernel_matrix(data, self.support_vectors)
		else:
			kernel_matrix = self.get_kernel_matrix(data, self.train_data)
			
		return self._test(kernel_matrix, labels, class_label=class_label)
		
	def _test(self, kernel_matrix, labels, class_label=None):
		"""検証
		
		Parameters
		----------
		kernel_matrix: array
			カーネル行列（検証データ）
		labels: array
			正解ラベル
		class_label: array
			labelsの割り当ての更新

		Returns
		-------
		result: dict
			結果のdictを返す
		"""

		predicted_labels = self._predict(kernel_matrix)
		result = eval_score(labels, predicted_labels)
		
		if class_label is not None:
			labels = class_label[labels.astype('int64')]
			predicted_labels = class_label[predicted_labels.astype('int64')]
		
		# y = predicted_labels * 2 - 1
		# confidence = np.sum(self.yin * self.alphas * kernel_matrix[:, self.support] - self.bias, axis=1)
		# fx = np.abs(confidence)
		test_result = {
			"correct_labels" : labels,
			"predicted_labels" : predicted_labels,
			# "hinge_loss" : np.sum(np.maximum(0, 1 - y * fx)),
			# "squared_hinge_loss" : np.sum(np.maximum(0, (1 - y * fx) ** 2)),
			"accuracy" : result["accuracy"],
			"precision" : result["precision"],
			"recall" : result["recall"],
			"specificity" : result["specificity"],
			"f_measure" : result["f-measure"],
			"confusion_matrix" : result["confusion_matrix"],
			"kernel_matrix": kernel_matrix
		}
		self.test_result = test_result
		
		return test_result
	
	def predict(self, data, use_support_vector_index=True, class_label=None):
		"""予測
		
		Parameters
		----------
		data: array
			予測するデータ
		use_support_vector_index: boolean
			サポートベクター以外の演算をするかどうか
		class_label: array
			labelsの割り当ての更新

		Returns
		-------
		result: array
			結果のarrayを返す
		"""

		if use_support_vector_index and self.support_vectors is not None:
			kernel_matrix = np.zeros((len(data), len(self.train_data)))
			kernel_matrix[:, self.support] = self.get_kernel_matrix(data, self.support_vectors)
		else:
			kernel_matrix = self.get_kernel_matrix(data, self.train_data)
			
		if class_label is None:
			return self._predict(kernel_matrix)
		else:
			return class_label[self._predict(kernel_matrix).astype('int64')]
		
	def _predict(self, kernel_matrix, class_label=None):
		"""予測
		
		Parameters
		----------
		kernel_matrix: array
			カーネル行列（予測するデータ）
		class_label: array
			labelsの割り当ての更新

		Returns
		-------
		result: array
			結果のarrayを返す
		"""

		if self.probability:
			return self.svc.predict_proba(kernel_matrix)
		else:
			if class_label is None:
				return self.svc.predict(kernel_matrix)
			else:
				return class_label[self.svc.predict(kernel_matrix).astype('int64')]
	
	def get_kernel_matrix(self, xi, xj=None):
		"""xiとxjのカーネル値を取得する
		
		Parameters
		----------
		xi: array
			データ1
		xj: array
			データ2

		Returns
		-------
		result: float
			xiとxjのカーネル値
		"""

		return self.kernel_func(xi, xj)
	
	def get_support_vector_index(self):
		"""サポートベクターの（学習データの）インデックスを返す

		Returns
		-------
		result: array
			サポートベクターの（学習データの）インデックス
		"""
		return self.support
	
	def get_support_vector(self):
		"""サポートベクターを返す

		Returns
		-------
		result: array
			サポートベクターを返す
		"""
		return self.support_vectors
	
	def get_train_result(self):
		"""学習結果を返す

		Returns
		-------
		result: array
			学習結果を返す
		"""
		return self.train_result
	
	def save_model(self, filepath="./sample_model"):
		"""モデルをシリアライズして保存する
		
		Parameters
		----------
		filepath: array
			保存ファイルパス
		"""
		import dill
		
		f = open(filepath, "wb")
		dill.dump(self, f)
		f.close()
	
	@staticmethod
	def load_model(filepath="./sample_model"):
		"""シリアライズしたモデルを読み込む
		
		Parameters
		----------
		filepath: array
			ファイルパス
		"""
		import dill
		
		f = open(filepath, "rb")
		return dill.loads(f)