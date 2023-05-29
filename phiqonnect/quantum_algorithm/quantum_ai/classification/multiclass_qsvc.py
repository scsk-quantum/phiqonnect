# multiclass_qsvc.py

import numpy as np
import itertools
from sklearn.svm import SVC
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterVector
from qiskit import Aer
from qiskit.utils import QuantumInstance

from .qsvc import QSVC
from phiqonnect.utils.calcuration.eval_classification_result import multi_class_eval_score
from phiqonnect.utils.device.braket_qunatum_instance import BraketQuantumInstance

class MulticlassQSVC():
	"""
	MulticlassQSVCQSVC(量子カーネルを用いたサポートベクター多クラス分類)クラス

	Parameters
	----------
	qkernel: QKernel
		QKernelを継承したクラスのオブジェクト
	class_num: int
		クラス数
	method: string
		ovr(One vs Rest) or ovro(One vs One)
	instance: QuantumInstance|String
		QuantumInstanceあるいは使用するバックエンド("quasm_simulator", "simulator_braket", "ionq_braket", "ionq_rigetti")
	shot: int
		観測数
	seed: int
		シード値
	batch_size: int
		バッチサイズ
	verbose: boolean
		詳細情報を出力する
	"""
	def __init__(self, qkernel, class_num, method="ovr", instance=None, shots=1000, seed=1234, batch_size=1000, verbose=False) -> None:
		self.qkernel = qkernel
		
		if instance == "qasm_simulator":
			shots = shots
			backend = Aer.get_backend('qasm_simulator')
			seed = seed
			self.instance = QuantumInstance(backend, shots=shots, seed_simulator=seed, seed_transpiler=seed)
		elif instance == "simulator_braket":
			self.instance = BraketQuantumInstance("simulator", shots)
		elif instance == "ionq_braket":
			self.instance = BraketQuantumInstance("ionq", shots)
		elif instance == "rigetti_braket":
			self.instance = BraketQuantumInstance("rigetti", shots)
		else:
			self.instance = instance
		
		self.class_num = class_num
		self.method = method
		
		if self.method == "ovr":
			self.classifiers = [QSVC(qkernel, instance=instance, seed=seed, batch_size=batch_size, verbose=verbose, probability=True) for _ in range(class_num)]
		elif self.method == "ovo":
			self.combinations = list(itertools.combinations(list(range(class_num)), 2))
			self.classifiers = [QSVC(qkernel, instance=instance, seed=seed, batch_size=batch_size, verbose=verbose, probability=False) for _ in range(len(self.combinations))]
		else:
			raise ValueError("method error.")
		self.train_kernel_matrix = []
		self.test_kernel_matrix = []
		self.predict_kernel_matrix = []
		self.train_data = None
		self.train_labels = None
		
		return
	
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
	
		if self.method == "ovr":
			for i in range(self.class_num):
				new_label = np.where(self.train_labels == i, 1, 0)
				self.classifiers[i]._train(self.train_kernel_matrix, new_label)
			if result:
				self.train_result = self._test(self.train_kernel_matrix, self.train_labels, class_label=class_label)
		elif self.method == "ovo":
			for i in range(len(self.classifiers)):
				cls1 = self.combinations[i][0]
				cls2 = self.combinations[i][1]
				new_kernel_matrix = kernel_matrix[:, (labels == cls1) | (labels == cls2)][(labels == cls1) | (labels == cls2)]
				new_label = labels[(labels == cls1) | (labels == cls2)]
				new_label = np.where(new_label == cls1, 1, 0)
				self.classifiers[i]._train(new_kernel_matrix, new_label)
			if result:
				self.train_result = self._test(self.train_kernel_matrix, self.train_labels, class_label=class_label)	
		if result:
			return self.train_result 
		else:
			return None
			
	def test(self, data, labels, class_label=None):
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
		use_support_vector_index: boolean
			サポートベクター以外の演算をするかどうか
		class_label: array
			labelsの割り当ての更新

		Returns
		-------
		result: dict
			結果のdictを返す
		"""

		predicted_labels = self._predict(kernel_matrix)
		result = multi_class_eval_score(labels, predicted_labels)
		
		if class_label is not None:
			labels = class_label[labels.astype('int64')]
			predicted_labels = class_label[predicted_labels.astype('int64')]
		
		test_result = {
			"correct_labels" : labels,
			"predicted_labels" : predicted_labels,
			"accuracy" : result["accuracy"],
			"precision" : result["precision"],
			"recall" : result["recall"],
			"f_measure" : result["f-measure"],
			"confusion_matrix" : result["confusion_matrix"],
			"kernel_matrix": kernel_matrix
		}
		return test_result
	
	def predict(self, data, class_label=None):
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
		use_support_vector_index: boolean
			サポートベクター以外の演算をするかどうか
		class_label: array
			labelsの割り当ての更新

		Returns
		-------
		result: array
			結果のarrayを返す
		"""

		if self.method == "ovr":
			scores = np.array([model._predict(kernel_matrix)[:, 1] for model in self.classifiers])
			if class_label is None:
				return np.argmax(scores, axis=0)
			else:
				return class_label[np.argmax(scores, axis=0).astype('int64')]
		elif self.method == "ovo":
			votes = np.zeros((len(kernel_matrix), self.class_num))
			for i in range(len(self.classifiers)):
				cls1 = self.combinations[i][0]
				cls2 = self.combinations[i][1]
				new_kernel_matrix = kernel_matrix[:, (self.train_labels == cls1) | (self.train_labels == cls2)]
				predicted_labels = self.classifiers[i]._predict(new_kernel_matrix)
				voted = np.where(predicted_labels == 1, self.combinations[i][0], self.combinations[i][1])
				for j in range(len(voted)):
					votes[j, voted[j]] += 1
			if class_label is None:
				return np.argmax(votes, axis=1)
			else:
				return class_label[np.argmax(votes, axis=1).astype('int64')]
	
	def get_train_result(self):
		"""学習結果を返す

		Returns
		-------
		result: array
			学習結果を返す
		"""

		return self.train_result
	
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

		return self.qkernel.get_kernel_matrix(xi, xj, self.instance)
	
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