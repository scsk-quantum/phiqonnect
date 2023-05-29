#qsvr.py

import numpy as np
from sklearn.svm import SVR
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterVector
from qiskit import Aer
from qiskit.utils import QuantumInstance

from phiqonnect.utils.calcuration.eval_regression_result import eval_score
from phiqonnect.utils.device.braket_qunatum_instance import BraketQuantumInstance

class QSVR():
	"""
	QSVR(量子カーネルを用いたサポートベクター回帰)クラス

	Parameters
	----------
	qkernel: QKernel
		QKernelを継承したクラスのオブジェクト
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
	def __init__(self, qkernel, instance=None, shots=1000, seed=1234, batch_size=1000, verbose=False) -> None:
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
			
		self.batch_size = batch_size
		self.verbose = verbose
		
		self.train_kernel_matrix = None
		self.test_kernel_matrix = None
		self.predict_kernel_matrix = None
		
		self.train_data = None
		self.train_target = None
		
		return
	
	def train(self, data, target, result=False):
		"""学習
		
		Parameters
		----------
		data: array
			学習データ
		target: array
			正解データ
		result: boolean
			結果を取得するかどうか

		Returns
		-------
		result: dict
			resultがTrue時に結果のdictを返す
		"""
		self.train_data = data
		self.train_target = target
	
		kernel_matrix = self.get_kernel_matrix(data)
		self.train_kernel_matrix = kernel_matrix
		
		if result:
			return self._train(kernel_matrix, target, True)
		else:
			self._train(kernel_matrix, target)
			return None
	
	def _train(self, kernel_matrix, target, result=False):
		"""学習
		
		Parameters
		----------
		kernel_matrix: array
			カーネル行列（学習データ）
		target: array
			正解データ
		result: boolean
			結果を取得するかどうか

		Returns
		-------
		result: dict
			resultがTrue時に結果のdictを返す
		"""
		self.svr = SVR(kernel="precomputed", verbose=self.verbose)
		self.svr.fit(kernel_matrix, target)
		
		self.alphas = self.svr.dual_coef_
		self.bias = self.svr.intercept_
		self.support = self.svr.support_
		self.support_vectors = self.train_data[self.svr.support_]
		self.yin = self.train_target[self.svr.support_]
		
		if result:
			self.train_result = self._test(kernel_matrix, target)
			return self.train_result
		else:
			return None
		
	def test(self, data, target, use_support_vector_index=True):
		"""検証
		
		Parameters
		----------
		data: array
			検証データ
		target: array
			正解データ
		use_support_vector_index: boolean
			サポートベクター以外の演算をするかどうか

		Returns
		-------
		result: dict
			結果のdictを返す
		"""
		if use_support_vector_index:
			kernel_matrix = np.zeros((len(data), len(self.train_data)))
			kernel_matrix[:, self.support] = self.get_kernel_matrix(data, self.support_vectors)
		else:
			kernel_matrix = self.get_kernel_matrix(data, self.train_data)
			
		return self._test(kernel_matrix, target)
		
	def _test(self, kernel_matrix, target):
		"""検証
		
		Parameters
		----------
		kernel_matrix: array
			カーネル行列（検証データ）
		target: array
			正解データ

		Returns
		-------
		result: dict
			結果のdictを返す
		"""
		predicted_values = self._predict(kernel_matrix)
		result = eval_score(target, predicted_values)
		
		# y = predicted_target * 2 - 1
		# confidence = np.sum(self.yin * self.alphas * kernel_matrix[:, self.support] - self.bias, axis=1)
		# fx = np.abs(confidence)
		test_result = {
			"correct_value" : target,
			"predicted_value" : predicted_values,
			"r2" : result["r2"],
			"mae" : result["mae"],
			"rmse" : result["rmse"],
			"evs" : result["evs"],
			"kernel_matrix": kernel_matrix
		}
		self.test_result = test_result
		
		return test_result
	
	def predict(self, data, use_support_vector_index=True):
		"""予測
		
		Parameters
		----------
		data: array
			予測するデータ
		use_support_vector_index: boolean
			サポートベクター以外の演算をするかどうか

		Returns
		-------
		result: array
			結果のarrayを返す
		"""
		if use_support_vector_index:
			kernel_matrix = np.zeros((len(data), len(self.train_data)))
			kernel_matrix[:, self.support] = self.get_kernel_matrix(data, self.support_vectors)
		else:
			kernel_matrix = self.get_kernel_matrix(data, self.train_data)
			
		return self._predict(kernel_matrix)
		
	def _predict(self, kernel_matrix):
		"""予測
		
		Parameters
		----------
		kernel_matrix: array
			カーネル行列（予測するデータ）

		Returns
		-------
		result: array
			結果のarrayを返す
		"""
		return self.svr.predict(kernel_matrix)
	
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