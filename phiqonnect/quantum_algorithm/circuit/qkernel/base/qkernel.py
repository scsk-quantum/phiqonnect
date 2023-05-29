from typing import Union, Optional, List, Any, Tuple, Sequence, Set, Callable
from itertools import combinations

import numpy as np
import math
import qiskit
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import iSwapGate, CSwapGate, CXGate, RZGate, RYGate, HGate
from qiskit.circuit import Instruction, Parameter, ParameterVector, ParameterExpression
from qiskit.circuit.parametertable import ParameterTable
from qiskit.utils.deprecation import deprecate_arguments
from qiskit.tools import parallel_map
from qiskit.compiler import transpile

from qiskit.circuit.library import BlueprintCircuit
from ...data_encoding.base.pqc import PQC

from pytket.extensions.qiskit import qiskit_to_tk
from .....utils.device.braket_qunatum_instance import BraketQuantumInstance
from .....utils.device.tk_to_braket import tk_to_braket

class Qkernel():
	"""
	Qkernel
	"""
	def __init__(self, 
				num_qubits: int = None, 
				base_qc: PQC = None,
				measurement_basis: str = None,
				batch_size: int = 1000) -> None:
		
		"""Initialize the Qkernel object.

		Args:
			num_qubits: The number of qubits.
			base_qc: The base quantum circuit.
			measurement_basis: The measurement basis.
			batch_size: The batch size.
		"""
				
		self.num_qubits = num_qubits
		self.base_qc = base_qc
		self.measurement_basis = measurement_basis
		self.batch_size = batch_size
		
		self.qc = None
	
	def draw(self, output=None, decompose=True):
		"""Draws a diagram of the circuit.

		Args:
			output: The output file to write. If None, then the output is
			displayed inline.
			decompose: Whether to decompose the circuit into a set of
			primitive operations.

		Returns:
			A matplotlib figure object.
		"""
		if self.qc is not None:
			qc =  self.qc
		else:
			xi_param_vec = ParameterVector('xi', self.num_qubits)
			xj_param_vec = ParameterVector('xj', self.num_qubits)
		
			qc = self._construct_circuit(xi_param_vec, xj_param_vec)
		
		if decompose:
			return qc.decompose().draw(output)
		else:
			return qc.draw(output)
	
	def __str__(self) -> str:
		"""Return the string representation of the circuit."""
		basis_gates = ['id', 'x', 'y', 'z', 'h', 's', 't', 'sdg', 'tdg', 'rx', 'ry', 'rz', 'cz', 'swap']
		
		xi_param_vec = ParameterVector('xi', self.num_qubits)
		xj_param_vec = ParameterVector('xj', self.num_qubits)
		
		if self.qc is not None:
			qc =  self.qc
		else:
			qc = self._construct_circuit(xi_param_vec, xj_param_vec)
		
		return transpile(qc, basis_gates=basis_gates,
						 optimization_level=0).draw(output='text').single_string()
	
	def construct_circuit(self):
		"""Construct a circuit that prepares the state |xi>|xj>.
		
		Returns:
			QuantumCircuit: A circuit that prepares the state |xi>|xj>.
		"""

		if self.qc is not None:
			return self.qc
		
		xi_param_vec = ParameterVector('xi', self.num_qubits)
		xj_param_vec = ParameterVector('xj', self.num_qubits)
		
		qc = self._construct_circuit(xi_param_vec, xj_param_vec)
		return qc
	
	def _construct_circuit(self, xi, xj):
		"""Construct a circuit that prepares the state |xi>|xj>.

		Args:
			xi (int): The value of the first qubit.
			xj (int): The value of the second qubit.

		Returns:
			QuantumCircuit: A circuit that prepares the state |xi>|xj>.
		"""
		q = QuantumRegister(self.num_qubits, 'q')
		c = ClassicalRegister(self.num_qubits, 'c')
		qc = QuantumCircuit(q, c)

		qc.barrier(range(self.num_qubits))

		psi_x1 = _assign_parameters(self.base_qc, xi)
		qc.append(psi_x1.to_instruction(), qc.qubits)
		
		qc.barrier(range(self.num_qubits))

		psi_x2_dag = _assign_parameters(self.base_qc, xj)
		qc.append(psi_x2_dag.to_instruction().inverse(), qc.qubits)

		qc.barrier(range(self.num_qubits))

		qc.measure(q, c)
		
		return qc
		
	@staticmethod
	def _compute(idx, results, measurement_basis):
		"""Compute the kernel value for a single pair of data points.

		Args:
			idx (int): The index of the pair of data points.
			results (list): The results of the quantum kernel.
			measurement_basis (str): The measurement basis.

		Returns:
			float: The kernel value.
		"""

		result = results.get_counts(idx)
		kernel_value = result.get(measurement_basis, 0) / sum(result.values())
		return kernel_value
		
	def get_kernel_matrix(self, xi_vec, xj_vec=None, quantum_instance=None):
		"""Compute the kernel matrix.

		Args:
			xi_vec (numpy.ndarray): The first vector of data points.
			xj_vec (numpy.ndarray): The second vector of data points.
			quantum_instance (QuantumInstance): The quantum instance used to run the circuits.

		Returns:
			numpy.ndarray: The kernel matrix.
		"""
		
		# If the quantum instance is not provided, then the default is to use the statevector simulator.
		if type(quantum_instance) == BraketQuantumInstance:
			# If the quantum instance is a BraketQuantumInstance, then the backend is the Braket.
			if xj_vec is None:
				xj_vec = xi_vec
				sym = True
			else:
				sym = False

			mat = np.ones((xi_vec.shape[0], xj_vec.shape[0]))
			mus, nus = np.indices((xi_vec.shape[0], xj_vec.shape[0]))
			mus = np.asarray(mus.flat)
			nus = np.asarray(nus.flat)

			max_i = len(xi_vec)
			max_j = len(xj_vec)
			shots = quantum_instance.shots

			device = quantum_instance.device

			task = []
			task_count = 0
			shot_count = 0
			
			if sym:
				# If the kernel matrix is symmetric, then only compute the upper triangle.
				for i in range(max_i):
					for j in range(i, max_j):
						xi_param_vec = ParameterVector('xi', self.num_qubits)
						xj_param_vec = ParameterVector('xj', self.num_qubits)

						qc = self._construct_circuit(xi_param_vec, xj_param_vec)
						# qc = quantum_instance.transpile(qc)[0]
						circuit = qc.assign_parameters({xi_param_vec: xi_vec[i], xj_param_vec: xj_vec[j]}).decompose()

						# Braketの場合はqiskitのcircuitをtkcに変換してからbraketのcircuitに変換する
						tkc = qiskit_to_tk(circuit)
						braket_circuit = tk_to_braket(tkc)
						
						# if task_count % 50 == 0 :
							# 確認用: 50回ごとに進捗を表示
							# print(task_count)
						task_count += 1
						shot_count += shots

						result = device.run(braket_circuit, shots=shots) #これを実行すると料金が発生するので注意！！！！！
						task.append(result)
			else:
				# If the kernel matrix is not symmetric, then compute the entire matrix.
				for i in range(max_i):
					for j in range(max_j):
						xi_param_vec = ParameterVector('xi', self.num_qubits)
						xj_param_vec = ParameterVector('xj', self.num_qubits)

						qc = self._construct_circuit(xi_param_vec, xj_param_vec)
						# qc = quantum_instance.transpile(qc)[0]
						circuit = qc.assign_parameters({xi_param_vec: xi_vec[i], xj_param_vec: xj_vec[j]}).decompose()

						# Braketの場合はqiskitのcircuitをtkcに変換してからbraketのcircuitに変換する
						tkc = qiskit_to_tk(circuit)
						braket_circuit = tk_to_braket(tkc)

						# if task_count % 50 == 0 :
							# 確認用: 50回ごとに進捗を表示
							# print(task_count)
						task_count += 1
						shot_count += shots

						result = device.run(braket_circuit, shots=shots) #これを実行すると料金が発生するので注意！！！！！
						task.append(result) 
			
			measurement_basis = self.measurement_basis
			
			task_count = 0
			if max_i == max_j:
				for i in range(max_i):
					for j in range(i, max_j):
						result = task[task_count].result()
						if result is not None:
							mat[i, j] = result.measurement_counts[measurement_basis] / shots
						else:
							mat[i, j] = 0.0
						mat[j, i] = mat[i, j]
						task_count += 1
			else:
				for i in range(max_i):
					for j in range(max_j):
						result = task[task_count].result()
						if result is not None:
							mat[i, j] = result.measurement_counts[measurement_basis] / shots
						else:
							mat[i, j] = 0.0
						task_count += 1   
							
			return mat
		else:
			# If the quantum instance is not a BraketQuantumInstance, then the backend is the destinated backend.
			if xj_vec is None:
				xj_vec = xi_vec
				sym = True
			else:
				sym = False

			mat = np.ones((xi_vec.shape[0], xj_vec.shape[0]))
			mus, nus = np.indices((xi_vec.shape[0], xj_vec.shape[0]))
			mus = np.asarray(mus.flat)
			nus = np.asarray(nus.flat)

			for _i in range(0, len(mus), self.batch_size):
				pair = []
				idx = []

				for _j in range(_i, min(_i + self.batch_size, len(mus))):
					i = mus[_j]
					j = nus[_j]
					_xi = xi_vec[i]
					_xj = xj_vec[j]
					if not np.all(_xi == _xj):
						pair.append((_xi, _xj))
						idx.append((i, j))
					if(i == j):
						pair.append((_xi, _xj))
						idx.append((i, j))       
				xi_param_vec = ParameterVector('xi', self.num_qubits)
				xj_param_vec = ParameterVector('xj', self.num_qubits)
				
				qc = self._construct_circuit(xi_param_vec, xj_param_vec)
				qc = quantum_instance.transpile(qc)[0]

				# 全組み合わせのcircuitsを作成
				circuits = [qc.assign_parameters({xi_param_vec: xi, xj_param_vec: xj}) for xi, xj in pair]
				results = quantum_instance.execute(circuits, had_transpiled=True)
				matrix_elements = parallel_map(Qkernel._compute, range(len(circuits)), task_args=(results, self.measurement_basis))

				for (i, j), value in zip(idx, matrix_elements):
						mat[i, j] = value
						if sym:
							mat[j, i] = mat[i, j]
						
			return mat
		
def _assign_parameters(circuit, params):
	"""Assigns parameters to a circuit.
	
	Args:
		circuit (QuantumCircuit): The circuit to assign parameters to.
		params (list): The parameters to assign.
		
	Returns:
		QuantumCircuit: The circuit with parameters assigned.	
	"""
	param_dict = dict(zip(circuit.ordered_parameters, params))
	return circuit.assign_parameters(param_dict)