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

from qiskit.circuit.library import BlueprintCircuit
from ...data_encoding.base.pqc import PQC

from pytket.extensions.qiskit import qiskit_to_tk
from .....utils.device.braket_qunatum_instance import BraketQuantumInstance
from .....utils.device.tk_to_braket import tk_to_braket

class QkernelTTN():
	"""
	QkernelTTN
	"""
	def __init__(self, 
				feature_dimension: int, 
				num_qubits: int = None, 
				base_qc: PQC = None,
				measurement_basis: str = None,
				batch_size: int = 1000) -> None:
				
		self.feature_dimension = feature_dimension
		self.num_qubits = num_qubits
		self.base_qc = base_qc
		self.measurement_basis = measurement_basis
		self.batch_size = batch_size
		
		self.qc = None
	
	def draw(self, output=None, decompose=True):
		if self.qc is not None:
			qc =  self.qc
		else:
			xi_param_vec = ParameterVector('xi', self.feature_dimension)
			xj_param_vec = ParameterVector('xj', self.feature_dimension)
		
			qc = self._construct_circuit(xi_param_vec, xj_param_vec)
		
		if decompose:
			return qc.decompose().draw(output)
		else:
			return qc.draw(output)
	
	def __str__(self) -> str:
		from qiskit.compiler import transpile
		basis_gates = ['id', 'x', 'y', 'z', 'h', 's', 't', 'sdg', 'tdg', 'rx', 'ry', 'rz', 'cz', 'swap']
		
		xi_param_vec = ParameterVector('xi', self.feature_dimension)
		xj_param_vec = ParameterVector('xj', self.feature_dimension)
		
		if self.qc is not None:
			qc =  self.qc
		else:
			qc = self._construct_circuit(xi_param_vec, xj_param_vec)
		
		return transpile(qc, basis_gates=basis_gates,
						 optimization_level=0).draw(output='text').single_string()
	
	def construct_circuit(self):
		if self.qc is not None:
			return self.qc
		
		xi_param_vec = ParameterVector('xi', self.feature_dimension)
		xj_param_vec = ParameterVector('xj', self.feature_dimension)
		
		qc = self._construct_circuit(xi_param_vec, xj_param_vec)
		return qc
	
	#TODO 4qubits以上
	def _construct_circuit(self, xi, xj):
		if self.num_qubits == 4:
			q = QuantumRegister(5, 'q')
			c = ClassicalRegister(1, 'c')
			qc = QuantumCircuit(q, c)
		
			qc.barrier(q)
			psi_x1 = _assign_parameters(self.base_qc, xi)
			qc.append(psi_x1.to_instruction(), [1, 2])
			psi_x2 = _assign_parameters(self.base_qc, xj)
			qc.append(psi_x2.to_instruction(), [3, 4])
			
			qc.barrier(q)
			q_swap = QuantumRegister(4, 'q')
			qc_swap = QuantumCircuit(q, name="swap_ttn")
			qc_swap.append(HGate(), [0])
			qc_swap.append(CSwapGate(), [0, 2, 4])
			qc_swap.append(HGate(), [0])
			qc.append(qc_swap.to_instruction(), qc_swap.qubits)
		elif self.num_qubits == 8:
			q = QuantumRegister(9, 'q')
			c = ClassicalRegister(1, 'c')
			qc = QuantumCircuit(q, c)
			
			qc.barrier(q)
			psi_x1 = _assign_parameters(self.base_qc, xi)
			qc.append(psi_x1.to_instruction(), [1, 2, 3, 4])
			psi_x2 = _assign_parameters(self.base_qc, xj)
			qc.append(psi_x2.to_instruction(), [5, 6, 7, 8])
			
			qc.barrier(q)
			q_swap = QuantumRegister(8, 'q')
			qc_swap = QuantumCircuit(q, name="swap_ttn")
			qc_swap.append(HGate(), [0])
			qc_swap.append(CSwapGate(), [0, 3, 7])
			qc_swap.append(HGate(), [0])
			qc.append(qc_swap.to_instruction(), qc_swap.qubits)
		else:
			q = QuantumRegister(self.num_qubits + 1, 'q')
			c = ClassicalRegister(1, 'c')
			qc = QuantumCircuit(q, c)
			
			
		
		qc.barrier(q)
		qc.measure(q[0], c)
		
		return qc
		
	@staticmethod
	def _compute(idx, results, measurement_basis):
		result = results.get_counts(idx)
		kernel_value = result.get(measurement_basis, 0) / sum(result.values())
		return kernel_value
		
	def get_kernel_matrix(self, xi_vec, xj_vec=None, quantum_instance=None):
		if type(quantum_instance) == BraketQuantumInstance:
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
				for i in range(max_i):
					for j in range(i, max_j):
						xi_param_vec = ParameterVector('xi', self.num_qubits)
						xj_param_vec = ParameterVector('xj', self.num_qubits)

						qc = self._construct_circuit(xi_param_vec, xj_param_vec)
						qc = quantum_instance.transpile(qc)[0]
						circuit = qc.assign_parameters({xi_param_vec: xi_vec[i], xj_param_vec: xj_vec[j]})

						tkc = qiskit_to_tk(circuit)
						braket_circuit = tk_to_braket(tkc)
						
						if task_count % 50 == 0 :
							print(task_count)
						task_count += 1
						shot_count += shots

						result = device.run(braket_circuit, shots=shots) #これを実行すると料金が発生するので注意！！！！！
						task.append(result)
			else:
				for i in range(max_i):
					for j in range(max_j):
						xi_param_vec = ParameterVector('xi', self.num_qubits)
						xj_param_vec = ParameterVector('xj', self.num_qubits)

						qc = self._construct_circuit(xi_param_vec, xj_param_vec)
						qc = quantum_instance.transpile(qc)[0]
						circuit = qc.assign_parameters({xi_param_vec: xi_vec[i], xj_param_vec: xj_vec[j]})

						tkc = qiskit_to_tk(circuit)
						braket_circuit = tk_to_braket(tkc)

						if task_count % 50 == 0 :
							print(task_count)
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

				xi_param_vec = ParameterVector('xi', self.feature_dimension)
				xj_param_vec = ParameterVector('xj', self.feature_dimension)
				
				qc = self._construct_circuit(xi_param_vec, xj_param_vec)
				qc = quantum_instance.transpile(qc)[0]
				circuits = [qc.assign_parameters({xi_param_vec: xi, xj_param_vec: xj}) for xi, xj in pair]
				results = quantum_instance.execute(circuits, had_transpiled=True)
				matrix_elements = parallel_map(QkernelTTN._compute, range(len(circuits)), task_args=(results, self.measurement_basis))

				for (i, j), value in zip(idx, matrix_elements):
						mat[i, j] = value
						if sym:
							mat[j, i] = mat[i, j]
						
			return mat
		
def _assign_parameters(circuit, params):
	param_dict = dict(zip(circuit.ordered_parameters, params))
	return circuit.assign_parameters(param_dict)