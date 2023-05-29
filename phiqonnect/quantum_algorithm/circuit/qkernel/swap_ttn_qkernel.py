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

from phiqonnect.quantum_algorithm.circuit.qkernel.base.qkernel_ttn import QkernelTTN
from phiqonnect.quantum_algorithm.circuit.data_encoding.tensor_tree_network import TensorTreeNetwork

class SwapTTN_QKernel(QkernelTTN):
	"""
	SwapTTN_QKernel

	Parameters
	----------
	feature_dimension: int
		特徴量の次元数
	num_qubits: int
		量子ビット数
	encode_reverse: boolean
		逆順エンコードをするかどうか
	encoding_gate: array
		エンコーディング量子ゲートセット
	entanglement_gate: qiskit.circuit.gate.Gate
		エンタングルメントゲート
	parameter_prefix: str
		パラメータ記号
	name: str
		回路名
	"""

	def __init__(self, 
				feature_dimension: int, 
				num_qubits: int = None, 
				encode_reverse = False,
				encoding_gate = "Ry", 
				entanglement_gate = "Cx",
				parameter_prefix: str = 'x',
				name: Optional[str] = 'ttn') -> None:

			
		if num_qubits is None:
			if feature_dimension == 3:
				num_qubits = 2
			if feature_dimension == 7:
				num_qubits = 4
			
		measurement_basis = '0'
		
		base_qc = TensorTreeNetwork(feature_dimension,
				num_qubits,
				encode_reverse, 
				encoding_gate, 
				entanglement_gate,
				parameter_prefix,
				name)
				
		super().__init__(feature_dimension, num_qubits * 2, base_qc, measurement_basis)