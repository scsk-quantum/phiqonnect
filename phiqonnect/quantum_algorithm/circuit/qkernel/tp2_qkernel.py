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

from phiqonnect.quantum_algorithm.circuit.qkernel.base.qkernel import Qkernel
from phiqonnect.quantum_algorithm.circuit.data_encoding.tp2 import TP2

class TP2_QKernel(Qkernel):
	"""
	TP2_QKernel

	Parameters
	----------
	feature_dimension: int
		特徴量の次元数
	num_qubits: int
		量子ビット数
	first_circuit: array
		第1エンコーディング量子ゲートセット
	latter_cirrcuit: array
		第2エンコーディング量子ゲートセット
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
				first_circuit = ["RY"], 
				latter_cirrcuit = [], 
				entanglement_gate = None,
				parameter_prefix: str = 'x',
				name: Optional[str] = 'tp2') -> None:
			
		if num_qubits is None:
			num_qubits = feature_dimension
			
		measurement_basis = '0' * num_qubits
		
		base_qc = TP2(feature_dimension,
				num_qubits,
				first_circuit, 
				latter_cirrcuit, 
				entanglement_gate,
				parameter_prefix,
				name)
				
		super().__init__(num_qubits, base_qc, measurement_basis)