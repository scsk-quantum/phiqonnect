import numpy as np
import qiskit
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import iSwapGate, CSwapGate, CXGate, RZGate, RYGate, HGate
from qiskit.utils.deprecation import deprecate_arguments
from typing import Union, Optional, List, Any, Tuple, Sequence, Set, Callable

from .base.pqc import PQC

class TensorTreeNetwork(PQC):
	"""
	TensorTreeNetwork
	"""
	def __init__(self, 
				feature_dimension: int, 
				num_qubits: Optional[int] = None, 
				encode_reverse: bool = False,
				encoding_gate: str = "Ry", 
				entanglement_gate: str = "Cx",
				reverse: bool = False,
				parameter_prefix: str = 'x',
				name: Optional[str] = 'tnn') -> None:
		
		if num_qubits is None:
			num_qubits = feature_dimension
			
		super().__init__(num_qubits, name)	
			
		self._num_qubits = num_qubits
		self._ordered_parameters = ParameterVector(name=parameter_prefix)
		self._data = None
		self._bounds = None
		
		self._feature_dimension = feature_dimension
		self.encode_reverse = encode_reverse
		self.encoding_gate = encoding_gate
		self._entanglement_gate = entanglement_gate
		self._reverse = reverse
		
		self._num_qubits = num_qubits
		self.qregs = [QuantumRegister(self._num_qubits, name='q')]
			
		self._build()
		
	def _build(self) -> None:
		
		if self._data:
			return
		
		_ = self._check_configuration()
		
		params = ParameterVector('x', self._feature_dimension)
		self._parameters = params
		self._ordered_parameters = params
		
		layer = QuantumCircuit(*self.qregs)
		qc = QuantumCircuit(self._num_qubits)
		
		#TODO 一般化
		# h = list(range(self._num_qubits))
		# for i in range(self._num_qubits):
		# 	for j in range(self._num_qubits):
				
			
			
		if self._feature_dimension == 3:
			if self._reverse == False:
				qc.append(RYGate(params[0]), [0])
				qc.append(RYGate(params[1]), [1])
				qc.append(CXGate(), [0, 1])
				qc.append(RYGate(params[2]), [1])
			else:
				qc.append(RYGate(params[2]), [0])
				qc.append(RYGate(params[1]), [1])
				qc.append(CXGate(), [0, 1])
				qc.append(RYGate(params[0]), [1])
			
		elif self._feature_dimension == 7:
			if self._reverse == False:
				qc.append(RYGate(params[0]), [0])
				qc.append(RYGate(params[2]), [1])
				qc.append(CXGate(), [0, 1])
				qc.append(RYGate(params[4]), [1])
				
				qc.append(RYGate(params[3]), [2])
				qc.append(RYGate(params[1]), [3])
				qc.append(CXGate(), [3, 2])
				qc.append(RYGate(params[5]), [2])
				
				qc.append(CXGate(), [1, 2])
				qc.append(RYGate(params[6]), [2])
			else:
				qc.append(RYGate(params[6]), [0])
				qc.append(RYGate(params[4]), [1])
				qc.append(CXGate(), [0, 1])
				qc.append(RYGate(params[2]), [1])
				
				qc.append(RYGate(params[3]), [2])
				qc.append(RYGate(params[5]), [3])
				qc.append(CXGate(), [3, 2])
				qc.append(RYGate(params[1]), [2])
				
				qc.append(CXGate(), [1, 2])
				qc.append(RYGate(params[0]), [2])
		
		layer.compose(qc, inplace=True)
		
		self._data = layer
		self.parameterized_circuit = layer
		
		self._parameter_table.clear()
		for instr, _, _ in self._data:
			self._update_parameter_table(instr)