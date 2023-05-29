import numpy as np
import qiskit
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import CU1Gate, HGate
from qiskit.utils.deprecation import deprecate_arguments
from typing import Union, Optional, List, Any, Tuple, Sequence, Set, Callable

from .base.pqc import PQC

class IQP1a(PQC):
	"""
	IQP1a
	"""
	def __init__(self, 
				num_qubits: int = 3, 
				repeated: int = 2,
				parameter_prefix: str = 'x',
				name: Optional[str] = 'iqp_1a') -> None:
			
		super().__init__(num_qubits, name)	
			
		self._num_qubits = num_qubits
		self._repeated = repeated
		self._ordered_parameters = ParameterVector(name=parameter_prefix)
		self._data = None
		self._bounds = None
		self.qregs = [QuantumRegister(self._num_qubits, name='q')]
		self._build()
		
	def _build(self) -> None:
		
		if self._data:
			return
		
		_ = self._check_configuration()
		params = ParameterVector('x', 1)
		self._parameters = params
		self._ordered_parameters = params
		
		layer: QuantumCircuit = QuantumCircuit(*self.qregs)
		
		for i in range(self._repeated):
			qc: QuantumCircuit = QuantumCircuit(self._num_qubits)
			for i in range(self._num_qubits):
				qc.append(HGate(), [i])
			layer.compose(qc, inplace=True)
			
			qc: QuantumCircuit = QuantumCircuit(self._num_qubits)
			for i in range(self._num_qubits - 1):
				qc.append(CU1Gate(params[0]), [i, i + 1])
			qc.append(CU1Gate(params[0]), [self._num_qubits - 1, 0])
			layer.compose(qc, inplace=True)
			
		qc: QuantumCircuit = QuantumCircuit(self._num_qubits)
		for i in range(self._num_qubits):
			qc.append(HGate(), [i])
		layer.compose(qc, inplace=True)
			   
		self._data = layer
		self.parameterized_circuit = layer
		
		self._parameter_table.clear()
		for instr, _, _ in self._data:
			self._update_parameter_table(instr)