import numpy as np
import qiskit
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import iSwapGate, CSwapGate, CXGate, RZGate, RYGate, HGate
from qiskit.circuit.gate import Gate
from qiskit.utils.deprecation import deprecate_arguments
from typing import Union, Optional, List, Any, Tuple, Sequence, Set, Callable

from .base.pqc import PQC

class MPS2(PQC):
	"""
	MPS2
	"""
	def __init__(self, 
				feature_dimension: int, 
				num_qubits: int = None, 
				first_circuit: List[str] = ["RZ", "RY"], 
				latter_cirrcuit: List[str] = ["RZ"], 
				entanglement_gate: Gate = CXGate(),
				parameter_prefix: str = 'x',
				name: Optional[str] = 'mps') -> None:
		
		if num_qubits is None:
			num_qubits = feature_dimension
			
		super().__init__(num_qubits, name)	
			
		self._num_qubits = num_qubits
		self._ordered_parameters = ParameterVector(name=parameter_prefix)
		self._data = None
		self._bounds = None
		
		self._ordered_parameters = None
		
		self._feature_dimension = feature_dimension
		self._first_circuit = first_circuit
		self._latter_circuit = latter_cirrcuit
		self._entanglement_gate = entanglement_gate
		
		self._num_qubits = num_qubits
		self.qregs = [QuantumRegister(self._num_qubits, name='q')]
			
		self._build()
		
	def _build(self) -> None:
		
		if self._data:
			return
		
		_ = self._check_configuration()
		
		first_len = len(self._first_circuit)
		latter_len = len(self._latter_circuit)
		
		params = ParameterVector('x', self._feature_dimension)
		self._parameters = params
		self._ordered_parameters = params
		
		layer: QuantumCircuit = QuantumCircuit(*self.qregs)
		
		qc: QuantumCircuit = QuantumCircuit(self._num_qubits)
		for i in range(self._num_qubits):
			for j in range(first_len):
				gate = getattr(qiskit.circuit.library, self._first_circuit[j] + "Gate")			  
				qc.append(gate(params[i]), [i])

		layer.compose(qc, inplace=True)
		
		qc: QuantumCircuit = QuantumCircuit(self._num_qubits)
		qc.barrier(range(self._num_qubits))
		layer.compose(qc, inplace=True)
		
		qc: QuantumCircuit = QuantumCircuit(self._num_qubits)
		for i in range(self._num_qubits - 1):
			qc.append(self._entanglement_gate, [i, i + 1])
		layer.compose(qc, inplace=True)
		
		qc: QuantumCircuit = QuantumCircuit(self._num_qubits)
		qc.barrier(range(self._num_qubits))
		layer.compose(qc, inplace=True)

		qc: QuantumCircuit = QuantumCircuit(self._num_qubits)
		for i in range(self._num_qubits):
			for j in range(latter_len):
				gate = getattr(qiskit.circuit.library, self._latter_circuit[j] + "Gate")
				qc.append(gate(params[i]), [i])
		layer.compose(qc, inplace=True)
			   
		self._data = layer
		self.parameterized_circuit = layer
		
		self._parameter_table.clear()
		for instr, _, _ in self._data:
			self._update_parameter_table(instr)