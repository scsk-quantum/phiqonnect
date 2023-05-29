import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector, ParameterExpression
from qiskit.circuit.parametertable import ParameterTable
from qiskit.utils.deprecation import deprecate_arguments
from typing import Union, Optional, List

class PQC(QuantumCircuit):
	"""
	PQC
	"""
	def __init__(self, num_qubits:int= None, name: Optional[str] = 'base') -> None:
		"""Initialize the PQC object.

		Args:
			num_qubits: The number of qubits.
			name: The name of the circuit.
		"""
		
		super().__init__(num_qubits, name=name)
		
	def _parameter_generator(self, rep: int, block: int, indices: List[int]
							 ) -> Optional[List[Parameter]]:
		
		"""Generate the parameters for the circuit.

		Args:
			rep: The repetition number.
			block: The block number.
			indices: The indices of the parameters.

		Returns:
			The parameters.
		"""

		params = [self.ordered_parameters[i] for i in indices]
		return params

	@property
	def num_parameters_settable(self):
		return self.feature_dimension
	
	@property
	def feature_dimension(self) -> int:
		return self._feature_dimension
	
	@feature_dimension.setter
	def feature_dimension(self, feature_dimension: int) -> None:
		self._feature_dimension = feature_dimension
	
	@property
	def num_qubits(self) -> int:
		return self._num_qubits
	
	@property
	def ordered_parameters(self) -> List[Parameter]:
		if isinstance(self._ordered_parameters, ParameterVector):
			self._ordered_parameters.resize(self.num_parameters_settable)
			return list(self._ordered_parameters)

		return self._ordered_parameters

	@ordered_parameters.setter
	def ordered_parameters(self, parameters: Union[ParameterVector, List[Parameter]]) -> None:
		self._ordered_parameters = parameters
		self._invalidate()
	
	@property
	def num_parameters_settable(self):
		return self.feature_dimension

	def __str__(self) -> str:
		from qiskit.compiler import transpile
		basis_gates = ['id', 'x', 'y', 'z', 'h', 's', 't', 'sdg', 'tdg', 'rx', 'ry', 'rz', 'cz', 'swap']
		return transpile(self, basis_gates=basis_gates,
						 optimization_level=0).draw(output='text').single_string()
	
	def _invalidate(self):
		"""Invalidate the current circuit build."""
		self._data = None
		self._parameter_table = ParameterTable()
	
	def _build(self) -> None:
		return
		
	def _check_configuration(self, raise_on_failure: bool = True) -> bool:
		valid = True
		if self.num_qubits is None:
			valid = False
			if raise_on_failure:
				raise ValueError('No number of qubits specified.')

		return valid
	
	@deprecate_arguments({'param_dict': 'parameters'})
	def assign_parameters(self, parameters: Union[dict, List[float], List[Parameter],
												  ParameterVector],
						  inplace: bool = False,
						  param_dict: Optional[dict] = None
						  ) -> Optional[QuantumCircuit]:
		"""Assign parameters to the circuit.

		Args:
			parameters: The parameters to assign.
			inplace: If True, the parameters are assigned inplace.
			param_dict: Deprecated.

		Returns:
			The circuit with the assigned parameters.
		"""
		if self._data is None:
			self._build()
		if not isinstance(parameters, dict):
			unbound_parameters = [param for param in self._ordered_parameters if
								  isinstance(param, ParameterExpression)]
			used = set()
			unbound_unique_parameters = []
			for param in unbound_parameters:
				if param not in used:
					unbound_unique_parameters.append(param)
					used.add(param)

			parameters = dict(zip(unbound_unique_parameters, parameters))

		if inplace:
			new = [parameters.get(param, param) for param in self.ordered_parameters]
			self._ordered_parameters = new
		return super().assign_parameters(parameters, inplace=inplace)
		