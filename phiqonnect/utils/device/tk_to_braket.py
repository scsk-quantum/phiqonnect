# Copyright 2020-2022 Cambridge Quantum Computing
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Conversion from tket to AQT
"""

from typing import (
    cast,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    TYPE_CHECKING,
)
from pytket.circuit import Circuit, OpType  # type: ignore
from braket.circuits import Circuit as BK_Circuit  # type: ignore
from numpy import pi

if TYPE_CHECKING:
    from pytket.circuit import Node  # type: ignore


def tk_to_braket(tkcirc: Circuit, allqbs: Optional[Iterable[int]] = None) -> BK_Circuit:
    """
    Convert a tket :py:class:`Circuit` to a braket circuit.

    `tkcirc` must be a circuit having a single one-dimensional register of qubits.
    If `allqbs` is not provided then it is taken to be the qubit set of `tkcirc`.
    The resulting circuit will begin with an identity gate on all qubits in `allqbs`.
    This is to work around a quirk in braket where circuits whose qubit set contains
    gaps are rejected.

    Any Measure gates present in the circuit are ignored.


    This code duplicated from pytket and partially updated by SCSK.

    :param tkcirc: circuit to be converted
    :param allqbs: all qubits on braket device (superset of indices of tkcirc qubits)

    :returns: circuit converted to braket
    """
    bkcirc = BK_Circuit()
    if allqbs is None:
        allqbs = [qb.index[0] for qb in tkcirc.qubits]
    for qb in allqbs:
        bkcirc.i(qb)
    # Add commands
    for cmd in tkcirc.get_commands():
        qbs = [qb.index[0] for qb in cmd.qubits]
        op = cmd.op
        optype = op.type
        
        # override
        if optype == OpType.Barrier:
            continue
        
        params = op.params
        
        if optype == OpType.CCX:
            bkcirc.ccnot(*qbs)
        elif optype == OpType.CX:
            bkcirc.cnot(*qbs)
        elif optype == OpType.CU1:
            bkcirc.cphaseshift(*qbs, params[0] * pi)
        elif optype == OpType.CSWAP:
            bkcirc.cswap(*qbs)
        elif optype == OpType.CY:
            bkcirc.cy(*qbs)
        elif optype == OpType.CZ:
            bkcirc.cz(*qbs)
        elif optype == OpType.H:
            bkcirc.h(*qbs)
        elif optype == OpType.noop:
            pass
        elif optype == OpType.ISWAPMax:
            bkcirc.iswap(*qbs)
        elif optype == OpType.U1:
            bkcirc.phaseshift(*qbs, params[0] * pi)
        elif optype == OpType.Rx:
            bkcirc.rx(*qbs, params[0] * pi)
        elif optype == OpType.Ry:
            bkcirc.ry(*qbs, params[0] * pi)
        elif optype == OpType.Rz:
            bkcirc.rz(*qbs, params[0] * pi)
        elif optype == OpType.S:
            bkcirc.s(*qbs)
        elif optype == OpType.Sdg:
            bkcirc.si(*qbs)
        elif optype == OpType.SWAP:
            bkcirc.swap(*qbs)
        elif optype == OpType.T:
            bkcirc.t(*qbs)
        elif optype == OpType.Tdg:
            bkcirc.ti(*qbs)
        # V amd Vdg differ by a pi/4 phase from braket according to the get_matrix
        # methods. However, braket circuits do not seem to be phase-aware.
        elif optype == OpType.V:
            bkcirc.v(*qbs)
        elif optype == OpType.Vdg:
            bkcirc.vi(*qbs)
        elif optype == OpType.X:
            bkcirc.x(*qbs)
        elif optype == OpType.XXPhase:
            bkcirc.xx(*qbs, params[0] * pi)
        elif optype == OpType.ISWAP:
            bkcirc.xy(*qbs, params[0] * pi)
        elif optype == OpType.Y:
            bkcirc.y(*qbs)
        elif optype == OpType.YYPhase:
            bkcirc.yy(*qbs, params[0] * pi)
        elif optype == OpType.Z:
            bkcirc.z(*qbs)
        elif optype == OpType.ZZPhase:
            bkcirc.zz(*qbs, params[0] * pi)
        elif optype == OpType.Measure:
            # Not wanted by braket, but may have been introduced by contextual
            # optimization: ignore.
            pass
        else:
            raise NotImplementedError(f"Cannot convert {op.get_name()} to braket")
    return bkcirc