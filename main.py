import numpy as np
import qiskit
import qiskit_nature.second_q.hamiltonians as h
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.hamiltonians.lattices import LineLattice

line_lattice = LineLattice(10)

ising_model = h.IsingModel(line_lattice.uniform_parameters(1.0, 0.0))
# fermi_hubbard = h.FermiHubbardModel(line_lattice.uniform_parameters(1.0, 0.0, 4.0, 0.0))
# do a 10 qubit state with all 0s

qc = qiskit.QuantumCircuit(4)
qc.h(0)
qc.sdg(1)
qc.h(1)
qc.cx(0, 1)
qc.cx(1, 2)
qc.crx(np.pi / 2, 2, 3)
qc.cx(1, 2)
qc.cx(0, 1)
qc.h(1)
qc.s(1)
qc.h(0)
print(qc.draw())
