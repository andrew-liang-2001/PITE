from generator import *
import qiskit
from qiskit import QuantumRegister, ClassicalRegister, transpile, QuantumCircuit

from qiskit_aer import Aer
import matplotlib.pyplot as plt

# %% 4 site 1D Ising model

line_TIM_lattice = LineLattice(4, boundary_condition=BoundaryCondition.PERIODIC)

ising_model = h.IsingModel(line_TIM_lattice.uniform_parameters(1.0, 0.0))
print(ising_model)

# %% 2 site 1D Hubbard model
delta_tau = 0.1
gamma = 1
phi = 2 * np.arccos(np.exp(-2 * np.abs(gamma) * delta_tau))

line_FHM_lattice = LineLattice(2, boundary_condition=BoundaryCondition.PERIODIC)
fermi_hubbard = h.FermiHubbardModel(line_FHM_lattice.uniform_parameters(-0.1, -0.1), onsite_interaction=0.1)
print(fermi_hubbard.interaction_matrix())

fermionic_op = fermi_hubbard.second_q_op()
qubit_jw_op = JordanWignerMapper().map(fermionic_op)
diagonalised_FHM = np.linalg.eigh(qubit_jw_op.to_matrix())

print(f"Number of Pauli strings: {len(qubit_jw_op)}")
print(qubit_jw_op)

# %%
# initial state preparation for 2 site 1D Hubbard model
theta = np.pi  # produces singlet state 1/sqrt(2) |0110> - |1001>

FHM_init = qiskit.QuantumCircuit(QuantumRegister(1, "anc"), QuantumRegister(4))
FHM_init.h(1)
FHM_init.ry(theta, 1)
FHM_init.x(2)
FHM_init.x(4)
FHM_init.cx(1, 2)
FHM_init.cx(2, 3)
FHM_init.cx(3, 4)

# %%
n_trotter_steps = 2
# circuits = [IYZY, IXZX, YZYI, IIII, IIZI, IIIZ, IIZZ, ZIII, IZII, ZZII]
circuits = generate_circuits(qubit_jw_op, phi)

# %%
full_trotter_step = qiskit.QuantumCircuit(5)

# Repeat the sequence of circuits n_trotter_steps times
full_trotter_step.compose(FHM_init, inplace=True)
for _ in range(n_trotter_steps):
    for circ in circuits:
        full_trotter_step.compose(circ, inplace=True)

full_trotter_step.draw()


# %% shots simulation
backend = Aer.get_backend("aer_simulator")
full_trotter_step = transpile(full_trotter_step, backend, basis_gates=["sdg", "s", "h", "cx", "ry", "measure"])
full_trotter_step.draw()

# %%
job = backend.run(full_trotter_step, shots=10 ** 5)
job.result().get_counts()

# %% statevector simulation
backend = Aer.get_backend("statevector_simulator")

full_trotter_step = transpile(full_trotter_step, backend, basis_gates=["sdg", "s", "h", "cx", "ry", "measure"])
job = backend.run(full_trotter_step)

statevector = job.result().get_statevector()

# %%

def compute_energy(statevector, hamiltonian):
    return np.dot(np.conj(statevector), hamiltonian @ statevector).real


# %% Generate circuits for each Pauli string.

# %%
np.linalg.eigvalsh(qubit_jw_op.to_matrix())

# %% Example circuit
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


# %%
