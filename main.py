from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
import numpy as np
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.hamiltonians.lattices import LineLattice, BoundaryCondition
import qiskit_nature.second_q.hamiltonians as h
import matplotlib.pyplot as plt
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector, Pauli, partial_trace, random_statevector
from qiskit.quantum_info.operators import Operator
from qiskit.visualization import plot_histogram
backend = Aer.get_backend("statevector_simulator")


# def generate_circuits(Pauli_op, phi) -> [QuantumCircuit]:
#     """
#     Generate quantum circuits for each Pauli string in the qubit Hamiltonian.
#     :param Pauli_op: Pauli operator object from the qiskit_nature library.
#     :param phi: Rotation angle for the CRX gate.
#     :return: List of quantum circuits for each Pauli string.
#     """
#     Pauli_circuits = []
#     for Pauli in Pauli_op:
#         string = Pauli.paulis.to_labels()
#         coeff = Pauli.coeffs[0].real  # All coefficients should be real in this project, so just check real component.
#         num_qubits = len(string)
#
#         qc = QuantumCircuit(QuantumRegister(1, "anc"), QuantumRegister(4), ClassicalRegister(1))
#
#         for i, pauli_gate in enumerate(string):
#             if pauli_gate == "X":
#                 qc.h(i + 1)  # offset 1 for the ancillary qubit
#             elif pauli_gate == "Y":
#                 qc.sdg(i + 1)
#                 qc.h(i + 1)
#
#         # Implement the controlled-NOT chain.
#         non_identity_indices = [i for i, gate in enumerate(string) if gate != "I"]
#         for i in range(len(non_identity_indices) - 1):
#             qc.cx(non_identity_indices[i] + 1, non_identity_indices[i + 1] + 1)
#
#         # Apply CRX gate with the ancilla.
#         if non_identity_indices:
#             # Apply X gates to the ancilla for a negatively controlled action
#             if coeff > 0:
#                 qc.x(non_identity_indices[-1] + 1)
#             qc.crx(phi, non_identity_indices[-1] + 1, 0)  # Apply CRX gate
#             if coeff > 0:
#                 qc.x(non_identity_indices[-1] + 1)
#
#         # Measurement of the ancilla qubit.
#         qc.measure(0, 0)
#         qc.reset(0)
#
#         # Reverse the CNOT operations.
#         for i in reversed(range(len(non_identity_indices) - 1)):
#             qc.cx(non_identity_indices[i] + 1, non_identity_indices[i + 1] + 1)
#
#         # Reverse the gate applications for X and Y gates based on the Pauli string.
#         for i, pauli_gate in enumerate(reversed(string)):
#             if pauli_gate == "Y":
#                 qc.h(num_qubits - i)  # Apply Hadamard gate in reverse order.
#                 qc.s(num_qubits - i)  # Apply S gate to reverse the earlier S dagger gate.
#             elif pauli_gate == "X":
#                 qc.h(num_qubits - i)  # Apply Hadamard gate in reverse order.
#
#         # Append the configured circuit to the list of circuits.
#         Pauli_circuits.append(qc)
#
#     return Pauli_circuits


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

n_trotter_steps = 1

# %%

def generate_circuit(Statevector, Pauli):
    if Pauli == "IYZY":
        IYZY = QuantumCircuit(QuantumRegister(1, "anc"), QuantumRegister(4), ClassicalRegister(1, "c"))
        IYZY.initialize(Statevector, [1, 2, 3, 4])
        IYZY.sdg(2)
        IYZY.h(2)
        IYZY.sdg(4)
        IYZY.h(4)
        IYZY.cx(2, 3)
        IYZY.cx(3, 4)
        IYZY.crx(phi, 4, 0)
        IYZY.measure(0, 0)
        IYZY.cx(3, 4)
        IYZY.cx(2, 3)
        IYZY.h(4)
        IYZY.s(4)
        IYZY.h(2)
        IYZY.s(2)
        return IYZY
    elif Pauli == "IXZX":
        IXZX = QuantumCircuit(QuantumRegister(1, "anc"), QuantumRegister(4), ClassicalRegister(1))
        IXZX.initialize(Statevector, [1, 2, 3, 4])
        IXZX.h(2)
        IXZX.h(4)
        IXZX.cx(2, 3)
        IXZX.cx(3, 4)
        IXZX.crx(phi, 4, 0)
        IXZX.measure(0, 0)
        IXZX.cx(3, 4)
        IXZX.cx(2, 3)
        IXZX.h(4)
        IXZX.h(2)
        return IXZX
    elif Pauli == "IIII":
        IIII = QuantumCircuit(QuantumRegister(1, "anc"), QuantumRegister(4), ClassicalRegister(1))
        IIII.initialize(Statevector, [1, 2, 3, 4])
        return IIII
    elif Pauli == "IIIZ":
        IIIZ = QuantumCircuit(QuantumRegister(1, "anc"), QuantumRegister(4), ClassicalRegister(1))
        IIIZ.initialize(Statevector, [1, 2, 3, 4])
        IIIZ.x(4)
        IIIZ.crx(phi, 4, 2)
        IIIZ.x(4)
        IIIZ.measure(0, 0)
        return IIIZ
    elif Pauli == "IZII":
        IZII = QuantumCircuit(QuantumRegister(1, "anc"), QuantumRegister(4), ClassicalRegister(1))
        IZII.initialize(Statevector, [1, 2, 3, 4])
        IZII.x(2)
        IZII.crx(phi, 2, 3)
        IZII.x(2)
        IZII.measure(0, 0)
        return IZII
    elif Pauli == "YZYI":
        YZYI = QuantumCircuit(QuantumRegister(1, "anc"), QuantumRegister(4), ClassicalRegister(1))
        YZYI.initialize(Statevector, [1, 2, 3, 4])
        YZYI.sdg(1)
        YZYI.h(1)
        YZYI.sdg(3)
        YZYI.h(3)
        YZYI.cx(1, 2)
        YZYI.cx(2, 3)
        YZYI.crx(phi, 3, 0)
        YZYI.measure(0, 0)
        YZYI.cx(2, 3)
        YZYI.cx(1, 2)
        YZYI.h(3)
        YZYI.s(3)
        YZYI.h(1)
        YZYI.s(1)
        return YZYI
    elif Pauli == "XZXI":
        XZXI = QuantumCircuit(QuantumRegister(1, "anc"), QuantumRegister(4), ClassicalRegister(1))
        XZXI.initialize(Statevector, [1, 2, 3, 4])
        XZXI.h(1)
        XZXI.h(3)
        XZXI.cx(1, 2)
        XZXI.cx(2, 3)
        XZXI.crx(phi, 3, 0)
        XZXI.measure(0, 0)
        XZXI.cx(2, 3)
        XZXI.cx(1, 2)
        XZXI.h(3)
        XZXI.h(1)
        return XZXI
    elif Pauli == "IIZI":
        IIZI = QuantumCircuit(QuantumRegister(1, "anc"), QuantumRegister(4), ClassicalRegister(1))
        IIZI.initialize(Statevector, [1, 2, 3, 4])
        IIZI.x(3)
        IIZI.crx(phi, 3, 0)
        IIZI.x(3)
        IIZI.measure(0, 0)
        return IIZI
    elif Pauli == "ZIII":
        ZIII = QuantumCircuit(QuantumRegister(1, "anc"), QuantumRegister(4), ClassicalRegister(1))
        ZIII.initialize(Statevector, [1, 2, 3, 4])
        ZIII.x(1)
        ZIII.crx(phi, 1, 0)
        ZIII.x(1)
        ZIII.measure(0, 0)
        return ZIII
    elif Pauli == "IIZZ":
        IIZZ = QuantumCircuit(QuantumRegister(1, "anc"), QuantumRegister(4), ClassicalRegister(1))
        IIZZ.initialize(Statevector, [1, 2, 3, 4])
        IIZZ.cx(3, 4)
        IIZZ.x(4)
        IIZZ.crx(phi, 4, 0)
        IIZZ.x(4)
        IIZZ.measure(0, 0)
        IIZZ.cx(3, 4)
        return IIZZ
    elif Pauli == "ZZII":
        ZZII = QuantumCircuit(QuantumRegister(1, "anc"), QuantumRegister(4), ClassicalRegister(1))
        ZZII.initialize(Statevector, [1, 2, 3, 4])
        ZZII.cx(1, 2)
        ZZII.x(2)
        ZZII.crx(phi, 2, 0)
        ZZII.x(2)
        ZZII.measure(0, 0)
        ZZII.cx(1, 2)
        return ZZII

# %%
job = backend.run(IYZY)

statevector = job.result().get_statevector()
# keep only every second element of the statevector to get the reduced state
reduced_state = statevector.data[::2]/np.linalg.norm(statevector.data[::2])
reduced_state = Statevector(reduced_state)

pauli_zz = Pauli('IYZY')
operator_zz = Operator(pauli_zz)
expectation_value = reduced_state.expectation_value(operator_zz)
print(expectation_value)

 # %%

FHM_circuits = ["IYZY", "IXZX", "IIII", "IIIZ", "IZII", "YZYI", "XZXI", "IIZI", "ZIII", "IIZZ", "ZZII"]
statevector = Statevector.from_label("0000")
Pauli_expectations = []

for _ in range(n_trotter_steps):
    for circuit_str in FHM_circuits:
        circ = generate_circuit(statevector, circuit_str)
        circ = transpile(circ, backend, basis_gates=["sdg", "s", "h", "cx", "ry", "measure"])
        job = backend.run(circ)
        result = job.result()
        statevector = result.get_statevector()
        print(statevector.data)
        reduced_state = statevector.data[::2] / np.linalg.norm(statevector.data[::2])
        statevector = Statevector(reduced_state)  # overwrite the statevector with the reduced state
        pauli = Pauli(circuit_str)
        operator = Operator(pauli)
        expectation_value = statevector.expectation_value(operator)
        Pauli_expectations.append(expectation_value)
        print(f"Done with {circuit_str} with expectation value {expectation_value}")


def compute_Hamiltonian_energy(Pauli_expectations, coefficients):
    return np.dot(Pauli_expectations, coefficients)

print(compute_Hamiltonian_energy(Pauli_expectations, qubit_jw_op.coeffs))


# %%
# initial state preparation for 2 site 1D Hubbard model
theta = np.pi  # produces singlet state 1/sqrt(2) |0110> - |1001>

FHM_init = QuantumCircuit(QuantumRegister(1, "anc"), QuantumRegister(4), ClassicalRegister(10))
FHM_init.h(1)
FHM_init.ry(theta, 1)
FHM_init.x(2)
FHM_init.x(4)
FHM_init.cx(1, 2)
FHM_init.cx(2, 3)
FHM_init.cx(3, 4)
FHM_init.draw("mpl")
plt.show()

# %%
n_trotter_steps


# %%

def compute_energy(statevector, pauli_string):
    return np.dot(np.conj(statevector), hamiltonian @ statevector).real

# %% Generate circuits for each Pauli string.

# %%
np.linalg.eigvalsh(qubit_jw_op.to_matrix())


# %% Test circuit to figure out the correct ordering of the qubits.

test_circuit = QuantumCircuit(QuantumRegister(4))
test_circuit.x(2)
test_circuit = transpile(test_circuit, backend)
test_circuit.draw("mpl")
plt.show()

test_statevector = backend.run(test_circuit).result().get_statevector()
print(test_statevector)

# ordering is 0000, 1000, 0100, 1100, 0010, 1010, 0110, 1110, 0001, 1001, 0101, 1101, 0011, 1011, 0111, 1111

# # %% shots simulation
# backend = Aer.get_backend("aer_simulator")
# full_trotter_step = transpile(full_trotter_step, backend, basis_gates=["sdg", "s", "h", "cx", "ry", "measure"])
#
# # %%
#
# job = backend.run(full_trotter_step, shots=10**5)
# result = job.result()
# counts = result.get_counts()
#
#
# def proportion_of_zeros(counts):
#     return counts.get("0", 0) / sum(counts.values())
#
#
# print(proportion_of_zeros(counts))
#
# # %%
#
#
# def is_key_all_zeros(d):
#     for key in d.keys():
#         if all(char == '0' for char in key):
#             return True
#     return False
#
#
# while not is_key_all_zeros(counts):
#     job = backend.run(full_trotter_step, shots=1)
#     result = job.result()
#     counts = result.get_counts()
#
# print(counts)
#
