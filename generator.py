from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import numpy as np
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.hamiltonians.lattices import LineLattice, BoundaryCondition
import qiskit_nature.second_q.hamiltonians as h


def generate_circuits(Pauli_op, phi) -> [QuantumCircuit]:
    """
    Generate quantum circuits for each Pauli string in the qubit Hamiltonian.
    :param Pauli_op: Pauli operator object from the qiskit_nature library.
    :param phi: Rotation angle for the CRX gate.
    :return: List of quantum circuits for each Pauli string.
    """
    Pauli_circuits = []
    for Pauli in Pauli_op:
        string = Pauli.paulis.to_labels()
        coeff = Pauli.coeffs[0].real  # All coefficients should be real in this project, so just check real component.
        num_qubits = len(string)

        qc = QuantumCircuit(QuantumRegister(1, "anc"), QuantumRegister(4), ClassicalRegister(1))

        for i, pauli_gate in enumerate(string):
            if pauli_gate == "X":
                qc.h(i + 1)  # offset 1 for the ancillary qubit
            elif pauli_gate == "Y":
                qc.sdg(i + 1)
                qc.h(i + 1)

        # Implement the controlled-NOT chain.
        non_identity_indices = [i for i, gate in enumerate(string) if gate != "I"]
        for i in range(len(non_identity_indices) - 1):
            qc.cx(non_identity_indices[i] + 1, non_identity_indices[i + 1] + 1)

        # Apply CRX gate with the ancilla.
        if non_identity_indices:
            # Apply X gates to the ancilla for a negatively controlled action
            if coeff > 0:
                qc.x(non_identity_indices[-1] + 1)
            qc.crx(phi, non_identity_indices[-1] + 1, 0)  # Apply CRX gate
            if coeff > 0:
                qc.x(non_identity_indices[-1] + 1)


        # Measurement of the ancilla qubit.
        qc.measure(0, 0)
        qc.reset(0)

        # Reverse the CNOT operations.
        for i in reversed(range(len(non_identity_indices) - 1)):
            qc.cx(non_identity_indices[i] + 1, non_identity_indices[i + 1] + 1)

        # Reverse the gate applications for X and Y gates based on the Pauli string.
        for i, pauli_gate in enumerate(reversed(string)):
            if pauli_gate == "Y":
                qc.h(num_qubits - i)  # Apply Hadamard gate in reverse order.
                qc.s(num_qubits - i)  # Apply S gate to reverse the earlier S dagger gate.
            elif pauli_gate == "X":
                qc.h(num_qubits - i)  # Apply Hadamard gate in reverse order.

        # Append the configured circuit to the list of circuits.
        Pauli_circuits.append(qc)

    return Pauli_circuits


if __name__ == "__main__":
    delta_tau = 0.1
    gamma = 1
    phi = 2 * np.arccos(np.exp(-2 * np.abs(gamma) * delta_tau))

    line_FHM_lattice = LineLattice(2, boundary_condition=BoundaryCondition.PERIODIC)
    fermi_hubbard = h.FermiHubbardModel(line_FHM_lattice.uniform_parameters(-0.1, -0.1), onsite_interaction=0.1)

    qubit_jw_op = JordanWignerMapper().map(fermi_hubbard.second_q_op())
    print(qubit_jw_op)
    circuits = generate_circuits(qubit_jw_op, phi)
    print(circuits[3].draw())
