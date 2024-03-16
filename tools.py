import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.quantum_info.operators import Operator


def get_statevector(circuit: QuantumCircuit) -> Statevector:
    """
    Get the statevector of a quantum circuit.
    :param circuit:
    :return:
    """
    # https://quantumcomputing.stackexchange.com/questions/28267/get-statevector-not-working-with-qiskit-aers-statevector-simulator
    return Statevector(circuit)


def post_select(statevector: Statevector) -> Statevector:
    """
    Post-select the ancilla qubit to the |0> state, assuming it to be the first qubit in the statevector.
    :param statevector: Qiskit statevector object.
    :return: Statevector object with the ancilla qubit post-selected to |0>.
    """
    reduced_state = statevector.data[::2] / np.linalg.norm(statevector.data[::2])
    statevector = Statevector(reduced_state)
    return statevector


def get_probability(statevector: Statevector) -> float:
    """
    Return the probability of measuring the ancilla qubit in the |0> state. Assume statevector is normalised to unity.
    :param statevector:
    :return:
    """
    return np.dot(statevector.data[::2], statevector.data[::2].conj())


def generate_circuit(statevector: Statevector, Pauli_coeff_pair: SparsePauliOp) -> (Statevector, float):
    """
    Generate quantum circuits for each Pauli string in the qubit Hamiltonian.
    :param Pauli_coeff_pair:
    :param statevector:
    :param Pauli_op: Pauli operator object from the qiskit_nature library.
    :param phi: Rotation angle for the CRX gate.
    :return: List of quantum circuits for each Pauli string.
    """
    string = Pauli_coeff_pair[0]
    coeff = Pauli_coeff_pair[1].real
    DELTA_TAU = 0.1
    num_qubits = len(string)
    phi = 2 * np.arccos(np.exp(-2 * np.abs(coeff) * DELTA_TAU))

    qc = QuantumCircuit(QuantumRegister(1, "anc"), QuantumRegister(num_qubits))
    qc.initialize(statevector, list(np.arange(1, num_qubits+1)))

    for i, pauli_gate in enumerate(string):
        if pauli_gate == "X":
            qc.h(i + 1)  # offset 1 for the ancillary qubit
        elif pauli_gate == "Y":
            qc.sdg(i + 1)
            qc.h(i + 1)

    # Implement the CNOT pauli gadget
    non_identity_indices = [i for i, gate in enumerate(string) if gate != "I"]
    for i in range(len(non_identity_indices) - 1):
        qc.cx(non_identity_indices[i] + 1, non_identity_indices[i + 1] + 1)

    if non_identity_indices:
        # Apply X gates to the ancilla for a negatively controlled action
        if coeff > 0:
            qc.x(non_identity_indices[-1] + 1)
        qc.crx(phi, non_identity_indices[-1] + 1, 0)  # Apply CRX gate
        if coeff > 0:
            qc.x(non_identity_indices[-1] + 1)

    new_statevector = get_statevector(qc)
    probability = get_probability(new_statevector)
    # print("Probability: ", probability)
    new_statevector = post_select(new_statevector)
    # print("New statevector: ", new_statevector.data)

    # print(np.dot(new_statevector.data, new_statevector.data.conj()))
    # print(new_statevector.data)
    qc2 = QuantumCircuit(QuantumRegister(1, "anc"), QuantumRegister(num_qubits))
    qc2.initialize(new_statevector, list(np.arange(1, num_qubits+1)))

    # Reverse the CNOT operations.
    for i in reversed(range(len(non_identity_indices) - 1)):
        qc2.cx(non_identity_indices[i] + 1, non_identity_indices[i + 1] + 1)

    # Reverse the gate applications for X and Y gates based on the Pauli string.
    for i, pauli_gate in enumerate(reversed(string)):
        if pauli_gate == "Y":
            qc2.h(num_qubits - i)
            qc2.s(num_qubits - i)
        elif pauli_gate == "X":
            qc2.h(num_qubits - i)

    result_statevector = get_statevector(qc2)
    result_statevector = post_select(result_statevector)
    return result_statevector, probability


def expectation_value(operator, statevector) -> float:
    """
    Compute the expectation value of a operator
    :param operator: Hermition operator object.
    :param statevector:
    :return:
    """
    operator = Operator(operator)
    return statevector.expectation_value(operator)


def compute_Hamiltonian_energy(Pauli_expectations, coefficients) -> float:
    """
    Compute the expectation of the Hamiltonian, where <H> = Σ_i c_i * <P_i>
    :param Pauli_expectations: Expectation values of the Pauli strings.
    :param coefficients: coefficients of the Pauli strings.
    :return: Expectation value of the Hamiltonian.
    """
    return np.dot(Pauli_expectations, coefficients.real)


def run_experiment(trotter_arr: np.ndarray, statevector_initial: Statevector, qubit_jw_op: SparsePauliOp) -> (list, list):
    """
    Run the PITE simulation
    :param trotter_arr: array of Trotter steps, where step size is when the energy is computed.
    :param statevector_initial: initial statevector of the system.
    :param qubit_jw_op:
    :return: list of energy values and list of probabilities.
    """
    n_trotter_steps = max(trotter_arr)
    energy_list = []
    probability_list = []
    statevector = statevector_initial
    p_success = 1

    for n in range(n_trotter_steps + 1):
        if n in trotter_arr:
            Pauli_expectations = []
            for Pauli in qubit_jw_op.paulis:
                Pauli_expectations.append(expectation_value(Pauli, statevector))
            energy_list.append(compute_Hamiltonian_energy(Pauli_expectations, qubit_jw_op.coeffs))
        for Pauli_coeff_tuple in qubit_jw_op.to_list():
            statevector, prob = generate_circuit(statevector, Pauli_coeff_tuple)
            p_success *= prob
        probability_list.append(p_success)
    return energy_list, probability_list


def matrix_expectation(matrix, statevector_array: np.ndarray):
    statevector_conj = statevector_array.transpose().conj()
    return statevector_conj @ matrix @ statevector_array
