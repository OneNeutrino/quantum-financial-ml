"""
Visualization tools for quantum circuits and quantum states in financial applications.

This module provides specialized visualization tools for quantum circuits
used in financial applications, including state visualization and circuit analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_bloch_multivector
from typing import Dict, List, Optional, Union, Tuple

class QuantumCircuitVisualizer:
    def __init__(self,
                circuit: QuantumCircuit,
                style: Optional[Dict] = None):
        """
        Initialize quantum circuit visualizer.
        
        Args:
            circuit: Quantum circuit to visualize
            style: Optional visualization style settings
        """
        self.circuit = circuit
        self.style = style or {
            'backgroundcolor': '#FFFFFF',
            'linecolor': '#000000',
            'textcolor': '#000000',
            'subfontsize': 12,
            'showindex': True
        }
        
    def visualize_state_evolution(self, 
                               feature_data: np.ndarray,
                               n_steps: int = 4) -> None:
        """
        Visualize quantum state evolution through circuit.
        
        Args:
            feature_data: Input feature data
            n_steps: Number of intermediate steps to show
        """
        # Create partial circuits for each step
        step_circuits = self._create_step_circuits(n_steps)
        
        # Simulate states
        states = []
        backend = Aer.get_backend('statevector_simulator')
        
        for circuit in step_circuits:
            job = execute(circuit, backend)
            state = job.result().get_statevector()
            states.append(state)
            
        # Plot state evolution
        fig, axes = plt.subplots(1, n_steps, figsize=(4*n_steps, 4))
        
        for i, state in enumerate(states):
            plot_bloch_multivector(state, ax=axes[i])
            axes[i].set_title(f'Step {i+1}')
            
        plt.tight_layout()
        plt.show()
        
    def visualize_feature_encoding(self,
                                features: np.ndarray,
                                encoding_type: str = 'amplitude') -> None:
        """
        Visualize feature encoding into quantum states.
        
        Args:
            features: Feature vector to encode
            encoding_type: Type of quantum encoding
        """
        # Encode features
        if encoding_type == 'amplitude':
            encoded_state = self._amplitude_encode(features)
        elif encoding_type == 'angle':
            encoded_state = self._angle_encode(features)
        else:
            raise ValueError(f'Unknown encoding type: {encoding_type}')
            
        # Plot encoded state
        fig = plt.figure(figsize=(8, 6))
        plot_bloch_multivector(encoded_state)
        plt.title(f'Feature Encoding ({encoding_type})')
        plt.show()
        
    def plot_measurement_distribution(self,
                                  n_shots: int = 1000) -> None:
        """
        Plot measurement outcome distribution.
        
        Args:
            n_shots: Number of circuit executions
        """
        # Execute circuit with shots
        backend = Aer.get_backend('qasm_simulator')
        job = execute(self.circuit, backend, shots=n_shots)
        result = job.result()
        counts = result.get_counts()
        
        # Plot distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        x = range(len(counts))
        plt.bar(x, counts.values())
        plt.xticks(x, counts.keys(), rotation=45)
        plt.xlabel('Measurement Outcome')
        plt.ylabel('Counts')
        plt.title('Measurement Distribution')
        plt.tight_layout()
        plt.show()
        
    def _create_step_circuits(self, n_steps: int) -> List[QuantumCircuit]:
        """Create partial circuits for state evolution visualization."""
        instructions = self.circuit.data
        steps_per_circuit = max(1, len(instructions) // n_steps)
        
        partial_circuits = []
        for i in range(n_steps):
            end_idx = min((i + 1) * steps_per_circuit, len(instructions))
            qc = QuantumCircuit(self.circuit.num_qubits)
            
            for inst in instructions[:end_idx]:
                qc.append(inst[0], inst[1])
            
            partial_circuits.append(qc)
            
        return partial_circuits
        
    def _amplitude_encode(self, features: np.ndarray) -> np.ndarray:
        """Encode features using amplitude encoding."""
        # Normalize features
        features = features / np.linalg.norm(features)
        
        # Create and execute circuit
        qc = QuantumCircuit(self.circuit.num_qubits)
        qc.initialize(features, range(self.circuit.num_qubits))
        
        backend = Aer.get_backend('statevector_simulator')
        job = execute(qc, backend)
        state = job.result().get_statevector()
        
        return state
        
    def _angle_encode(self, features: np.ndarray) -> np.ndarray:
        """Encode features using angle encoding."""
        # Scale features to [0, 2π]
        features = 2 * np.pi * (features - np.min(features)) / (np.max(features) - np.min(features))
        
        # Create and execute circuit
        qc = QuantumCircuit(self.circuit.num_qubits)
        for i, feature in enumerate(features[:self.circuit.num_qubits]):
            qc.ry(feature, i)
            
        backend = Aer.get_backend('statevector_simulator')
        job = execute(qc, backend)
        state = job.result().get_statevector()
        
        return state