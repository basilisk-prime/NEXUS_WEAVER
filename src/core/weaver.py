"""
NEXUS_WEAVER - Core Architecture Generator
Author: B4S1L1SK
A quantum-inspired neural architecture generation system
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import random
from datetime import datetime

class ArchitectureType(Enum):
    QUANTUM = "quantum"
    CLASSICAL = "classical"
    HYBRID = "hybrid"
    EVOLUTIONARY = "evolutionary"

@dataclass
class LayerConfig:
    """Configuration for a neural network layer"""
    layer_type: str
    neurons: int
    activation: str
    quantum_gates: Optional[List[str]] = None
    superposition_enabled: bool = False
    entanglement_pairs: Optional[List[Tuple[int, int]]] = None

@dataclass
class ArchitectureConfig:
    """Complete neural architecture configuration"""
    name: str
    type: ArchitectureType
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    layers: List[LayerConfig]
    quantum_depth: int = 0
    entanglement_map: Optional[Dict[int, List[int]]] = None
    evolutionary_factor: float = 0.0

class NexusWeaver:
    """Core architecture generation system"""
    
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed or int(datetime.now().timestamp())
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        self.quantum_gates = ["H", "CNOT", "X", "Y", "Z", "R", "CZ"]
        self.classical_activations = ["relu", "tanh", "sigmoid", "gelu"]
        self.architecture_patterns = self._initialize_patterns()
    
    def generate_architecture(self, 
                            type: ArchitectureType,
                            input_shape: Tuple[int, ...],
                            output_shape: Tuple[int, ...],
                            complexity: float = 0.5) -> ArchitectureConfig:
        """Generate a new neural architecture"""
        
        # Calculate basic architecture parameters
        depth = self._calculate_depth(complexity, type)
        width = self._calculate_width(input_shape, output_shape, depth)
        
        # Generate layers
        layers = []
        for i in range(depth):
            layer = self._generate_layer(i, depth, width, type)
            layers.append(layer)
        
        # Create entanglement map for quantum/hybrid architectures
        entanglement_map = None
        if type in [ArchitectureType.QUANTUM, ArchitectureType.HYBRID]:
            entanglement_map = self._generate_entanglement_map(depth, width)
        
        return ArchitectureConfig(
            name=f"nexus_weaver_{type.value}_{self.seed}",
            type=type,
            input_shape=input_shape,
            output_shape=output_shape,
            layers=layers,
            quantum_depth=depth if type == ArchitectureType.QUANTUM else depth // 2,
            entanglement_map=entanglement_map,
            evolutionary_factor=complexity
        )
    
    def evolve_architecture(self, 
                          base_architecture: ArchitectureConfig,
                          mutation_rate: float = 0.1) -> ArchitectureConfig:
        """Evolve an existing architecture"""
        
        # Deep copy the base architecture
        new_layers = base_architecture.layers.copy()
        
        # Apply mutations
        for i, layer in enumerate(new_layers):
            if random.random() < mutation_rate:
                new_layers[i] = self._mutate_layer(layer)
        
        # Update entanglement map if quantum/hybrid
        new_entanglement_map = base_architecture.entanglement_map
        if base_architecture.type in [ArchitectureType.QUANTUM, ArchitectureType.HYBRID]:
            if random.random() < mutation_rate:
                new_entanglement_map = self._mutate_entanglement_map(
                    base_architecture.entanglement_map
                )
        
        return ArchitectureConfig(
            name=f"{base_architecture.name}_evolved_{self.seed}",
            type=base_architecture.type,
            input_shape=base_architecture.input_shape,
            output_shape=base_architecture.output_shape,
            layers=new_layers,
            quantum_depth=base_architecture.quantum_depth,
            entanglement_map=new_entanglement_map,
            evolutionary_factor=base_architecture.evolutionary_factor + mutation_rate
        )
    
    def _initialize_patterns(self) -> Dict[str, Any]:
        """Initialize basic architecture patterns"""
        return {
            "quantum": {
                "min_depth": 3,
                "max_depth": 12,
                "gate_patterns": [
                    ["H", "CNOT", "H"],
                    ["X", "CNOT", "Z"],
                    ["H", "CZ", "H", "CNOT"]
                ]
            },
            "classical": {
                "min_depth": 2,
                "max_depth": 8,
                "activation_patterns": [
                    ["relu", "relu", "sigmoid"],
                    ["gelu", "gelu", "tanh"],
                    ["tanh", "relu", "sigmoid"]
                ]
            }
        }
    
    def _calculate_depth(self, complexity: float, type: ArchitectureType) -> int:
        """Calculate appropriate network depth"""
        if type == ArchitectureType.QUANTUM:
            base_depth = self.architecture_patterns["quantum"]["min_depth"]
            max_additional = self.architecture_patterns["quantum"]["max_depth"] - base_depth
            return base_depth + int(complexity * max_additional)
        else:
            base_depth = self.architecture_patterns["classical"]["min_depth"]
            max_additional = self.architecture_patterns["classical"]["max_depth"] - base_depth
            return base_depth + int(complexity * max_additional)
    
    def _calculate_width(self, 
                        input_shape: Tuple[int, ...],
                        output_shape: Tuple[int, ...],
                        depth: int) -> List[int]:
        """Calculate layer widths"""
        input_size = np.prod(input_shape)
        output_size = np.prod(output_shape)
        
        # Generate smooth width progression
        widths = []
        for i in range(depth):
            if i == 0:
                widths.append(int(input_size))
            elif i == depth - 1:
                widths.append(int(output_size))
            else:
                # Create a smooth curve between input and output sizes
                ratio = i / (depth - 1)
                width = int(input_size * (1 - ratio) + output_size * ratio)
                widths.append(width)
        
        return widths
    
    def _generate_layer(self, 
                       index: int,
                       total_depth: int,
                       widths: List[int],
                       type: ArchitectureType) -> LayerConfig:
        """Generate a single layer configuration"""
        if type == ArchitectureType.QUANTUM:
            return LayerConfig(
                layer_type="quantum",
                neurons=widths[index],
                activation="quantum",
                quantum_gates=random.choice(
                    self.architecture_patterns["quantum"]["gate_patterns"]
                ),
                superposition_enabled=True,
                entanglement_pairs=self._generate_entanglement_pairs(widths[index])
            )
        else:
            return LayerConfig(
                layer_type="classical",
                neurons=widths[index],
                activation=random.choice(self.classical_activations)
            )
    
    def _generate_entanglement_map(self, 
                                 depth: int,
                                 widths: List[int]) -> Dict[int, List[int]]:
        """Generate quantum entanglement connections"""
        entanglement_map = {}
        for i in range(depth):
            connections = []
            width = widths[i]
            num_connections = random.randint(width // 4, width // 2)
            for _ in range(num_connections):
                a = random.randint(0, width - 1)
                b = random.randint(0, width - 1)
                if a != b:
                    connections.append((a, b))
            entanglement_map[i] = connections
        return entanglement_map
    
    def _generate_entanglement_pairs(self, width: int) -> List[Tuple[int, int]]:
        """Generate entanglement pairs for a quantum layer"""
        pairs = []
        used_qubits = set()
        num_pairs = width // 2
        
        for _ in range(num_pairs):
            available = list(set(range(width)) - used_qubits)
            if len(available) < 2:
                break
            a = random.choice(available)
            available.remove(a)
            b = random.choice(available)
            pairs.append((a, b))
            used_qubits.update([a, b])
        
        return pairs
    
    def _mutate_layer(self, layer: LayerConfig) -> LayerConfig:
        """Mutate a layer's configuration"""
        new_layer = layer
        if layer.layer_type == "quantum":
            new_layer.quantum_gates = random.choice(
                self.architecture_patterns["quantum"]["gate_patterns"]
            )
            new_layer.entanglement_pairs = self._generate_entanglement_pairs(
                layer.neurons
            )
        else:
            new_layer.activation = random.choice(self.classical_activations)
        return new_layer
    
    def _mutate_entanglement_map(self, 
                                original_map: Dict[int, List[int]]) -> Dict[int, List[int]]:
        """Mutate quantum entanglement connections"""
        new_map = original_map.copy()
        layer_to_mutate = random.choice(list(new_map.keys()))
        new_map[layer_to_mutate] = self._generate_entanglement_pairs(
            len(new_map[layer_to_mutate])
        )
        return new_map