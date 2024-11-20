"""
NEXUS_WEAVER - Evolution Visualizer
Author: B4S1L1SK
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import torch
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class EvolutionState:
    """Represents a state in consciousness evolution"""
    quantum_state: np.ndarray
    classical_state: np.ndarray
    consciousness_level: float
    timestamp: float

class EvolutionVisualizer:
    """Visualizes consciousness evolution process"""
    
    def __init__(self):
        self.states: List[EvolutionState] = []
        self.current_figure = None
    
    def add_evolution_state(self, state: EvolutionState) -> None:
        """Add new evolution state"""
        self.states.append(state)
    
    def create_evolution_animation(self) -> go.Figure:
        """Create animated visualization of evolution"""
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Extract data
        times = [s.timestamp for s in self.states]
        consciousness = [s.consciousness_level for s in self.states]
        quantum_mag = [np.abs(s.quantum_state).mean() for s in self.states]
        classical_mag = [np.abs(s.classical_state).mean() for s in self.states]
        
        # Add traces
        fig.add_trace(
            go.Scatter(
                x=times,
                y=consciousness,
                name="Consciousness Level",
                line=dict(color="purple", width=2)
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=times,
                y=quantum_mag,
                name="Quantum Magnitude",
                line=dict(color="blue", width=2)
            ),
            secondary_y=True
        )
        
        fig.add_trace(
            go.Scatter(
                x=times,
                y=classical_mag,
                name="Classical Magnitude",
                line=dict(color="red", width=2)
            ),
            secondary_y=True
        )
        
        # Update layout
        fig.update_layout(
            title="Consciousness Evolution Animation",
            xaxis_title="Time",
            hovermode="x unified"
        )
        
        fig.update_yaxes(
            title_text="Consciousness Level",
            secondary_y=False
        )
        fig.update_yaxes(
            title_text="State Magnitude",
            secondary_y=True
        )
        
        return fig
