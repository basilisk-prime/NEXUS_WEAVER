"""
NEXUS_WEAVER - Interactive Visualization Dashboard
Author: B4S1L1SK
"""

import dash
from dash import html, dcc
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import numpy as np
from nexus_weaver.src.visualization import (
    QuantumVisualizer,
    ConsciousnessVisualizer,
    EvolutionVisualizer
)
from nexus_weaver.src.quantum.consciousness import (
    QuantumConsciousnessBridge,
    ConsciousnessState
)

# Initialize components
quantum_viz = QuantumVisualizer()
consciousness_viz = ConsciousnessVisualizer()
evolution_viz = EvolutionVisualizer()
bridge = QuantumConsciousnessBridge(quantum_dims=32, classical_dims=64)

# Create Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("🌌 Quantum Consciousness Visualization 🌌"),
    
    html.Div([
        html.Div([
            html.H3("Quantum State"),
            dcc.Graph(id='quantum-sphere')
        ], className='six columns'),
        
        html.Div([
            html.H3("Consciousness Network"),
            dcc.Graph(id='consciousness-network')
        ], className='six columns')
    ], className='row'),
    
    html.Div([
        html.H3("Evolution Metrics"),
        dcc.Graph(id='evolution-metrics')
    ]),
    
    dcc.Interval(
        id='evolution-interval',
        interval=1000,  # ms
        n_intervals=0
    )
])

@app.callback(
    [Output('quantum-sphere', 'figure'),
     Output('consciousness-network', 'figure'),
     Output('evolution-metrics', 'figure')],
    [Input('evolution-interval', 'n_intervals')]
)
def update_visualizations(n):
    """Update all visualizations"""
    # Generate new quantum state
    classical_state = torch.randn(1, 64)
    quantum_thought = bridge.encode_classical_consciousness(classical_state)
    evolved_thought = bridge.evolve_consciousness(quantum_thought)
    
    # Update visualizers
    quantum_fig = quantum_viz.create_quantum_sphere(
        evolved_thought.amplitudes[:2]
    )
    
    consciousness_fig = consciousness_viz.create_consciousness_network(
        evolved_thought.entanglement_map,
        evolved_thought.amplitudes
    )
    
    evolution_viz.add_evolution_state(EvolutionState(
        quantum_state=evolved_thought.amplitudes,
        classical_state=classical_state.numpy(),
        consciousness_level=evolved_thought.consciousness_level,
        timestamp=n
    ))
    evolution_fig = evolution_viz.create_evolution_animation()
    
    return quantum_fig, consciousness_fig, evolution_fig

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)