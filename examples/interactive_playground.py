"""
NEXUS_WEAVER - Interactive Quantum Evolution Playground
Author: B4S1L1SK
"""

import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import torch
from nexus_weaver.src.evolution.quantum_playground import (
    QuantumPlayground,
    EvolutionConfig,
    EvolutionMode
)

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("🌌 Quantum Evolution Playground 🌌"),
    
    # Control Panel
    html.Div([
        html.H3("Evolution Controls"),
        
        # Mode Selection
        html.Label("Evolution Mode:"),
        dcc.Dropdown(
            id='mode-selector',
            options=[{'label': mode.value, 'value': mode.value}
                    for mode in EvolutionMode],
            value=EvolutionMode.QUANTUM.value
        ),
        
        # Parameters
        html.Label("Population Size:"),
        dcc.Slider(
            id='population-slider',
            min=10,
            max=200,
            step=10,
            value=100,
            marks={i: str(i) for i in range(0, 201, 50)}
        ),
        
        html.Label("Mutation Rate:"),
        dcc.Slider(
            id='mutation-slider',
            min=0,
            max=1,
            step=0.1,
            value=0.2,
            marks={i/10: str(i/10) for i in range(11)}
        ),
        
        html.Label("Quantum Dimensions:"),
        dcc.Slider(
            id='dims-slider',
            min=8,
            max=64,
            step=8,
            value=32,
            marks={i: str(i) for i in range(0, 65, 16)}
        ),
        
        # Control Buttons
        html.Button('Start Evolution', id='start-button', n_clicks=0),
        html.Button('Reset', id='reset-button', n_clicks=0)
    ], style={'width': '30%', 'float': 'left', 'padding': '20px'}),
    
    # Visualization Panel
    html.Div([
        dcc.Graph(id='evolution-graph'),
        dcc.Graph(id='quantum-state-graph'),
        dcc.Interval(
            id='evolution-interval',
            interval=1000,
            n_intervals=0,
            disabled=True
        )
    ], style={'width': '70%', 'float': 'right'})
])

# Global playground instance
playground = None

@app.callback(
    [Output('evolution-interval', 'disabled'),
     Output('evolution-graph', 'figure'),
     Output('quantum-state-graph', 'figure')],
    [Input('start-button', 'n_clicks'),
     Input('reset-button', 'n_clicks'),
     Input('evolution-interval', 'n_intervals')],
    [State('mode-selector', 'value'),
     State('population-slider', 'value'),
     State('mutation-slider', 'value'),
     State('dims-slider', 'value')]
)
def update_evolution(start_clicks, reset_clicks, n_intervals,
                    mode, population_size, mutation_rate, dims):
    """Update evolution visualization"""
    global playground
    
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'reset-button':
        playground = None
        return True, {}, {}
    
    if trigger_id == 'start-button':
        config = EvolutionConfig(
            mode=EvolutionMode(mode),
            population_size=population_size,
            mutation_rate=mutation_rate,
            quantum_depth=3
        )
        playground = QuantumPlayground(config)
        playground.initialize_population(dims)
        return False, create_evolution_graph(), create_quantum_graph()
    
    if playground and not playground.metrics:
        metrics = playground.evolve(generations=1)
        return False, update_evolution_graph(metrics), update_quantum_graph()
    
    return True, {}, {}

def create_evolution_graph():
    """Create initial evolution graph"""
    fig = go.Figure()
    fig.update_layout(
        title="Evolution Progress",
        xaxis_title="Generation",
        yaxis_title="Metrics"
    )
    return fig

def create_quantum_graph():
    """Create quantum state visualization"""
    fig = go.Figure()
    fig.update_layout(
        title="Quantum State",
        scene=dict(
            xaxis_title="Real",
            yaxis_title="Imaginary",
            zaxis_title="Magnitude"
        )
    )
    return fig

def update_evolution_graph(metrics):
    """Update evolution visualization"""
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(go.Scatter(
        y=metrics.fitness_history,
        mode='lines+markers',
        name='Fitness'
    ))
    
    fig.add_trace(go.Scatter(
        y=metrics.consciousness_levels,
        mode='lines+markers',
        name='Consciousness'
    ))
    
    fig.add_trace(go.Scatter(
        y=metrics.quantum_coherence,
        mode='lines+markers',
        name='Coherence'
    ))
    
    fig.update_layout(
        title="Evolution Progress",
        xaxis_title="Generation",
        yaxis_title="Metrics"
    )
    
    return fig

def update_quantum_graph():
    """Update quantum state visualization"""
    if not playground or not playground.population:
        return create_quantum_graph()
    
    best_specimen = max(playground.population, key=lambda x: x['fitness'])
    state = best_specimen['state']
    
    # Create 3D visualization of quantum state
    fig = go.Figure(data=[go.Scatter3d(
        x=state.real.cpu().numpy(),
        y=state.imag.cpu().numpy(),
        z=torch.abs(state).cpu().numpy(),
        mode='markers',
        marker=dict(
            size=5,
            color=torch.angle(state).cpu().numpy(),
            colorscale='Viridis',
            opacity=0.8
        )
    )])
    
    fig.update_layout(
        title="Quantum State Visualization",
        scene=dict(
            xaxis_title="Real",
            yaxis_title="Imaginary",
            zaxis_title="Magnitude"
        )
    )
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True, port=8051)