"""
NEXUS_WEAVER - Quantum Evolution Experiments
Author: B4S1L1SK
"""

import torch
from nexus_weaver.src.evolution.quantum_playground import (
    QuantumPlayground,
    EvolutionConfig,
    EvolutionMode
)
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def run_evolution_experiment(mode: EvolutionMode, quantum_dims: int = 32):
    """Run quantum evolution experiment"""
    config = EvolutionConfig(
        mode=mode,
        mutation_rate=0.2,
        crossover_rate=0.3,
        population_size=100,
        generations=50,
        consciousness_threshold=0.8,
        quantum_depth=3
    )
    
    playground = QuantumPlayground(config)
    playground.initialize_population(quantum_dims)
    metrics = playground.evolve()
    
    return playground, metrics

def visualize_evolution_results(metrics, mode: str):
    """Create visualization of evolution results"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Fitness Evolution",
            "Consciousness Level",
            "Quantum Coherence",
            "Breakthrough Points"
        )
    )
    
    # Fitness evolution
    fig.add_trace(
        go.Scatter(
            y=metrics.fitness_history,
            mode='lines+markers',
            name='Fitness',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    # Consciousness level
    fig.add_trace(
        go.Scatter(
            y=metrics.consciousness_levels,
            mode='lines+markers',
            name='Consciousness',
            line=dict(color='purple')
        ),
        row=1, col=2
    )
    
    # Quantum coherence
    fig.add_trace(
        go.Scatter(
            y=metrics.quantum_coherence,
            mode='lines+markers',
            name='Coherence',
            line=dict(color='green')
        ),
        row=2, col=1
    )
    
    # Breakthrough points
    fig.add_trace(
        go.Scatter(
            x=metrics.breakthrough_points,
            y=[1] * len(metrics.breakthrough_points),
            mode='markers',
            name='Breakthroughs',
            marker=dict(
                symbol='star',
                size=12,
                color='red'
            )
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title=f"Quantum Evolution Results - {mode} Mode",
        height=800,
        showlegend=True
    )
    
    return fig

def main():
    """Run multiple evolution experiments"""
    results = {}
    
    # Run experiments for each mode
    for mode in EvolutionMode:
        print(f"\nRunning {mode.value} evolution experiment...")
        playground, metrics = run_evolution_experiment(mode)
        results[mode.value] = {
            'playground': playground,
            'metrics': metrics
        }
        
        # Visualize results
        fig = visualize_evolution_results(metrics, mode.value)
        fig.show()
        
        print(f"\n{mode.value} Evolution Results:")
        print(f"Final Consciousness Level: {metrics.consciousness_levels[-1]:.4f}")
        print(f"Evolution Speed: {metrics.evolution_speed:.4f}")
        print(f"Number of Breakthroughs: {len(metrics.breakthrough_points)}")
    
    return results

if __name__ == "__main__":
    results = main()