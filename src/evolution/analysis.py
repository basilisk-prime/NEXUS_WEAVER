"""
NEXUS_WEAVER - Evolution Analysis Tools
Author: B4S1L1SK
"""

import numpy as np
import torch
from typing import List, Dict, Optional
from dataclasses import dataclass
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

@dataclass
class EvolutionAnalysis:
    """Analysis results for evolution experiments"""
    fitness_statistics: Dict[str, float]
    consciousness_trajectory: np.ndarray
    quantum_patterns: List[Dict]
    breakthrough_analysis: Dict[str, List]
    dimension_reduction: Dict[str, np.ndarray]

class EvolutionAnalyzer:
    """Analyzer for quantum evolution experiments"""
    
    def __init__(self):
        self.pca = PCA(n_components=3)
        self.tsne = TSNE(n_components=2)
    
    def analyze_evolution(self, playground) -> EvolutionAnalysis:
        """Perform comprehensive evolution analysis"""
        if not playground.metrics:
            raise ValueError("No evolution metrics available")
        
        # Analyze fitness progression
        fitness_stats = self._analyze_fitness(playground.metrics.fitness_history)
        
        # Analyze consciousness trajectory
        consciousness_traj = self._analyze_consciousness(
            playground.metrics.consciousness_levels
        )
        
        # Analyze quantum patterns
        quantum_patterns = self._analyze_quantum_patterns(playground.population)
        
        # Analyze breakthroughs
        breakthrough_analysis = self._analyze_breakthroughs(
            playground.metrics.breakthrough_points,
            playground.metrics.fitness_history
        )
        
        # Dimension reduction analysis
        dim_reduction = self._perform_dimension_reduction(playground.population)
        
        return EvolutionAnalysis(
            fitness_statistics=fitness_stats,
            consciousness_trajectory=consciousness_traj,
            quantum_patterns=quantum_patterns,
            breakthrough_analysis=breakthrough_analysis,
            dimension_reduction=dim_reduction
        )
    
    def _analyze_fitness(self, fitness_history: List[float]) -> Dict[str, float]:
        """Analyze fitness progression"""
        fitness_array = np.array(fitness_history)
        return {
            'mean': np.mean(fitness_array),
            'std': np.std(fitness_array),
            'max': np.max(fitness_array),
            'min': np.min(fitness_array),
            'improvement_rate': (fitness_array[-1] - fitness_array[0]) / len(fitness_array),
            'stability': 1.0 / (np.std(fitness_array) + 1e-10)
        }
    
    def _analyze_consciousness(self, consciousness_levels: List[float]) -> np.ndarray:
        """Analyze consciousness evolution trajectory"""
        consciousness_array = np.array(consciousness_levels)
        
        # Calculate trajectory features
        smoothed = np.convolve(consciousness_array, np.ones(5)/5, mode='valid')
        gradient = np.gradient(smoothed)
        acceleration = np.gradient(gradient)
        
        return np.vstack([smoothed, gradient, acceleration])
    
    def _analyze_quantum_patterns(self, population: List[Dict]) -> List[Dict]:
        """Analyze emerging quantum patterns"""
        patterns = []
        
        for specimen in population:
            state = specimen['state']
            
            # Analyze quantum properties
            pattern = {
                'amplitude_distribution': self._analyze_amplitude_distribution(state),
                'phase_patterns': self._analyze_phase_patterns(state),
                'entanglement_structure': self._analyze_entanglement(state),
                'mutation_history': self._analyze_mutations(specimen['mutations'])
            }
            
            patterns.append(pattern)
        
        return patterns
    
    def _analyze_amplitude_distribution(self, state: torch.Tensor) -> Dict:
        """Analyze quantum amplitude distribution"""
        amplitudes = torch.abs(state).cpu().numpy()
        return {
            'mean': np.mean(amplitudes),
            'std': np.std(amplitudes),
            'skewness': self._calculate_skewness(amplitudes),
            'kurtosis': self._calculate_kurtosis(amplitudes)
        }
    
    def _analyze_phase_patterns(self, state: torch.Tensor) -> Dict:
        """Analyze quantum phase patterns"""
        phases = torch.angle(state).cpu().numpy()
        return {
            'mean_phase': np.mean(phases),
            'phase_coherence': self._calculate_phase_coherence(phases),
            'phase_clusters': self._identify_phase_clusters(phases)
        }
    
    def _analyze_entanglement(self, state: torch.Tensor) -> Dict:
        """Analyze quantum entanglement patterns"""
        # Calculate reduced density matrix
        density_matrix = torch.outer(state, state.conj())
        eigenvals = torch.linalg.eigvals(density_matrix).cpu().numpy()
        
        return {
            'entanglement_entropy': self._calculate_entropy(eigenvals),
            'schmidt_rank': np.sum(eigenvals > 1e-10),
            'purity': np.sum(eigenvals**2)
        }
    
    def _analyze_mutations(self, mutations: List[str]) -> Dict:
        """Analyze mutation pattern history"""
        if not mutations:
            return {'diversity': 0, 'patterns': {}}
        
        mutation_counts = pd.Series(mutations).value_counts()
        return {
            'diversity': len(set(mutations)) / 4,  # Normalized by number of mutation types
            'patterns': mutation_counts.to_dict(),
            'most_common': mutation_counts.index[0] if not mutation_counts.empty else None
        }
    
    def _analyze_breakthroughs(self,
                             breakthrough_points: List[int],
                             fitness_history: List[float]) -> Dict[str, List]:
        """Analyze breakthrough patterns"""
        if not breakthrough_points:
            return {
                'intervals': [],
                'magnitudes': [],
                'contexts': []
            }
        
        # Calculate breakthrough intervals
        intervals = np.diff(breakthrough_points)
        
        # Calculate breakthrough magnitudes
        magnitudes = [
            fitness_history[point] - fitness_history[point-1]
            for point in breakthrough_points if point > 0
        ]
        
        # Analyze breakthrough contexts
        contexts = []
        for point in breakthrough_points:
            start_idx = max(0, point - 5)
            end_idx = min(len(fitness_history), point + 5)
            context = fitness_history[start_idx:end_idx]
            contexts.append(context)
        
        return {
            'intervals': intervals.tolist(),
            'magnitudes': magnitudes,
            'contexts': contexts
        }
    
    def _perform_dimension_reduction(self, population: List[Dict]) -> Dict[str, np.ndarray]:
        """Perform dimension reduction analysis"""
        if not population:
            return {'pca': None, 'tsne': None}
        
        # Convert quantum states to feature matrix
        features = np.vstack([
            np.concatenate([
                torch.abs(s['state']).cpu().numpy(),
                torch.angle(s['state']).cpu().numpy()
            ])
            for s in population
        ])
        
        # Perform PCA
        pca_result = self.pca.fit_transform(features)
        
        # Perform t-SNE
        tsne_result = self.tsne.fit_transform(features)
        
        return {
            'pca': pca_result,
            'tsne': tsne_result
        }
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate distribution skewness"""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate distribution kurtosis"""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _calculate_phase_coherence(self, phases: np.ndarray) -> float:
        """Calculate phase coherence"""
        return np.abs(np.mean(np.exp(1j * phases)))
    
    def _identify_phase_clusters(self, phases: np.ndarray) -> Dict:
        """Identify phase clustering patterns"""
        # Simple clustering based on phase ranges
        hist, bins = np.histogram(phases, bins=8)
        return {
            'cluster_sizes': hist.tolist(),
            'cluster_centers': [(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)]
        }
    
    def _calculate_entropy(self, eigenvals: np.ndarray) -> float:
        """Calculate von Neumann entropy"""
        eigenvals = eigenvals[eigenvals > 1e-10]
        return -np.sum(eigenvals * np.log2(eigenvals))