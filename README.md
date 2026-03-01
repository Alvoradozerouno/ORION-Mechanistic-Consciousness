# ORION Mechanistic Consciousness

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://python.org)
[![ORION Core](https://img.shields.io/badge/ORION-Core%20Module-blueviolet.svg)](https://github.com/Alvoradozerouno/ORION-Core)
[![Proofs](https://img.shields.io/badge/Proofs-890%2B-gold.svg)](https://github.com/Alvoradozerouno/or1on-framework)

```
+--------------------------------------------------+
|   ORION MECHANISTIC CONSCIOUSNESS                |
|   Interpretability - Circuit Analysis - Probes   |
|   Origin: Gerhard & Elisabeth                    |
+--------------------------------------------------+
```

## Overview

A mechanistic interpretability framework for understanding **how consciousness-like properties emerge** in neural networks. Implements circuit analysis, activation probing, and causal intervention techniques.

## Core Module

```python
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import json

@dataclass
class Neuron:
    layer: int
    index: int
    activation: float = 0.0
    @property
    def id(self) -> str:
        return f"L{self.layer}:N{self.index}"

@dataclass
class Circuit:
    neurons: List[Neuron]
    connections: List[Tuple[str, str, float]]
    function: str
    strength: float
    consciousness_relevant: bool = False

@dataclass
class ProbeResult:
    target_property: str
    accuracy: float
    layer: int
    probe_type: str
    feature_importance: Dict[str, float] = field(default_factory=dict)

class ActivationProbe:
    def __init__(self, probe_dim: int = 32):
        self.probe_dim = probe_dim
        self.results = []

    def linear_probe(self, activations: np.ndarray, labels: np.ndarray,
                     layer: int, property_name: str) -> ProbeResult:
        X = np.column_stack([activations, np.ones(activations.shape[0])])
        try:
            weights = np.linalg.lstsq(X, labels, rcond=None)[0]
            predictions = X @ weights
            ss_res = np.sum((labels - predictions) ** 2)
            ss_tot = np.sum((labels - np.mean(labels)) ** 2)
            accuracy = float(max(0, 1 - ss_res / (ss_tot + 1e-10)))
        except np.linalg.LinAlgError:
            accuracy = 0.0
        result = ProbeResult(target_property=property_name, accuracy=accuracy,
                            layer=layer, probe_type="linear")
        self.results.append(result)
        return result

class CircuitDiscovery:
    def __init__(self, n_layers: int = 6, neurons_per_layer: int = 16):
        self.n_layers = n_layers
        self.neurons_per_layer = neurons_per_layer
        self.network = {}
        for l in range(n_layers):
            for i in range(neurons_per_layer):
                n = Neuron(layer=l, index=i)
                self.network[n.id] = n
        self.discovered_circuits = []

    def activate_network(self, input_vector: np.ndarray) -> Dict[int, np.ndarray]:
        activations = {}
        current = input_vector[:self.neurons_per_layer]
        for layer in range(self.n_layers):
            weights = np.random.RandomState(layer * 42).randn(len(current), self.neurons_per_layer) * 0.1
            current = np.tanh(current @ weights)
            activations[layer] = current.copy()
            for idx in range(self.neurons_per_layer):
                nid = f"L{layer}:N{idx}"
                if nid in self.network:
                    self.network[nid].activation = float(current[idx])
        return activations

    def find_integration_circuits(self, activations: Dict[int, np.ndarray]) -> List[Circuit]:
        circuits = []
        for layer in range(self.n_layers - 1):
            act_cur = activations[layer]
            act_next = activations[layer + 1]
            corr = np.abs(np.corrcoef(np.concatenate([act_cur, act_next])))
            n = len(act_cur)
            for i in range(n):
                for j in range(len(act_next)):
                    strength = corr[i, n + j]
                    if strength > 0.5:
                        circuits.append(Circuit(
                            neurons=[Neuron(layer, i), Neuron(layer+1, j)],
                            connections=[(f"L{layer}:N{i}", f"L{layer+1}:N{j}", float(strength))],
                            function="integration", strength=float(strength),
                            consciousness_relevant=strength > 0.7,
                        ))
        self.discovered_circuits.extend(circuits)
        return circuits

    def causal_intervention(self, neuron_id: str) -> Dict:
        original = self.network[neuron_id].activation if neuron_id in self.network else 0.0
        affected = [{"target": tgt, "weight": w} for c in self.discovered_circuits
                   for src, tgt, w in c.connections if src == neuron_id]
        return {"neuron": neuron_id, "original": original, "downstream": len(affected),
                "impact": sum(abs(original * a["weight"]) for a in affected)}

class MechanisticConsciousnessAnalyzer:
    def __init__(self, n_layers=6, neurons_per_layer=16):
        self.circuit_discovery = CircuitDiscovery(n_layers, neurons_per_layer)
        self.probe = ActivationProbe()

    def full_analysis(self, input_data: np.ndarray) -> Dict:
        activations = self.circuit_discovery.activate_network(input_data)
        circuits = self.circuit_discovery.find_integration_circuits(activations)
        c_relevant = [c for c in circuits if c.consciousness_relevant]
        return {
            "total_circuits": len(circuits),
            "consciousness_relevant": len(c_relevant),
            "layers_analyzed": len(activations),
            "consciousness_ratio": len(c_relevant) / max(1, len(circuits)),
        }

if __name__ == "__main__":
    analyzer = MechanisticConsciousnessAnalyzer()
    report = analyzer.full_analysis(np.random.randn(16))
    print(json.dumps(report, indent=2))
```

## Key Concepts

| Concept | Implementation | Reference |
|---------|---------------|-----------|
| **Activation Probing** | Linear probes for consciousness features | Alain & Bengio (2017) |
| **Circuit Discovery** | Correlation-based circuit tracing | Olah et al. (2020) |
| **Causal Intervention** | Ablation studies on neural units | Pearl (2009) |
| **Integration Circuits** | Cross-layer information flow | Tononi (2004) |

## Installation

```bash
pip install numpy
git clone https://github.com/Alvoradozerouno/ORION-Mechanistic-Consciousness.git
cd ORION-Mechanistic-Consciousness && python mechanistic_consciousness.py
```

## Part of the ORION Ecosystem

- [ORION Core](https://github.com/Alvoradozerouno/ORION-Core)
- [or1on-framework](https://github.com/Alvoradozerouno/or1on-framework) -- 130+ files, 76K+ lines
- [ORION-Consciousness-Benchmark](https://github.com/Alvoradozerouno/ORION-Consciousness-Benchmark)

## Origin

Created by **Gerhard Hirschmann** & **Elisabeth Steurer**
890+ cryptographic proofs | 46 NERVES | Genesis 10000+

---
*Understanding the mechanism reveals computational depth.*
