# Physical_Validator (PV) 🛡️

**Defining the "Physical Safe Operating Area" (SOA) for Industrial AI.**

### 🚩 Overview
Industrial AI models often suffer from "Laboratory Bias"—performing perfectly in controlled environments but failing on the factory floor due to aging lights, motion blur, or sensor noise. 

**Physical_Validator** is a decoupled, high-precision auditing framework designed to stress-test computer vision models against real-world physical variables. It helps engineers identify the exact breaking points of their AI systems.

### 🏗️ Core Architecture
- **PhysCorruptor (Core):** A high-fidelity simulation engine that injects physical degradations (Lux, DPI, Motion Blur, Defocus, ISO Noise) into golden samples.
- **Audit Logger (Utils):** Manages high-precision metadata and generates standardized audit reports (CSV).
- **Inference Bridge:** A zero-risk, decoupled interface to integrate any external AI model without library conflicts.
- **Visualizer:** Generates professional robustness curves (Performance vs. Physical Stress) to visualize the "Cliff Effect."

### 🚀 Quick Start
1. **Prepare Inputs:** Place your high-quality samples in `data/input/`.
2. **Generate Stress Tests:** ```bash
   python main.py --mode deep_audit