# 🦢 Supply Chain Black Swan Simulator

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange)
![pgmpy](https://img.shields.io/badge/pgmpy-Bayesian%20Networks-green)

A generative AI framework designed to stress-test downstream supply chain forecasting models against rare, high-impact "Black Swan" events. This project was built to demonstrate advanced ML/AI capabilities for supply chain optimization.

## 🎯 Architecture Diagram & Workflow

The system consists of three main components:

1. **The Probabilistic Engine (Bayesian Network):** Uses `pgmpy` to model the causal relationships between macro-environmental factors (e.g., Geopolitical Instability, Viral Social Media Trends) and local supply chain disruptions (e.g., Port Closures, Demand Spikes).
2. **The Generative Engine (Conditional VAE):** A PyTorch-based CVAE trained on historical retail demand data (inspired by the Kaggle M5 dataset). It generates highly realistic 28-day synthetic demand curves conditioned on the specific disruptions outputted by the Bayesian Network.
3. **The Stress-Test Harness (LSTM):** A standard baseline LSTM demand forecaster trained *only* on normal operational data. We evaluate its performance degradation (MAPE, RMSE) when hit with the CVAE's synthetic Black Swan events.

## 🚀 Quick Start

Ensure you have the required packages:
```bash
pip install torch pandas numpy matplotlib pgmpy gradio
```

**Run the Gradio Web Demo:**
Interact with the Bayesian Network and CVAE in real-time through a web interface.
```bash
python app.py
```

## 📁 File Structure

* `data_loader.py` - Synthetic M5 dataset generator with condition labeling.
* `vae_base.py` & `train_vae.py` - Base Variational Autoencoder.
* `cvae.py` & `train_cvae.py` - Conditional VAE for scenario-specific generation.
* `bayesian_network.py` - Causal modeling of global disruptions.
* `lstm_forecaster.py` - Downstream baseline forecasting model.
* `evaluate.py` - Harness measuring LSTM degradation under stress scenarios.

## 📊 Why this matters

Traditional demand forecasting models (like LSTMs, Prophet, or ARIMA) fail catastrophically during unprecedented events because they lack historical training data for those specific shocks. By utilizing a Generative CVAE paired with causal Bayesian Networks, we can create infinite, statistically sound "synthetic histories" of rare events, allowing us to robustify our downstream models *before* a real crisis hits.
