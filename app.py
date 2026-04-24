import gradio as gr
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import io
from PIL import Image
from cvae import CVAE
from bayesian_network import build_supply_chain_network, sample_black_swan_events
import numpy as np

# Initialize models
cvae = CVAE(input_dim=28, num_classes=3, hidden_dim=32, latent_dim=4)
# Try to load pre-trained weights if available
try:
    cvae.load_state_dict(torch.load('cvae_model.pth'))
except:
    pass # Will use random weights for demo if not trained
cvae.eval()

bn_model = build_supply_chain_network()

def run_simulation(geopolitics, social_media):
    """
    Takes manual user input for the macro-environment, uses the Bayesian Net 
    to figure out the resulting condition, and then uses the CVAE to draw it.
    """
    # 1. Map string inputs to integers
    geo_map = {"Stable": 0, "Unstable": 1}
    social_map = {"Quiet": 0, "Viral Trend": 1}
    
    geo_val = geo_map[geopolitics]
    soc_val = social_map[social_media]
    
    # 2. Query the Bayesian Network manually 
    # (Simplified for Gradio, we just look up our CPD table)
    # [Geo=Stable & Social=Quiet, Geo=Stable & Social=Viral, Geo=Unstable & Social=Quiet, Geo=Unstable & Social=Viral]
    column_idx = (geo_val * 2) + soc_val
    
    # These match the TabularCPD in bayesian_network.py
    probabilities = [
        [0.98, 0.10, 0.10, 0.05], # Normal
        [0.01, 0.00, 0.85, 0.50], # Port Closure
        [0.01, 0.90, 0.05, 0.45]  # Demand Spike
    ]
    
    p_normal = probabilities[0][column_idx]
    p_port = probabilities[1][column_idx]
    p_spike = probabilities[2][column_idx]
    
    # Roll the dice based on these probabilities
    final_condition = np.random.choice([0, 1, 2], p=[p_normal, p_port, p_spike])
    condition_names = {0: "Normal Operations", 1: "Port Closure Occurred", 2: "Demand Spike Occurred"}
    result_text = f"Bayesian Network Output:\n{condition_names[final_condition]}\n(Probabilities: Normal: {p_normal*100}%, Port: {p_port*100}%, Spike: {p_spike*100}%)"

    # 3. Generate the Chart with CVAE
    z = torch.randn(1, 4)
    c_one_hot = F.one_hot(torch.tensor([final_condition]), num_classes=3).float()
    
    with torch.no_grad():
        generated_sequence = cvae.decode(z, c_one_hot).numpy()[0]
        
    plt.figure(figsize=(10, 5))
    plt.plot(generated_sequence, color='purple' if final_condition==2 else 'red' if final_condition==1 else 'blue', linewidth=3)
    plt.title(f'Simulated 28-Day Demand: {condition_names[final_condition]}')
    plt.xlabel('Days')
    plt.ylabel('Normalized Demand Volume')
    plt.grid(True, alpha=0.3)
    
    # Save plot to PIL Image to send to Gradio
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    
    return result_text, img

# --- Gradio Interface Layout ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🦢 Supply Chain Black Swan Simulator")
    gr.Markdown("Adjust global macroeconomic factors and watch the Bayesian Network determine the likelihood of supply chain disruptions. The Conditional VAE then generates a synthetic 28-day demand curve for stress-testing downstream forecasting models.")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Global Environment")
            geo_input = gr.Radio(["Stable", "Unstable"], label="Geopolitics", value="Stable")
            social_input = gr.Radio(["Quiet", "Viral Trend"], label="Social Media", value="Quiet")
            sim_btn = gr.Button("Run Simulation", variant="primary")
            
            output_text = gr.Textbox(label="Probabilistic Outcome", lines=3)
            
        with gr.Column():
            gr.Markdown("### CVAE Generated Demand Curve")
            output_img = gr.Image(label="28-Day Demand Forecast")
            
    sim_btn.click(fn=run_simulation, inputs=[geo_input, social_input], outputs=[output_text, output_img])

if __name__ == "__main__":
    demo.launch()
