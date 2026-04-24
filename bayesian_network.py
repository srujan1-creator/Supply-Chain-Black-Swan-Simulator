import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling

def build_supply_chain_network():
    """
    Builds a Probabilistic Graphical Model (Bayesian Network) representing 
    cascading real-world events that lead to supply chain disruptions.
    """
    # 1. Define the network structure (Nodes and Edges)
    # The edges represent causality:
    # Geopolitics influences the Final Condition (e.g. causes Port Closures)
    # Social Media influences the Final Condition (e.g. causes Demand Spikes)
    model = BayesianNetwork([
        ('Geopolitics', 'Final_Condition'),
        ('Social_Media', 'Final_Condition')
    ])

    # 2. Define Conditional Probability Distributions (CPDs)
    
    # Root Node: Geopolitics
    # States: [0: Stable, 1: Unstable]
    # Let's say there's a 95% chance the world is stable, 5% chance of instability
    cpd_geopolitics = TabularCPD(variable='Geopolitics', variable_card=2,
                                 values=[[0.95], [0.05]])

    # Root Node: Social Media Trend
    # States: [0: Quiet, 1: Viral Trend]
    # Let's say there's a 90% chance things are quiet, 10% chance a product goes viral
    cpd_social_media = TabularCPD(variable='Social_Media', variable_card=2,
                                  values=[[0.90], [0.10]])

    # Child Node: Final Condition
    # This is the exact condition we feed into our CVAE!
    # States: [0: Normal, 1: Port Closure, 2: Demand Spike]
    # Since it has two parents (Geopolitics and Social Media), we need to define 
    # probabilities for every possible combination of parent states (2x2 = 4 columns).
    
    # Columns represent:
    # [Geo=Stable & Social=Quiet, Geo=Stable & Social=Viral, 
    #  Geo=Unstable & Social=Quiet, Geo=Unstable & Social=Viral]
    
    cpd_final_condition = TabularCPD(
        variable='Final_Condition', 
        variable_card=3, # 3 output states
        values=[
            # State 0: Normal
            [0.98, 0.10, 0.10, 0.05],
            # State 1: Port Closure
            [0.01, 0.00, 0.85, 0.50],
            # State 2: Demand Spike
            [0.01, 0.90, 0.05, 0.45]
        ],
        evidence=['Geopolitics', 'Social_Media'],
        evidence_card=[2, 2] # Two states for Geo, two states for Social
    )

    # 3. Add CPDs to the model and validate
    model.add_cpds(cpd_geopolitics, cpd_social_media, cpd_final_condition)
    assert model.check_model() == True, "Error: Network structure or probabilities are invalid!"
    
    return model

def sample_black_swan_events(model, num_samples=10):
    """
    Simulates the world for a given number of 'days' or 'events',
    returning the resulting conditions to feed into our CVAE.
    """
    print(f"Simulating {num_samples} global scenarios...")
    
    # Create a sampler
    sampler = BayesianModelSampling(model)
    
    # Forward sample through the network
    samples = sampler.forward_sample(size=num_samples)
    
    print("\nSimulation Results:")
    print(samples)
    
    # Extract just the final conditions to feed the CVAE
    # 0=Normal, 1=Port Closure, 2=Demand Spike
    conditions = samples['Final_Condition'].values
    
    return conditions

if __name__ == "__main__":
    network_model = build_supply_chain_network()
    
    # Let's simulate 15 distinct global scenarios
    generated_conditions = sample_black_swan_events(network_model, num_samples=15)
    
    print("\nFinal Condition array for CVAE:")
    print(generated_conditions)
    
    # Quick count statistics
    print(f"\nCondition Breakdown:")
    print(f"Normal (0): {np.sum(generated_conditions == 0)}")
    print(f"Port Closure (1): {np.sum(generated_conditions == 1)}")
    print(f"Demand Spike (2): {np.sum(generated_conditions == 2)}")
