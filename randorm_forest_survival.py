#### 1 - Importing packages
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from pysurvival.models.simulations import SimulationModel
from pysurvival.models.survival_forest import RandomSurvivalForestModel
from pysurvival.utils.metrics import concordance_index
from pysurvival.utils.display import integrated_brier_score

#### 2 - Generating the dataset from a Exponential parametric model
# Initializing the simulation model
sim = SimulationModel( survival_distribution = 'exponential',
                       risk_type = 'linear',
                       censored_parameter = 1,
                       alpha = 3)

# Generating N random samples 
N = 1000
dataset = sim.generate_data(num_samples = N, num_features=4)

# Showing a few data-points 
print(dataset.head(2))