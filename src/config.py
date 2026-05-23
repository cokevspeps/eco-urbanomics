import torch
from pathlib import Path

# Random Seed
SEED = 42

# Device Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Base Directory (Project Root)
BASE_DIR = Path(__file__).resolve().parent.parent

# File System Directories
DATA_RAW = BASE_DIR / 'data' / 'raw' / 'CO2_Emissions_Canada.csv'
DATA_PROCESSED = BASE_DIR / 'data' / 'processed' / 'processed_co2_data.csv'
MODELS_DIR = BASE_DIR / 'models'
OUTPUTS_DIR = BASE_DIR / 'outputs'

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Target Columns
TARGET_C = 'High_Emitter'
TARGET_R = 'CO2 Emissions(g/km)'

# E85 / Alternative Fuel specific constants
ETHANOL_ENERGY_FACTOR = 0.659
CNG_CO2_PER_FC = 15.3
