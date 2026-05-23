import torch
import pandas as pd
import numpy as np
from src.config import (
    DEVICE, MODELS_DIR, DATA_PROCESSED, TARGET_R,
    ETHANOL_ENERGY_FACTOR, CNG_CO2_PER_FC
)
from src.models import MainMLP, E85MLP

def load_normalization_ranges():
    """
    Loads raw normalization min/max ranges for regression targets.
    Falls back to hardcoded dataset ranges if the processed CSV isn't found.
    """
    try:
        df = pd.read_csv(DATA_PROCESSED)
        
        # Main model route normalization (Gasoline/Diesel/Hybrid)
        mask_alt = (df['Fuel_E'] == 1) | (df['Fuel_N'] == 1)
        df_main = df[~mask_alt]
        yr_main = df_main[TARGET_R].values.astype(np.float32)
        yr_main_min, yr_main_max = yr_main.min(), yr_main.max()
        
        # E85 submodel route normalization
        df_e85 = df[df['Fuel_E'] == 1]
        yr_e85 = df_e85[TARGET_R].values.astype(np.float32)
        yr_e85_min, yr_e85_max = yr_e85.min(), yr_e85.max()
    except Exception:
        # Standard fallback values from full Canadian CO2 Dataset range
        yr_main_min, yr_main_max = 96.0, 522.0
        yr_e85_min, yr_e85_max = 128.0, 418.0
        
    return yr_main_min, yr_main_max, yr_e85_min, yr_e85_max


def predict_cng_row(row):
    """Rule-based CNG override using EPA emission factor."""
    fc = row.get('Fuel Consumption Comb (L/100 km)', 12.7)
    co2 = fc * CNG_CO2_PER_FC
    return {
        'source': 'CNG_rule',
        'Predicted_CO2_q50': round(co2, 1),
        'Predicted_High_Emitter': int(co2 >= 278),
        'Emission_Risk_Score': float(int(co2 >= 278))
    }


def unified_predict(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Routes each vehicle record to the correct model (Main Dual-Head MLP,
    E85 submodel, or CNG rule-based override) and returns a unified predictions DataFrame.
    """
    # Load normalization bounds
    yr_main_min, yr_main_max, yr_e85_min, yr_e85_max = load_normalization_ranges()
    
    # Initialize models
    # Detect input feature size dynamically by removing target columns if present
    target_cols = [TARGET_R, 'High_Emitter']
    feat_cols = [c for c in df_input.columns if c not in target_cols]
    
    # Instantiate PyTorch networks
    main_model = MainMLP(in_dim=len(feat_cols)).to(DEVICE)
    main_model_path = MODELS_DIR / 'carbon_predictor_nn.pth'
    if main_model_path.exists():
        main_model.load_state_dict(torch.load(main_model_path, map_location=DEVICE))
    main_model.eval()
    
    # E85 features include 2 additional engineered features
    e85_feat_cols = feat_cols + ['FC_energy_adjusted', 'Energy_adj_efficiency']
    e85_model = E85MLP(in_dim=len(e85_feat_cols)).to(DEVICE)
    e85_model_path = MODELS_DIR / 'e85_submodel_nn.pth'
    if e85_model_path.exists():
        e85_model.load_state_dict(torch.load(e85_model_path, map_location=DEVICE))
    e85_model.eval()
    
    results = []
    
    for _, row in df_input.iterrows():
        is_cng = row.get('Fuel_N', 0) == 1
        is_e85 = row.get('Fuel_E', 0) == 1
        
        if is_cng:
            pred = predict_cng_row(row)
            
        elif is_e85:
            # Prepare E85 specific energy adjusted features
            e85_row = row.copy()
            fc_comb = row.get('Fuel Consumption Comb (L/100 km)', 16.9)
            e85_row['FC_energy_adjusted'] = fc_comb * ETHANOL_ENERGY_FACTOR
            e85_row['Energy_adj_efficiency'] = 1.0 / (fc_comb * ETHANOL_ENERGY_FACTOR + 1e-6)
            
            x_t = torch.tensor(
                e85_row[e85_feat_cols].values.astype(np.float32)
            ).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                clf_logit, reg_out = e85_model(x_t)
            prob = torch.sigmoid(clf_logit).item()
            co2 = reg_out.item() * (yr_e85_max - yr_e85_min) + yr_e85_min
            
            pred = {
                'source': 'E85_submodel',
                'Predicted_CO2_q50': round(co2, 1),
                'Predicted_High_Emitter': int(prob >= 0.5),
                'Emission_Risk_Score': round(prob, 4)
            }
            
        else:
            # Standard main model route
            row_feats = row[feat_cols].values.astype(np.float32)
            x_t = torch.tensor(row_feats).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                clf_logit, reg_out = main_model(x_t)
            prob = torch.sigmoid(clf_logit).item()
            co2 = reg_out.item() * (yr_main_max - yr_main_min) + yr_main_min
            
            pred = {
                'source': 'main_model',
                'Predicted_CO2_q50': round(co2, 1),
                'Predicted_High_Emitter': int(prob >= 0.5),
                'Emission_Risk_Score': round(prob, 4)
            }
            
        # Append true value if present in the test dataframe
        pred['Actual_CO2_gkm'] = row.get(TARGET_R, np.nan)
        results.append(pred)
        
    return pd.DataFrame(results)
