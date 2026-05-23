import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.config import TARGET_C, TARGET_R, ETHANOL_ENERGY_FACTOR

def parse_trans_type(t):
    """
    Returns 1 for Automatic, 0 for Manual.
    Any transmission code starting with 'M' is manual.
    """
    return 0 if str(t).startswith('M') else 1

def parse_num_gears(t):
    """
    Extracts gear count from transmission code (e.g. AS6 -> 6).
    Returns 0 for CVT or variable transmissions (e.g. AV).
    """
    nums = re.findall(r'\d+', str(t))
    return int(nums[0]) if nums else 0

def add_e85_features(df_subset):
    """
    Engineers energy-adjusted features for E85 ethanol vehicles
    to correct the fuel-volume-to-emissions bias.
    """
    df_out = df_subset.copy()
    
    # Feature A: Energy-adjusted fuel consumption (gasoline equivalent)
    df_out['FC_energy_adjusted'] = (
        df_out['Fuel Consumption Comb (L/100 km)'] * ETHANOL_ENERGY_FACTOR
    )
    
    # Feature B: CO2 per energy-adjusted litre
    df_out['CO2_per_energy_litre'] = (
        df_out['CO2 Emissions(g/km)'] /
        (df_out['FC_energy_adjusted'] + 1e-6)
    )
    
    # Feature C: Energy-adjusted efficiency score
    df_out['Energy_adj_efficiency'] = 1.0 / (df_out['FC_energy_adjusted'] + 1e-6)
    
    return df_out

def preprocess_raw_data(df_raw):
    """
    Processes the raw dataset to build targets, parse transmission columns,
    encode categoricals, engineer features, and scale the numeric features.
    
    Returns:
        df_processed (pd.DataFrame): The preprocessed, encoded, and scaled dataframe.
        feat_cols (list): List of feature column names.
        scale_cols (list): List of scaled continuous columns.
    """
    df = df_raw.copy()
    
    # 1. Define Target variables
    # Class threshold: 70th percentile (~278 g/km)
    THRESHOLD = 278
    df[TARGET_C] = (df['CO2 Emissions(g/km)'] >= THRESHOLD).astype(int)
    
    # 2. Drop Model (high cardinality)
    if 'Model' in df.columns:
        df = df.drop(columns=['Model'])
        
    # 3. Parse Transmission
    if 'Transmission' in df.columns:
        df['Trans_Type'] = df['Transmission'].apply(parse_trans_type)
        df['Num_Gears'] = df['Transmission'].apply(parse_num_gears)
        df = df.drop(columns=['Transmission'])
        
    # 4. Encode Fuel Type
    if 'Fuel Type' in df.columns:
        fuel_dummies = pd.get_dummies(df['Fuel Type'], prefix='Fuel', drop_first=True)
        # Ensure we have all E, N, X, Z dummy columns
        for col in ['Fuel_E', 'Fuel_N', 'Fuel_X', 'Fuel_Z']:
            if col not in fuel_dummies.columns:
                fuel_dummies[col] = False
        df = pd.concat([df, fuel_dummies], axis=1)
        df = df.drop(columns=['Fuel Type'])
        
    # 5. Label-encode Make and Vehicle Class
    le_make = LabelEncoder()
    le_vc = LabelEncoder()
    if 'Make' in df.columns:
        df['Make_Enc'] = le_make.fit_transform(df['Make'])
        df = df.drop(columns=['Make'])
    if 'Vehicle Class' in df.columns:
        df['Vehicle_Class_Enc'] = le_vc.fit_transform(df['Vehicle Class'])
        df = df.drop(columns=['Vehicle Class'])
        
    # 6. Engineer interaction features
    df['City_Hwy_Ratio'] = (
        df['Fuel Consumption City (L/100 km)'] /
        (df['Fuel Consumption Hwy (L/100 km)'] + 1e-6)
    )
    df['Displacement_per_Cyl'] = df['Engine Size(L)'] / (df['Cylinders'] + 1e-6)
    df['Fuel_Efficiency_Score'] = 1.0 / (df['Fuel Consumption Comb (L/100 km)'] + 1e-6)
    df['Engine_Load_Index'] = df['Engine Size(L)'] * df['Cylinders']
    df['Green_Score'] = df['Fuel Consumption Comb (mpg)'] * df['Fuel_Efficiency_Score']
    
    # 7. Assemble feature column list
    feat_cols = [c for c in df.columns if c not in [TARGET_C, TARGET_R]]
    
    # 8. Scale numeric columns
    scale_cols = [
        'Engine Size(L)', 'Cylinders',
        'Fuel Consumption City (L/100 km)', 'Fuel Consumption Hwy (L/100 km)',
        'Fuel Consumption Comb (L/100 km)', 'Fuel Consumption Comb (mpg)',
        'Num_Gears', 'City_Hwy_Ratio', 'Displacement_per_Cyl',
        'Fuel_Efficiency_Score', 'Engine_Load_Index', 'Green_Score'
    ]
    
    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])
    
    return df, feat_cols, scale_cols
