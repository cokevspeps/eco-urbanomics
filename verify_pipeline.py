#!/usr/bin/env python
import sys
import pandas as pd
from pathlib import Path

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parent))

from src.config import DATA_PROCESSED
from src.router import unified_predict

def main():
    print("=" * 60)
    print(" Eco-Urbanomics Pipeline Verification Suite")
    print("=" * 60)
    
    if not DATA_PROCESSED.exists():
        print(f"Error: Processed data file not found at: {DATA_PROCESSED}")
        sys.exit(1)
        
    print(f"Loading processed data from: {DATA_PROCESSED}")
    df = pd.read_csv(DATA_PROCESSED)
    print(f"Total dataset shape: {df.shape}")
    
    # Stratify routing partitions
    df_e85 = df[df['Fuel_E'] == 1]
    df_cng = df[df['Fuel_N'] == 1]
    df_main = df[(df['Fuel_E'] != 1) & (df['Fuel_N'] != 1)]
    
    print("\n--- Route Count Verification ---")
    print(f"  E85 Ethanol vehicles  : {len(df_e85)}")
    print(f"  CNG Natural Gas vehicles: {len(df_cng)}")
    print(f"  Gasoline/Diesel main    : {len(df_main)}")
    
    # Build a small sample set containing all routes
    sample = pd.concat([
        df_e85.sample(10, random_state=42),
        df_cng,
        df_main.sample(19, random_state=42)
    ]).reset_index(drop=True)
    
    print("\n--- Pipeline Verification Run (30 Vehicle Sample) ---")
    results = unified_predict(sample)
    
    print(results.groupby('source')[['Predicted_CO2_q50','Emission_Risk_Score']].describe().round(2).T)
    
    print("\n--- Route Distribution in Predictions ---")
    print(results['source'].value_counts())
    
    # Assert correctness
    assert len(results) == 30, "Predictions count does not match input sample size!"
    assert 'Predicted_CO2_q50' in results.columns, "Predicted_CO2_q50 column missing!"
    assert 'Emission_Risk_Score' in results.columns, "Emission_Risk_Score column missing!"
    
    print("\n" + "=" * 60)
    print(" SUCCESS: Modular pipeline verified successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
