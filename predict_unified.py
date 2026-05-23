#!/usr/bin/env python
import argparse
import sys
import pandas as pd
from pathlib import Path

# Add project root to python path to ensure src can be imported
sys.path.append(str(Path(__file__).resolve().parent))

from src.router import unified_predict

def main():
    parser = argparse.ArgumentParser(
        description="Unified vehicle CO2 emissions and High Emitter prediction router CLI."
    )
    parser.add_argument(
        "--input", 
        type=str, 
        required=True, 
        help="Path to preprocessed input CSV file."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        required=True, 
        help="Path to save the resulting unified predictions CSV file."
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' does not exist.")
        sys.exit(1)
        
    print(f"Loading input data from: {input_path}")
    df_input = pd.read_csv(input_path)
    
    print("Running unified prediction router...")
    results = unified_predict(df_input)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"Predictions successfully written to: {output_path}")
    print(f"Output columns: {list(results.columns)}")
    print(f"Predictions count: {len(results)}")
    print(results.groupby('source')[['Predicted_CO2_q50', 'Emission_Risk_Score']].describe().round(2).T)

if __name__ == "__main__":
    main()
