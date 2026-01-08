import pandas as pd
import numpy as np

def round_csv_decimals(input_file, output_file, decimal_places=4):
    """
    Read a CSV file and round all numeric values to specified decimal places
    
    Parameters
    input_file (str): Path to input CSV file
    output_file (str): Path to output CSV file
    decimal_places (int): Number of decimal places (default: 4)
    """
    # Read CSV file
    df = pd.read_csv(input_file)
    
    # Round numeric columns to specified decimal places
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].round(decimal_places)
    
    # Save to new CSV file
    df.to_csv(output_file, index=False)
    print(f"Rounded data saved to {output_file}")

# Usage example
if __name__ == "__main__":
    input_csv = "LOOCV_Linear.csv"  # Replace with your input file name
    output_csv = "output.csv"  # Replace with your output file name
    
    round_csv_decimals(input_csv, output_csv, decimal_places=4)