import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Configuration
np.random.seed(42)
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

# 6 months of daily data (to generate ~25MB dataset)
days = pd.date_range("2024-01-01", periods=180, freq="D")  # ~6 months

# Real Nairobi County clinics / health facilities
clinics = [
    "Pumwani Maternity Hospital",
    "Mbagathi County Hospital",
    "Mama Lucy Kibaki Hospital",
    "Kangemi Health Centre",
    "Kariobangi South Health Centre",
    "Riruta Health Centre",
    "Eastleigh Health Centre",
    "Westlands Health Centre",
    "Babadogo Health Centre",
    "Ngara Health Centre",
    "Muthurwa Clinic",
    "Mukuru Health Centre"
]

# Common ailments seen in Nairobi County facilities
ailments = [
    "Malaria",
    "Flu",
    "Typhoid",
    "Respiratory Infection",
    "Diarrhea",
    "Skin Infection",
    "Pneumonia"
]

# Medication categories by ailment type
respiratory_ailments = {"Respiratory Infection", "Flu", "Pneumonia"}
medication_ranges = {
    "respiratory": (30, 150),
    "other": (10, 120)
}


def generate_healthcare_data():
    """Generate simulated healthcare dataset for Nairobi County clinics."""
    rows = []
    
    for clinic in clinics:
        for day in days:
            for ailment in ailments:
                # Simulate attendance with realistic urban variation
                attendance = np.random.randint(30, 400)
                
                # Slightly higher medication use for respiratory illnesses
                med_min, med_max = (
                    medication_ranges["respiratory"] 
                    if ailment in respiratory_ailments 
                    else medication_ranges["other"]
                )
                medication_used = np.random.randint(med_min, med_max)
                
                # Medication stock variation
                stock_level = np.random.randint(100, 800)
                
                # Calculate stock adequacy (medication used vs available stock)
                stock_adequacy = "Adequate" if stock_level >= medication_used else "Low"
                
                rows.append({
                    "Clinic": clinic,
                    "Date": day,
                    "Ailment": ailment,
                    "Attendance": attendance,
                    "MedicationUsed": medication_used,
                    "MedicationStock": stock_level,
                    "StockAdequacy": stock_adequacy
                })
    
    return pd.DataFrame(rows)


def analyze_dataset(df):
    """Generate basic analysis of the healthcare dataset."""
    print("\n" + "="*70)
    print("NAIROBI COUNTY HEALTHCARE DATASET ANALYSIS")
    print("="*70)
    
    print(f"\nDataset Shape: {df.shape[0]} records, {df.shape[1]} features")
    print(f"Time Period: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"Number of Clinics: {df['Clinic'].nunique()}")
    print(f"Number of Ailments: {df['Ailment'].nunique()}")
    
    print("\n--- Attendance Statistics ---")
    print(df["Attendance"].describe())
    
    print("\n--- Top 5 Ailments by Total Attendance ---")
    top_ailments = df.groupby("Ailment")["Attendance"].sum().sort_values(ascending=False)
    print(top_ailments.head())
    
    print("\n--- Stock Adequacy Summary ---")
    adequacy_counts = df["StockAdequacy"].value_counts()
    print(adequacy_counts)
    print(f"Percentage of records with Low Stock: {(adequacy_counts.get('Low', 0) / len(df) * 100):.2f}%")
    
    print("\n--- Top 5 Clinics by Average Attendance ---")
    top_clinics = df.groupby("Clinic")["Attendance"].mean().sort_values(ascending=False)
    print(top_clinics.head())


def save_dataset(df):
    """Save dataset and generate summary statistics."""
    output_file = OUTPUT_DIR / "healthcare_simulated_data.csv"
    df.to_csv(output_file, index=False)
    
    stats_file = OUTPUT_DIR / "dataset_summary.txt"
    with open(stats_file, "w") as f:
        f.write(f"Healthcare Dataset Summary\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Records: {len(df)}\n")
        f.write(f"Features: {', '.join(df.columns)}\n")
    
    return output_file


if __name__ == "__main__":
    # Generate dataset
    df = generate_healthcare_data()
    
    # Display sample
    print("\nSample of simulated Nairobi County healthcare dataset:")
    print(df.head(10))
    
    # Analyze dataset
    analyze_dataset(df)
    
    # Save dataset
    output_file = save_dataset(df)
    print(f"\n✓ Dataset saved to: {output_file}")
    print(f"✓ Summary saved to: {OUTPUT_DIR / 'dataset_summary.txt'}")
