import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Dataset parameters
n_samples = 5000
output_path = r"C:\Users\user\Documents\01 machine learning\usa_housing_prices_2018_regression.csv"

# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

print("Generating realistic USA housing dataset (2018 baseline)...")
print(f"Target: {n_samples} samples for supervised regression")
print()

# Generate features starting from 2018 baseline
base_year = 2018
data = {
    # Core housing features
    'square_footage': np.random.normal(2100, 600, n_samples).clip(800, 6000),
    'bedrooms': np.random.choice([2, 3, 4, 5, 6], n_samples, p=[0.15, 0.35, 0.35, 0.12, 0.03]),
    'bathrooms': np.random.choice([1, 2, 3, 4], n_samples, p=[0.1, 0.45, 0.4, 0.05]),
    'lot_size_acres': np.random.exponential(0.5, n_samples).clip(0.1, 5.0),
    
    # Location and quality factors
    'location_score': np.random.normal(5.5, 2.0, n_samples).clip(1, 10),  # 1=poor, 10=prime
    'zip_code': np.random.choice([90210, 10001, 60601, 33101, 77001, 94101, 30301, 19101], n_samples),
    
    # Temporal features (2018 market)
    'year_built': np.random.normal(1995, 25, n_samples).clip(1920, 2018).astype(int),
    'listing_date': pd.date_range(start=f'2018-01-01', periods=n_samples, freq='D')[:n_samples],
    
    # Amenities (binary flags)
    'has_garage': np.random.choice([0, 1], n_samples, p=[0.25, 0.75]),
    'has_pool': np.random.choice([0, 1], n_samples, p=[0.15, 0.85]),
    'has_basement': np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
}

df = pd.DataFrame(data)

# Generate realistic 2018 USA housing prices (regression target)
# Formula reflects real market dynamics: size + location + quality + age depreciation + amenities
df['price'] = (
    120000 +  # 2018 USA median baseline
    df['square_footage'] * 140 +  # $140/sqft avg
    df['bedrooms'] * 25000 +
    df['bathrooms'] * 35000 +
    df['location_score'] * 18000 +
    df['lot_size_acres'] * 50000 +
    (2018 - df['year_built']) * -800 +  # Age penalty
    df['has_garage'] * 22000 +
    df['has_pool'] * 35000 +
    df['has_basement'] * 15000 +
    np.random.normal(0, 35000, n_samples)  # Market noise
)

# Ensure realistic price bounds ($100K - $2.5M)
df['price'] = df['price'].clip(100000, 2500000).round(-2)  # Round to nearest $100

# Add engineered feature: age at listing (2018)
df['age_years'] = 2018 - df['year_built']

print("Dataset summary:")
print(f"- Shape: {df.shape}")
print(f"- Price range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
print(f"- Median price: ${df['price'].median():,.0f}")
print(f"- Avg sqft: {df['square_footage'].mean():.0f}")
print("- Ready for regression: continuous price target with 11 input features")
print()

# Save as CSV (faster/better for ML than Excel)
df.to_csv(output_path, index=False)
print(f"✅ SAVED: {output_path}")
print("\nDataset ready for supervised learning!")
print("Load with: pd.read_csv(r'C:\\Users\\user\\Documents\\01 machine learning\\usa_housing_prices_2018_regression.csv')")
