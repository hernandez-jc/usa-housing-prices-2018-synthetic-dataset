# Synthetic USA Housing Prices 2018 Dataset + Generator

**Programmatically generated synthetic housing dataset** (~5000 rows) simulating realistic 2018 U.S. real estate market dynamics. Created for regression practice, feature engineering demos, and privacy-safe ML prototyping.

**Important Transparency**  
- **100% synthetic / artificial** — no real transactions, no scraped data, no MLS/Zillow sourcing.  
- Designed to reflect plausible economic relationships (size, location, age depreciation, amenities) with controlled randomness.  
- Not intended for production forecasting — it's an exploratory / educational artifact.

**Dataset Columns**  
- `square_footage`, `bedrooms`, `bathrooms`, `lot_size_acres`  
- `location_score` (1–10 simulated desirability)  
- `zip_code` (sampled major metros)  
- `year_built`, `listing_date` (all 2018)  
- `has_garage`, `has_pool`, `has_basement` (binary)  
- `price` (target — generated via economic formula + noise)  
- `age_years` (engineered: 2018 − year_built)

**Generation Logic Highlights**  
- Realistic distributions: normal (sqft, location), exponential (lot), discrete choice (bed/bath)  
- Price formula: baseline + $140/sqft + bedroom/bath bonuses + location/lot premiums + age penalty − $800/year + amenity add-ons + N(0, $35k) noise  
- Clipped/bounded values for realism  
- Full reproducibility: `np.random.seed(42)`

**Skills Demonstrated**  
- Synthetic tabular data generation with economic realism  
- Feature engineering (age derivation, binary encoding)  
- Reproducible Python scripting (pandas, numpy, seed control)  
- Constraint-aware simulation (clipping, bounding, noise modeling)  
- Ready for regression: supervised target (price) + 11 input features

**Files**  
- `synthetic-usa-housing-prices-2018.py` — full generation script  
- `usa_housing_prices_2018_sample.csv` — 200-row excerpt (full 5000 rows generated on demand)  
- (optional) `quick_eda.ipynb` — basic exploration & regression baselines

**Uses**  
- Regression model practice (linear, tree-based, XGBoost)  
- Feature importance / ablation studies  
- Synthetic data prototyping when real estate data is restricted  
- Computational economics sensitivity analysis (e.g., impact of amenities vs age)

Open to discussions on synthetic data pipelines, regression annotation, or computational economics modeling.

**Keywords:** synthetic housing dataset, regression synthetic data, python data generation, house price simulation, computational economics dataset

Last updated: March 2026
