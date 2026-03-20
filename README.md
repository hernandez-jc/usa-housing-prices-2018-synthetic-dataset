# usa-housing-prices-2018-synthetic-dataset
# USA Housing Prices 2018 – Synthetic Dataset Sample

**Synthetic housing dataset** (programmatically generated in Python) for regression practice and feature exploration. Created ~2018 as a self-contained demo of tabular data simulation — mimics realistic U.S. real estate patterns without using any actual proprietary or scraped records.

**Key Transparency**  
- This is **100% synthetic/artificial data** — not derived from real transactions, MLS listings, or public APIs.  
- Generated via Python (pandas, numpy) using statistical distributions, logical rules, and a price formula with added noise to simulate market variability.  
- Purpose: Demonstrate data generation, feature engineering, basic regression workflows, and EDA in a privacy-safe way.  
- No claims of real-world predictive power or production use — it's a learning/exploratory artifact.

**Dataset Structure** (CSV columns)  
- `square_footage` — Living area in sq ft (continuous)  
- `bedrooms` — Number of bedrooms (integer)  
- `bathrooms` — Number of bathrooms (integer/float)  
- `lot_size_acres` — Lot size in acres (continuous)  
- `location_score` — Simulated desirability score (higher = better area; continuous)  
- `zip_code` — U.S. ZIP code (integer; sampled from major metros)  
- `year_built` — Construction year (integer)  
- `listing_date` — Date listed (string; all 2018)  
- `has_garage` — 1/0 binary  
- `has_pool` — 1/0 binary  
- `has_basement` — 1/0 binary  
- `price` — Target house price in USD (continuous; derived formula + noise)  
- `age_years` — Derived: listing year minus year_built (integer)  

**Sample Stats (from excerpt)**  
- ~20–30 rows shown; full synthetic set ~5000 rows (sample only uploaded)  
- Price range: ~$500k–$950k (simulated mid-high market)  
- Strong expected correlations: price ↑ with sq ft, bedrooms, newer build, amenities, better location_score  

**Example Rows** (first few)  



**Skills Demonstrated** 🔧  
- Synthetic tabular data generation (numpy distributions, rule-based pricing logic, noise addition)  
- Python data manipulation (pandas for creation, derivation of `age_years`)  
- Feature engineering basics (age calculation, binary amenities)  
- Reproducible scripting — seed-controlled if applicable  
- EDA potential: correlations (price vs sq ft/location), scatter plots, distribution checks  

**Uses & Modern Relevance**  
- Practice regression models (linear, tree-based, XGBoost) on clean synthetic data  
- Test feature importance (e.g., sq ft vs amenities vs age)  
- Simulate price evolution scenarios (e.g., update with hypothetical 2025 data via script tweaks)  
- Transferable to computational economics: modeling asset pricing, sensitivity analysis, synthetic data for privacy-safe ML prototyping  

**Files in Repo**  
- `usa_housing_prices_2018_sample.csv` — 100–200 row excerpt (full synthetic set generated on demand)  
- `generate_housing_data.py` (if you have it; placeholder script showing logic)  
- Notebooks: basic EDA / regression demo (add if you create one)  

Open to vetted discussions: synthetic data pipelines, regression annotation, computational economics datasets. Contact via GitHub issues or LinkedIn.

**Keywords:** synthetic house price dataset, regression practice dataset, Python generated real estate data, house price prediction sample, feature engineering demo

Last updated: March 2026
