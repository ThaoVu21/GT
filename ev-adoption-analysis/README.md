## Step A — Data Cleaning  
**Notebook:** `Thao_Data_Cleaning_Final.ipynb`

### Instructions

1. Place the required input files in the **same folder** as the notebook:
   - `alt_fuel_sations-all.csv`  (unzip first)
   - `FINAL_ev_prevalence.csv`  

2. In Jupyter:
   - **Kernel → Restart & Run All** (R kernel)

---

### Expected Outputs

- **Final dataset (used in analysis):**
  - `ev_income_master.csv`  
  *(Primary dataset passed to the analysis notebook)*  

  **Note:** ACS data is retrieved dynamically via API, so results may vary over time.  
  
- **Intermediate datasets (not used in final modeling):**
  - `acs_income_raw_2024.csv`  
  - `acs_income_2024.csv`  
  - `ev_charging_2024.csv`  

- **Validation checks:**
  - Confirms no duplicate `state_zip` keys  
  - Displays basic sanity checks  

---

## Step B — Analysis & Modeling  
**Notebook:** `Thao_Data_Analysis_Final.ipynb`

### Instructions

1. Ensure `ev_income_master.csv` is in the **same folder** as the notebook  
   *(or set the working directory using `setwd()`)*

2. In Jupyter:
   - **Kernel → Restart & Run All** (R kernel)

---

## Notes

- Always run **Step A before Step B** to ensure the latest cleaned data is used  
- Use **Restart & Run All** to avoid issues from cached variables or partial execution  
- Keep all files in the same directory to prevent path errors  
- The files `Thao_Data_Cleaning_Final.html` and `Thao_Data_Analysis_Final.html` are provided for reference.  
