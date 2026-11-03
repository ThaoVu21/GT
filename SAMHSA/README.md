# Client-Profile Explorer (SAMHSA MH-CLD 2024)

Academic project — CSE 6242: Data and Visual Analytics

An interactive Dash dashboard that groups 6,889,385 mental-health clients from the
2024 SAMHSA Mental Health Client-Level Data into 7 client profiles and compares
observed vs. expected inpatient use (O/E) across states.

- **RQ1 – Who are the clients?** K-Means profiles built from age, SMI/SED status,
  number of diagnoses, substance use and 13 diagnosis flags.
- **RQ4 – Why do states differ?** State × profile O/E heatmap, a network of states
  grouped by the profile that drives their gap, and side-by-side state comparisons.

## Files

| File | Purpose |
| --- | --- |
| `profile_explorer_dash.py` | The dashboard (reads the summary tables in `outputs/`) |
| `profile_pipeline.py` | Builds the profiles and O/E tables from the raw MH-CLD CSV |
| `outputs/profile_explorer/*.csv` | Pre-computed summary tables used by the dashboard |
| `requirements.txt` | Packages for the dashboard |
| `requirements-pipeline.txt` | Extra packages for rerunning the pipeline |
| `render.yaml` | Render deployment settings |

## Run the dashboard

```bash
pip install -r requirements.txt
python profile_explorer_dash.py
```

Then open http://localhost:8050.

## Rebuild the tables (optional)

The raw file `mhcld_puf_2024.csv` is not included (about 7M rows). Download it from
SAMHSA, place it in this folder, then:

```bash
pip install -r requirements-pipeline.txt
python profile_pipeline.py
```
