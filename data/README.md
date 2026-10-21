# Data

Raw data is not committed (file size and data ownership: Monash Smart
Infrastructure, provided for MMA3001 2026).

1. Download `5EnvSensor_MayToDec2024_180kRows.zip` from the MMA3001 Dataset 2 folder.
2. Unzip it so the file is at `data/raw/5EnvSensor_MayToDec2024_180kRows.csv`.
3. Run `notebooks/01_data_audit.ipynb` to create `data/processed/`.

Do not open or save the CSV in Excel: it rounds the 13-digit sensor IDs to 6.01E+12.