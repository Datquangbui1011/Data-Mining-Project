# Data Mining Project - H-1B Visa Petitions (2011-2016)

CSCE 474/874 | Group 6

## Group Members
- **Dat Bui** (Data Preprocessing Lead)
- **Komlan**
- **Thang**
- **Nick**

## Project Contributions

### Dat Bui
- Led the data preprocessing pipeline for the H-1B dataset (~3 million records).
- Addressed missing values (e.g., worksite, latitude, longitude) and removed invalid entries (non-positive wages).
- Filtered wage outliers using the Interquartile Range (IQR) method.
- Handled feature engineering: extracted states from worksites, grouped top 20 occupations, and bucketed wages for Apriori.
- Encoded categorical variables and scaled numeric variables using `MinMaxScaler` and `LabelEncoder`.
- Exported formatted datasets for downstream analysis:
  - `clean_full.csv` for Classification & Clustering
  - `clean_apriori.csv` for Association Rule Mining
- Created the project structure and configured Git ignore rules to prevent tracking large raw and cleaned dataset files.

## How to Run the Code

### Prerequisites
- Python 3.x (We recommend using a Conda environment)
- Required packages: `pandas`, `numpy`, `scikit-learn`

You can install the dependencies via:
```bash
pip install pandas numpy scikit-learn
```

### Setup
1. Download the `h1b_kaggle.csv` dataset.
2. Create a folder named `data/` in the root of the project directory.
3. Place the downloaded `h1b_kaggle.csv` inside the `data/` folder.

### Running Data Preprocessing
To clean the data and generate the output files, simply run:
```bash
python3 Preprocessing.py
```

This script will process the dataset and output the following files:
- `clean_full.csv`: Dataset optimized for classification, clustering, and outlier detection.
- `clean_apriori.csv`: Discretized dataset optimized for association rule mining.
- `preprocessing_report.txt`: A summary report detailing the cleaning steps and final dataset statistics.