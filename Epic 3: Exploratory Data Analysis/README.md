# Epic 3: Exploratory Data Analysis (EDA)

## Objective
To explore and understand the insurance fraud dataset using visualizations
and statistical summaries. This helps identify patterns, trends, anomalies,
and relationships between features before model building.

---

## Dataset
- Source: Insurance claims dataset (as provided in the project)
- Target Variable: Fraud (Fraud / Non-Fraud)

---

## Tasks Completed

### 1. Univariate Analysis
Performed analysis on individual features to understand their distributions.

**Visualizations:**
- Distribution of Fraud vs Non-Fraud cases  
- Histogram and boxplot of Claim Amount  
- Distribution of Age  
- Bar charts for categorical features (Gender, Policy Type, Claim Type)

**Insights:**
- Fraud cases are fewer compared to non-fraud cases (class imbalance)  
- Claim amount shows right-skewed distribution  
- Some categories appear more frequently than others  

---

### 2. Multivariate Analysis
Studied relationships between multiple variables.

**Visualizations:**
- Fraud vs Claim Amount  
- Fraud vs Age  
- Fraud vs Policy Type  
- Correlation heatmap of numerical features  

**Insights:**
- Higher claim amounts show relatively higher fraud tendency  
- Certain policy types have more fraud cases  
- Some numerical features show correlation  

---

## Preprocessing Notes
- Missing values handled  
- Outliers analyzed using boxplots  
- Categorical features prepared for encoding (in next epic)

---

## Deliverables
- EDA Notebook (`.ipynb`)  
- Visualizations (plots/images)  
- Summary of insights and observations  

---

## Conclusion
Exploratory Data Analysis provided useful insights into the structure of the
dataset and key factors influencing fraud. These findings will guide feature 
engineering and model building in the next epics.
