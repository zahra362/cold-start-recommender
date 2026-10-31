# Mitigating the Cold-Start Problem in Movie Recommender Systems

Code behind the paper:

> Zokaei, Z., Shahrokhzadeh, B., Farahani, M., & Alizadeh, S. H. "Mitigation of the cold start problem in movie recommender systems using fuzzification of users' social network information." 10th International Conference on Artificial Intelligence and Robotics (QICAR), February 2024. [IEEE Xplore](https://ieeexplore.ieee.org/document/10496631/)

## Idea

New users have no rating history, so classic collaborative filtering cannot recommend for them (the cold-start problem). This project instead predicts a user's taste for a movie from their **social-network profile features** — no prior ratings needed.

## What the notebook does

`new_model_recommendation.ipynb` implements the **non-fuzzy baseline** of the paper's approach:

1. Loads movie ratings (`ratings.csv`) and Twitter-based user profiles (`user_profiles.csv`: friends count, account creation year, preferred posting hour, weekend-tweeter flag, and similar features; the raw Twitter handle column is dropped).
2. Keeps movies with more than 480 ratings and builds one dataset per movie.
3. Binarizes the target: rating > 7 → 1 (liked), otherwise 0.
4. Standardizes the profile features (`StandardScaler`).
5. Trains a **Decision Tree** and a **Random Forest** classifier per movie and compares them on accuracy, RMSE and weighted F1.
6. Exports per-user predictions and the metrics table (`nofuzzy_DT_pre.csv`, `nofuzzy_RF_pre.csv`, `nofuzzy_metrics_df.csv`).

In the paper, the proposed model adds **fuzzification of the social-network information** on top of this setup: the followers count, friends count and favorites count are converted into fuzzy numbers (Low / Medium / High) with triangular membership functions whose ranges come from the data percentiles (implemented with scikit-fuzzy). This notebook is the non-fuzzy baseline the fuzzy model is compared against.

## Results (from the paper)

Average performance across the evaluated movies, 70/30 train/test split:

| Dataset | Model | Accuracy | RMSE | F1 |
|---|---|---|---|---|
| Fuzzy | Random Forest | 0.895 | 0.324 | 0.870 |
| Fuzzy | Decision Tree | 0.842 | 0.397 | 0.860 |
| Non-fuzzy (baseline) | Random Forest | ~0.79 | ~0.45 | ~0.69 |

Fuzzifying the social-network features clearly improved accuracy over the non-fuzzy baseline (e.g., movie 160882: Random Forest 0.896 fuzzy vs 0.789 non-fuzzy).

## Data

The experiments use the **CSP Dataset** (Herce-Zelaya et al.), a multi-source dataset of Twitter user profiles, movie features and ratings built for cold-start research. The CSV files are not included in this repository; place `ratings.csv` and `user_profiles.csv` next to the notebook to run it.

## Tools

Python, pandas, scikit-learn, NumPy

## Author

Zahra Zokaei — [LinkedIn](https://www.linkedin.com/in/zahra-zokaei/)
