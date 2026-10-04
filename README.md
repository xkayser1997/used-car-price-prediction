# Used Car Price Prediction

## Overview

This project predicts used-car prices and examines which vehicle characteristics the selected model relies on most.

I compared linear regression, ridge regression, decision tree, and random forest models using five-fold cross-validation. Mean absolute error (MAE) was the primary model-selection metric. The selected model was then evaluated on a held-out test set.

## Dataset

Source: [Used Car Price Prediction Dataset by Taeef Najib on Kaggle](https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset)

The original dataset contains 4,009 vehicle listings and 12 columns, including:

- Brand and model
- Model year and mileage
- Fuel type and engine description
- Transmission
- Exterior and interior colors
- Accident history and clean-title information
- Price

## Tools

- Python
- pandas and NumPy
- scikit-learn
- Jupyter Notebook

## Workflow

### Data Inspection and Cleaning

- Inspected data types, missing values, duplicate rows, and categorical values.
- Converted price and mileage from strings to numeric values.
- Standardized equivalent transmission descriptions.
- Filled missing fuel types with Electric after inspecting the associated vehicles.
- Represented missing accident and title information as Unknown.
- Removed 45 rows with missing engine and fuel-type descriptions.
- Removed one extreme price considered a suspected data-entry error.
- Retained the remaining high-priced vehicles and examined performance separately by price band.

### Feature Engineering and Preprocessing

- Calculated vehicle age using 2024 as the reference year.
- Extracted horsepower and engine displacement from engine descriptions.
- Imputed missing horsepower using the brand mean and missing displacement using the brand mode.
- Learned brand-based imputation values within the pipeline to avoid using validation or test data.
- Used median imputation as a fallback when brand-specific values were unavailable.
- Standardized numerical features and one-hot encoded categorical features.

### Model Development

Reserved 20% of the data for final testing and used five-fold cross-validation on the training set.

The models evaluated were:

1. Linear regression
2. Ridge regression
3. Decision tree
4. Random forest

Hyperparameters were selected using cross-validation MAE. Log-transformed targets were also evaluated for ridge regression and the decision tree.

## Model Comparison

| Model | Cross-Validation MAE |
|---|---:|
| Linear Regression | $15,543.51 |
| Tuned Ridge Regression | $13,036.06 |
| Tuned Ridge Regression — Log Target | $9,307.17 |
| Tuned Decision Tree | $12,871.81 |
| Tuned Random Forest | $10,410.43 |

Ridge regression with a log-transformed target achieved the lowest overall cross-validation MAE and was selected for final evaluation.

It also achieved the lowest MAE in every price band below $100,000. Random forest performed slightly better in the $100,000+ band.

## Test Results

Only the selected model was evaluated on the held-out test set.

| Metric | Result |
|---|---:|
| MAE | $8,466.27 |
| RMSE | $25,752.15 |
| R² | 0.74 |

The model explained approximately 74% of the variation in test-set prices. The difference between MAE and RMSE indicates that some predictions had particularly large errors.

### Performance by Actual Price Band

| Price Band | Test Vehicles | MAE |
|---|---:|---:|
| Under $20,000 | 235 | $2,721.67 |
| $20,000–$40,000 | 263 | $4,832.62 |
| $40,000–$60,000 | 159 | $5,326.50 |
| $60,000–$100,000 | 94 | $11,270.38 |
| $100,000+ | 42 | $68,972.57 |

Performance was strongest for lower-priced vehicles. Errors increased substantially for expensive vehicles, showing why overall MAE alone does not fully describe the model's performance.

## Feature Importance

Permutation importance was calculated on the test set using MAE.

The features producing the largest increases in error when shuffled were:

- Vehicle age
- Horsepower
- Brand
- Mileage
- Engine displacement

These results describe the selected model's reliance on each feature. They do not establish causal relationships or indicate whether a feature increases or decreases price. Related features, such as engine description, horsepower, and displacement, can also complicate interpretation.

## Limitations

- Original listings were unavailable, limiting verification of suspicious prices and missing information.
- Brand-based imputation provides approximate engine characteristics that may not represent individual vehicles accurately.
- Expensive vehicles were relatively uncommon and had substantially higher prediction errors.
- Many model and engine categories had few observations, limiting what the model could learn about them.
- Results reflect a random split of this dataset and do not establish performance on future vehicle listings.
- The model is an exploratory price estimator rather than a validated appraisal tool.

## Running the Notebook

1. Download `used_cars.csv` from the linked Kaggle dataset.
2. Place it in the same folder as `used_car_price_prediction.ipynb`.
3. Install the required packages:

   ```bash
   pip install pandas numpy matplotlib scikit-learn jupyter
   ```

4. Open the notebook in Jupyter and run the cells in order.

## Conclusion

Log-transformed ridge regression achieved the best overall cross-validation performance among the models evaluated. Price-band analysis showed that its predictions were more accurate for lower-priced vehicles, while expensive vehicles remained a significant limitation.

This project demonstrates data cleaning, feature engineering, preprocessing pipelines, hyperparameter tuning, model selection, and evaluation of regression errors across different segments.
