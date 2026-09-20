# Car Price Prediction — Streamlit App

Interactive Streamlit app for the Car Price Prediction regression assignment.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The bundled `CarPrice_Assignment.xlsx` loads automatically. You can also upload
your own copy of the dataset from the sidebar.

## Features

- **Data Overview** — head/tail, shape, dtypes, missing values, duplicates, describe()
- **Preprocessing** — encoding steps and resulting feature matrix
- **EDA** — price distribution, horsepower/enginesize/curbweight vs price, correlation heatmap
- **Model & Evaluation** — Multiple Linear Regression with MAE, MSE, RMSE, R²
- **Predictions** — Actual vs Predicted plot, residual plots
- **Predict a New Car** — interactive form to price a custom car spec
- **Self Learning: Ridge & Lasso** — regularization comparison against plain OLS, with
  adjustable alpha sliders and Lasso feature-elimination inspection
