# 🛵 Zomato Delivery Time Predictor

A machine learning web app that predicts food delivery time from order, delivery-partner, weather, traffic, and location data — built end-to-end from raw data exploration to a deployed, interactive Streamlit app.

## 🔗 Live Demo

> _Add your Streamlit Community Cloud link here after deploying, e.g._
> **[Try it live →](https://your-app-name.streamlit.app)**

## 📋 Overview

This project walks through a complete regression workflow: exploratory data analysis, missing-value imputation, feature engineering, outlier treatment, multicollinearity checks, encoding, scaling, model training, and statistical validation — then wraps the trained model in a usable, realistically-designed web app.

## 🧠 How It Works

**Preprocessing pipeline (see the notebook for full detail):**
- Missing-value imputation (KNN / simple imputation)
- Haversine distance calculated from restaurant and delivery coordinates
- Time features extracted from order date/timestamp (day of week, hour)
- Outlier capping via the IQR method on age, ratings, and distance
- Ordinal encoding for traffic density; binary encoding for festival day
- One-hot encoding (`drop_first=True`) for weather, order type, vehicle type, and city
- Multicollinearity checked with VIF before finalizing features
- 80/20 train-test split, features scaled with `StandardScaler` (fit on training data only)

**Model:** scikit-learn `LinearRegression`, cross-checked against a `statsmodels` OLS fit for coefficient significance and regression assumption diagnostics (residual plots, normality, homoscedasticity).

Live R², Adjusted R², RMSE, and MAE on the held-out test set are shown in the app's sidebar.

## 🖥️ App Features

- **Simple, realistic form** — only asks what a real customer would actually know: order type, city, and distance.
- **"Delivery conditions" panel (collapsed by default)** — weather, traffic, festival day, and delivery-partner details. In a live product these would come from a weather/maps API and partner assignment, not the customer, so they're defaulted sensibly here and left adjustable for testing.
- **Instant prediction** with a styled result card.
- **Live model performance metrics** in the sidebar, loaded directly from the trained model's evaluation — not hardcoded.

## 🗂️ Project Structure

```
zomato-delivery-time-predictor/
├── app.py                                              # Streamlit app
├── requirements.txt                                    # Python dependencies
├── zomato_delivery_time_prediction_linear_regression.ipynb   # full training notebook
├── Zomato Dataset.csv                                  # raw dataset
├── zomato_model.pkl                                    # trained LinearRegression model
├── zomato_scaler.pkl                                   # fitted StandardScaler
├── zomato_feature_columns.pkl                          # exact training column order
├── zomato_iqr_bounds.pkl                               # outlier-capping bounds
├── zomato_category_options.pkl                         # valid categories per field
├── zomato_metrics.pkl                                  # test-set performance metrics
├── zomato_logo.png                                     # app header logo
├── README.md
├── .gitignore
└── LICENSE
```

## ⚙️ Tech Stack

`Python` · `pandas` / `numpy` · `scikit-learn` · `statsmodels` · `Streamlit` · `joblib`

## 🚀 Getting Started (Run Locally)

```bash
git clone https://github.com/<your-username>/zomato-delivery-time-predictor.git
cd zomato-delivery-time-predictor
pip install -r requirements.txt
streamlit run app.py
```

The app expects all six `.pkl` files and `zomato_logo.png` to be in the same folder as `app.py`.

## 📊 Dataset

Order-level delivery records including delivery-partner age and rating, restaurant and delivery-location coordinates, order and pickup timestamps, weather conditions, road traffic density, vehicle type, festival-day flag, and city type, with actual delivery time as the target.

## 🧩 Design Decisions

The raw dataset is collected *retrospectively* — it logs delivery-partner details that, in a real order flow, wouldn't be known until *after* a partner is assigned. Rather than pretend a customer would supply that information, the app separates fields into what a customer genuinely knows (order type, city, distance) versus what a live system would supply automatically (weather, traffic, partner details, current time) — the latter are defaulted but left editable for testing and demonstration.

## 🔮 Future Improvements

- Compare against regularized models (Ridge / Lasso / ElasticNet) and tree-based baselines
- Cross-validation and hyperparameter tuning
- Feature importance / coefficient visualization in the app
- Replace manual weather/traffic inputs with live API calls
- Predicted-vs-actual chart on the test set

## 📸 Screenshots

> _Add a screenshot or two of the running app here once deployed._

## 📄 License

Released under the [MIT License](LICENSE).

## 🙋 Author

**Your Name** — add links to your GitHub, LinkedIn, or portfolio here.

---
*Portfolio project — not affiliated with, endorsed by, or sponsored by Zomato Media Pvt. Ltd.*
