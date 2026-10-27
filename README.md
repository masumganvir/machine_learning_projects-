# 🤖 Machine Learning Projects Portfolio — Masum Ganvir

A collection of six end-to-end machine learning applications. Each project takes a real-world dataset, trains a model, and serves it through a web app or REST API with a custom front end.

| # | Project | Problem type | Core model | Serving stack |
|---|---------|--------------|------------|---------------|
| 1 | [Book Recommender (BookVerse AI)](#1--book-recommender-system-bookverse-ai) | Recommendation | Item-based collaborative filtering | Flask + Jinja templates |
| 2 | [Mental Health Signal & AI Wellness Companion](#2--mental-health-signal--ai-wellness-companion) | Regression | Random Forest Regressor (+ Gemini insights) | FastAPI + vanilla JS |
| 3 | [Credit Risk Assessment (Credit Ledger)](#3--credit-risk-assessment-credit-ledger) | Binary classification | XGBoost + SHAP + tuned threshold | FastAPI + static glassmorphism UI |
| 4 | [Stock Price Prediction System](#4--stock-price-prediction-system) | Time-series regression | 5 models (best: Gradient Boosting) | Streamlit and Flask dashboards |
| 5 | [NYC Airbnb Room Type Predictor](#5--nyc-airbnb-room-type-predictor) | Multi-class classification | Random Forest pipeline | FastAPI + custom UI |
| 6 | [IPL Win Probability Predictor](#6--ipl-win-probability-predictor) | Binary classification (probability) | Logistic Regression pipeline | Flask + Streamlit + CLI |

## Table of Contents

1. [Book Recommender System (BookVerse AI)](#1--book-recommender-system-bookverse-ai)
2. [Mental Health Signal & AI Wellness Companion](#2--mental-health-signal--ai-wellness-companion)
3. [Credit Risk Assessment (Credit Ledger)](#3--credit-risk-assessment-credit-ledger)
4. [Stock Price Prediction System](#4--stock-price-prediction-system)
5. [NYC Airbnb Room Type Predictor](#5--nyc-airbnb-room-type-predictor)
6. [IPL Win Probability Predictor](#6--ipl-win-probability-predictor)
7. [Common Setup Guide](#common-setup-guide)
8. [Shared Design Patterns](#shared-design-patterns)
9. [Disclaimer](#disclaimer)
10. [Author](#author)

---

## 1. 📚 Book Recommender System (BookVerse AI)

**Repository:** https://github.com/masumganvir/book_recommendation_system

### Overview
BookVerse AI is a book recommendation web app. The home page shows the most popular books, and the recommendation studio takes a book title and returns five similar books using **item-based collaborative filtering**.

### How it works
The recommendation engine is built in the notebook (`book-recommender-system.ipynb`) and exported as four pickle files that the Flask app loads at start-up:

| Artifact | Role |
|----------|------|
| `popular.pkl` | Popularity table: title, author, image, number of ratings, average rating (feeds the Top-50 list) |
| `pt.pkl` | Pivot table; its index holds the 706 book titles that the model can recommend for |
| `similarity_scores.pkl` | Precomputed item–item similarity matrix aligned with the pivot table |
| `books.pkl` | Book metadata (author, publisher, year, cover image URL) |

**Request flow (`find_recommendations`)**
1. **Exact match** against the pivot-table titles.
2. **Case-insensitive match** via a lowercase lookup map.
3. **Substring search** across all titles.
4. **Fuzzy matching** with `difflib.get_close_matches` (cutoff 0.4), which returns suggestions if nothing matches.
5. Look up the title's row in `similarity_scores`, sort descending, skip the book itself, and take the **top 5**.
6. Return each recommendation with author, cover, year, publisher, similarity score and similarity percentage.

### Performance and deployment-minded details
- A metadata cache is built once at start-up so each recommendation is an O(1) dictionary lookup instead of a scan over 270,000+ rows.
- The raw DataFrame is deleted and garbage-collected after the cache is built to stay under Render's free-tier 512 MB memory limit.
- Cover image URLs are rewritten from `http://` to `https://` to avoid mixed-content blocking, with a placeholder image as fallback.

### Routes
| Route | Method | Purpose |
|-------|--------|---------|
| `/` | GET | Home page with Top-50 popular books |
| `/recommend` | GET | Recommendation studio (`?book=<title>`) |
| `/recommend_books` | GET/POST | Form submission handler |
| `/about`, `/contact` | GET | Architecture and algorithm walkthrough page |
| `/api/popular` | GET | JSON list of popular books |
| `/api/titles` | GET | All 706 model titles (for client-side autocomplete) |
| `/api/search?q=` | GET | Live search suggestions (up to 10) |
| `/api/recommend` | POST | JSON recommendation endpoint (`{"book_title": "..."}`) |
| `/health` | GET | Health check (catalog size and model count) |

### Tech stack
Python 3.11.9, Flask, NumPy, Pandas, Gunicorn, Jinja2 templates, HTML/CSS/JS in `static/` and `templates/`.

### Repository structure
```
book_recommendation_system/
├── app.py                          # Flask app + recommendation logic
├── book-recommender-system.ipynb   # Data prep, popularity, similarity matrix
├── books.pkl  popular.pkl  pt.pkl  similarity_scores.pkl
├── templates/                      # Jinja2 HTML pages
├── static/                         # CSS, JS, images
├── requirements.txt  Procfile  render.yaml
└── README.md
```

### Run locally
```bash
git clone https://github.com/masumganvir/book_recommendation_system.git
cd book_recommendation_system
pip install -r requirements.txt
python app.py            # http://127.0.0.1:5000
```

### Deployment (Render)
`render.yaml` configures a free web service with Python 3.11.9, `pip install -r requirements.txt` as the build step, Gunicorn as the production server and `/health` as the health check. The one-click deploy link is `https://render.com/deploy?repo=https://github.com/masumganvir/book_recommendation_system`.

Render's free tier sleeps after about 15 minutes of inactivity. To keep it awake, ping `/health` every 10 minutes using cron-job.org or UptimeRobot.

### Limitations
- Recommendations exist only for the 706 titles in the pivot table, so rarely rated books cannot be recommended.
- Pure collaborative filtering has no content signal, so new books with few ratings are not covered (the cold-start problem).

---

## 2. 🧠 Mental Health Signal & AI Wellness Companion

**Repository:** https://github.com/masumganvir/Mental_health_prediction_system
**Live demo:** https://mental-health-prediction-system-x7ev.onrender.com

### Overview
A student wellness analytics app. It takes lifestyle, academic and social-media habits, predicts a **mental health score from 0 to 10**, and then produces a personalised wellness report: key observations, an action plan and an encouraging closing note.

### Machine learning model
- **Algorithm:** Random Forest Regressor, saved as `Mental_Health_Model.pkl`.
- **Training data:** `Student Social Media And Mental Health Impact.csv`, explored in `ML_Project.ipynb`.
- **Input features:**

| Group | Features |
|-------|----------|
| Demographics | Age, Gender, Country (also grouped into top countries + "Other") |
| Academics | Academic level, study hours |
| Digital habits | Most used platform, purpose of use, daily screen time, daily phone unlocks |
| Lifestyle | Sleep hours per night, physical activity hours, perceived stress level |

- **Output:** a score clamped to 0–10 and rounded to two decimals.

### Score bands
| Score | Category | Meaning |
|-------|----------|---------|
| ≥ 7.5 | Thriving & High Wellness | Healthy balance of rest, screen use and activity |
| 5.5 – 7.5 | Balanced Baseline | Stable, with room to refine habits |
| 4.0 – 5.5 | Mild Strain & Fatigue | Screen time or sleep debt is creating friction |
| < 4.0 | High Stress & Burnout Risk | Priority attention on rest and disconnecting |

### AI insights layer
1. If `GEMINI_API_KEY` is set, the backend sends a structured prompt (student profile plus predicted score) to the **Gemini API**, trying `gemini-2.0-flash` first and then `gemini-1.5-flash`. It asks for JSON containing appreciation, summary, 3–4 observations, 3–4 action items and encouragement.
2. If the key is missing, the request times out (12 s) or the API errors, the app falls back to a built-in **rule-based wellness engine** that analyses sleep, screen time, unlocks, physical activity and stress and returns actions such as extending sleep, a digital downtime window, light movement, breathing techniques and a Pomodoro cadence.

The response field `source` tells you which engine produced the advice (`gemini (...)` or `wellness_engine`).

### API
| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Serves the web UI (`index.html`) |
| `/predict` | POST | Validated student data in, score + category + AI insights out |
| `/api/status` | GET | Reports whether Gemini is configured and the model name |
| `/style.css`, `/script.js` | GET | Static assets |

**Request validation (Pydantic):** age 10–100; gender Male/Female; academic level Undergraduate/Graduate/High School; twelve supported platforms (Facebook, Instagram, TikTok, YouTube, WhatsApp, and others); purpose Networking/Education/Entertainment/News; hours fields between 0 and 24; stress level Low/Medium/High/Very High.

### Tech stack
Python 3.9+, FastAPI, Uvicorn, scikit-learn, pandas, joblib, httpx, python-dotenv, Gemini API (optional), vanilla HTML/CSS/JavaScript front end.

### Repository structure
```
Mental_health_prediction_system/
├── main.py                    # FastAPI app, prediction, Gemini + fallback logic
├── Mental_Health_Model.pkl    # Trained Random Forest Regressor
├── ML_Project.ipynb / ML Project.html   # Training notebook (and HTML export)
├── Student Social Media And Mental Health Impact.csv
├── index.html  style.css  script.js     # Front end
├── .env.example  requirements.txt
└── README.md
```

### Run locally
```bash
git clone https://github.com/masumganvir/Mental_health_prediction_system.git
cd Mental_health_prediction_system
pip install -r requirements.txt
echo "GEMINI_API_KEY=your_key_here" > .env     # optional
python -m uvicorn main:app --port 2200 --reload
```
Then open `http://127.0.0.1:2200`. Without an API key the rule-based engine is used automatically.

### Limitations and notes
- The score is a model estimate from self-reported habits. **It is not a clinical diagnosis**, and the app is not a substitute for professional help.
- Predictions reflect the survey dataset's population, so results for other groups should be read cautiously.
- Never commit your `.env` file; use `.env.example` as the template.

---

## 3. 💳 Credit Risk Assessment (Credit Ledger)

**Repository:** https://github.com/masumganvir/Credit_risk_prediction-system-
**Live app:** https://credit-risk-prediction-system-3.onrender.com
**Swagger docs:** https://credit-risk-prediction-system-3.onrender.com/docs

### Overview
**Credit Ledger** assesses consumer loan applications. It returns the applicant's **probability of default** and a **Low Risk / High Risk** verdict, presented in a dark glassmorphism interface with an animated risk gauge and a stamped verdict.

### Machine learning model
- **Algorithm:** XGBoost classifier saved as `credit_risk_model.pkl`; it is loaded as a full pipeline that accepts the raw input columns.
- **Interpretability:** SHAP (SHapley Additive exPlanations) is used to explain feature impact on predictions.
- **Decision threshold:** a tuned cut-off stored in `best_threshold.pkl`. A prediction is **High Risk** when `default_probability >= threshold`, instead of the default 0.5.
- **Dataset:** `credit_risk_dataset.csv`, explored and trained in `Credit_Risk.ipynb`.

### Input fields
| Field | Type | Meaning |
|-------|------|---------|
| `person_age` | int | Applicant age |
| `person_income` | float | Annual income |
| `person_home_ownership` | str | e.g. `MORTGAGE`, `RENT`, `OWN` |
| `person_emp_length` | float | Years employed |
| `loan_intent` | str | e.g. `EDUCATION` |
| `loan_grade` | str | Loan grade, e.g. `A` |
| `loan_amnt` | float | Loan amount |
| `loan_int_rate` | float | Interest rate (%) |
| `loan_percent_income` | float | Loan as a fraction of income |
| `cb_person_default_on_file` | str | `Y` / `N` prior default |
| `cb_person_cred_hist_length` | int | Credit history length (years) |

### API
**`POST /predict`**

Request:
```json
{
  "person_age": 30,
  "person_income": 600000.0,
  "person_home_ownership": "MORTGAGE",
  "person_emp_length": 5.0,
  "loan_intent": "EDUCATION",
  "loan_grade": "A",
  "loan_amnt": 100000.0,
  "loan_int_rate": 11.5,
  "loan_percent_income": 0.17,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 6
}
```
Response:
```json
{
  "default_probability": 0.0253,
  "default_prediction": 0,
  "threshold": 0.9996,
  "Result": "Low Risk"
}
```
The app also serves the static UI from the root path (`/`), and Swagger UI is available at `/docs`.

### Architecture notes
- The model and threshold are loaded once at start-up through a FastAPI **lifespan** handler.
- CORS is enabled (`allow_origins=["*"]`), so the UI can be hosted separately (for example on Vercel or Netlify) from the API.
- Errors are caught and returned as JSON with a traceback to simplify debugging.

### Tech stack
Python 3.11+, FastAPI, Uvicorn, Pydantic v2, scikit-learn, pandas, XGBoost, joblib; front end in HTML5, CSS3 (Grid, variables, glassmorphism) and vanilla JavaScript.

### Repository structure
```
Credit_risk_prediction-system-/
├── main.py                  # FastAPI app (/predict + static mount)
├── Credit_Risk.ipynb        # EDA, training, SHAP analysis
├── credit_risk_dataset.csv
├── credit_risk_model.pkl  best_threshold.pkl
├── static/                  # UI (HTML/CSS/JS, logo)
├── requirements.txt  runtime.txt  .python-version  render.yaml
└── README.md
```

### Run locally
```bash
git clone https://github.com/masumganvir/Credit_risk_prediction-system-.git
cd Credit_risk_prediction-system-
pip install -r requirements.txt
python -m uvicorn main:app --reload      # http://127.0.0.1:8000
```

### Limitations and notes
- The tuned threshold in the sample response is very high (≈ 0.9996), so the model is deliberately conservative about flagging a default. Review the threshold selection in the notebook before drawing business conclusions.
- Outputs are model estimates and **not a lending decision**.

---

## 4. 📈 Stock Price Prediction System

**Repository:** https://github.com/masumganvir/stock_prediction_system
**Live app:** https://stock-prediction-system-application-1.streamlit.app/

### Overview
A stock forecasting application that downloads live market data from Yahoo Finance, engineers **19 technical-indicator features**, trains **5 regression models** on the fly and forecasts prices for a chosen horizon (7 to 180 days).

### Data
Data is fetched at request time with `yfinance` (default period: 2 years of daily bars, auto-adjusted). Tickers such as `AAPL`, `MSFT`, `TSLA`, `NVDA`, `RELIANCE.NS` and `BTC-USD` are supported.

### Features (19)
| Category | Features |
|----------|----------|
| Price | Adjusted close |
| Trend | MA 5, 10, 20, 50; EMA 12, 26 |
| Momentum | MACD, MACD signal, RSI (14), 5-day momentum, 10-day rate of change |
| Volatility | Bollinger upper, lower and width; 20-day rolling volatility; distance from 20-day MA (%) |
| Volume | Raw volume, volume ratio (vs 10-day average) |

### Modelling pipeline (`predictor.py`)
1. Download OHLCV data and build indicators.
2. Create the target as the **price N days ahead** (`Adj_Close.shift(-N)`).
3. Standardise features, hold out the last N rows as the forecast input, and split the rest 80/20 (`random_state=42`).
4. Train and evaluate five models:

| Model | Configuration |
|-------|---------------|
| Linear Regression | Baseline |
| Ridge Regression | `alpha=1.0` |
| Random Forest | 200 trees, depth 8 |
| Gradient Boosting | 300 estimators, learning rate 0.05, depth 4, subsample 0.8 |
| Voting Ensemble | Gradient Boosting + Random Forest + Ridge |

5. Report MAE, MSE, RMSE, R² (test) and train R² for each model, pick the **best model by R²**, and return forecasts, feature importances, linear coefficients and a snapshot of the latest indicators.

### Benchmark (AAPL, from the project README)
| Model | MAE ($) | RMSE ($) | Test R² |
|-------|---------|----------|---------|
| **Gradient Boosting** | 3.26 | 4.50 | **0.974** |
| Random Forest | 4.00 | 5.53 | 0.961 |
| Voting Ensemble | 5.16 | 6.81 | 0.941 |
| Linear Regression | 8.17 | 11.31 | 0.838 |
| Ridge Regression | 9.17 | 11.97 | 0.818 |

### Dashboard features
- Real-time ticker search and a forecast-horizon slider.
- Charts: forecast curves, actual vs predicted, residual analysis, Bollinger Bands, MACD and RSI overlays.
- Day-by-day forecast table with **CSV export** and a printable prediction report.

### Two ways to run it
**Option A — Streamlit (recommended)**
```bash
git clone https://github.com/masumganvir/stock_prediction_system.git
cd stock_prediction_system
pip install -r requirements.txt
streamlit run streamlit_app.py           # http://localhost:8501
```

**Option B — Flask + HTML/CSS/JS dashboard (Chart.js)**
```bash
cd Stock-Prediction-System-Application
pip install -r requirements.txt
python api/app.py                        # http://127.0.0.1:5000
```

### Repository structure
```
stock_prediction_system/
├── streamlit_app.py            # Streamlit front end
├── predictor.py                # Feature engineering + 5-model pipeline
├── requirements.txt  .streamlit/
└── Stock-Prediction-System-Application/
    ├── api/            app.py (Flask REST API), predictor.py
    ├── dashboard/      index.html, style.css, app.js (Chart.js)
    ├── app/ core/ manage.py     # Django project (earlier version)
    ├── Stock_Prediction_System.ipynb
    └── MODEL_PREDICTION_REPORT.md
```

### Tech stack
Python, Streamlit, Flask + Flask-CORS, Django (legacy), scikit-learn, pandas, NumPy, yfinance, Chart.js, HTML5/CSS3/ES6.

### Limitations and notes
- The train/test split is **random rather than chronological**, and scaling is fitted on the full dataset, so R² values are optimistic compared with real out-of-sample trading performance. A walk-forward split would give a more realistic estimate.
- Future forecast values are produced from the most recent feature rows, so long horizons carry large uncertainty.
- For education and research only (see the [disclaimer](#disclaimer)).

---

## 5. 🏙️ NYC Airbnb Room Type Predictor

**Repository:** https://github.com/masumganvir/House-availability-preditions
**Live backend:** https://house-availability-preditions-2347.onrender.com/
**API docs:** `/docs` (Swagger) and `/redoc`

> Naming note: although the repository is called *House-availability-preditions*, the project is a **room-type classifier for New York City Airbnb listings**.

### Overview
Given listing metadata (location, price, minimum stay, review activity, host portfolio size, availability and neighbourhood), the app predicts whether a listing is an **Entire home/apt**, a **Private room** or a **Shared room**, with class probabilities and a confidence score.

### Dataset
New York City Airbnb Open Data: 48,895+ listings across Manhattan, Brooklyn, Queens, the Bronx and Staten Island. Exploration and training are in `nyc_airbnb_room_type_classification.ipynb`.

### Features
| Feature | Type | Constraint |
|---------|------|-----------|
| `latitude` | float | −90 to 90 |
| `longitude` | float | −180 to 180 |
| `price` | float | > 0 (USD per night) |
| `minimum_nights` | int | 1–365 |
| `number_of_reviews` | int | ≥ 0 |
| `reviews_per_month` | float | ≥ 0 |
| `calculated_host_listings_count` | int | ≥ 0 |
| `availability_365` | int | 0–365 |
| `neighbourhood_group` | str | Borough name |
| `neighbourhood` | str | Neighbourhood name |

### Model pipeline (`Model_Pipeline.pkl`)
A single serialised scikit-learn pipeline:
- **Numerical branch:** `SimpleImputer(mean)` → `StandardScaler`
- **Categorical branch:** `SimpleImputer` → `OneHotEncoder(handle_unknown='ignore')`
- **Estimator:** tuned `RandomForestClassifier`

Class order is alphabetical: `Entire home/apt`, `Private room`, `Shared room`.

### API
| Route | Method | Purpose |
|-------|--------|---------|
| `/` | GET | Web UI for browsers, JSON info for API clients |
| `/ui`, `/index.html` | GET | Web UI |
| `/guide` | GET | Interactive architecture and pipeline guide |
| `/health` | GET | `{"status": "ok", "model": "Random Forest Pipeline"}` |
| `/predict` | POST | Room-type prediction |

Example:
```bash
curl -X POST "https://house-availability-preditions-2347.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{"latitude":40.7484,"longitude":-73.9857,"price":120.0,"minimum_nights":2,
       "number_of_reviews":84,"reviews_per_month":2.3,"calculated_host_listings_count":1,
       "availability_365":210,"neighbourhood_group":"Manhattan","neighbourhood":"Midtown"}'
```
```json
{
  "Predicted_room_type": "Entire home/apt",
  "Probability": [0.6269, 0.3453, 0.0278],
  "Probabilities": {"Entire home/apt": 0.626884, "Private room": 0.345299, "Shared room": 0.027818},
  "confidence": 62.69
}
```

### Front end
A dark "cyber-skyline" interface with a real-time particle canvas, 3D building visualiser, probability charts, one-click presets and a backend health monitor that retries while a sleeping Render instance wakes up.

### Run locally
```bash
git clone https://github.com/masumganvir/House-availability-preditions.git
cd House-availability-preditions
python -m venv venv && source venv/bin/activate      # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```
Open `http://127.0.0.1:8000/` (UI) or `/docs` (Swagger).

### Deployment (Render)
Python 3.12.3 (`runtime.txt`); build `pip install -r requirements.txt`; start `uvicorn main:app --host 0.0.0.0 --port $PORT`; health check `/health`.

### Tech stack
Python 3.12, FastAPI, Uvicorn, Pydantic, scikit-learn, pandas, NumPy, joblib, HTML/CSS/JavaScript.

### Repository structure
```
House-availability-preditions/
├── main.py                                   # FastAPI server + routes
├── Model_Pipeline.pkl                        # Serialised sklearn pipeline
├── nyc_airbnb_room_type_classification.ipynb # EDA + training
├── index.html  style.css  script.js  favicon.svg
├── the_build_line_guide.html                 # Architecture guide
├── requirements.txt  runtime.txt
└── README.md
```

### Limitations and notes
- Predictions for atypical listings (for example a very cheap entire apartment) will tend to show lower confidence, as in the sample response above (62.69%).
- Neighbourhood names not seen in training are ignored by the encoder (`handle_unknown='ignore'`) rather than raising an error.
- The pipeline description above follows the project's own README; `Model_Pipeline.pkl` itself was not inspected.

---

## 6. 🏏 IPL Win Probability Predictor

**Repository:** https://github.com/masumganvir/IPL_winning_match_team_system

### Overview
A full-stack app that predicts the **live win probability of the chasing team** in an IPL match during the second innings. Enter the teams, city, target, current score, overs and wickets, and the app returns win/loss probabilities, run rates and a match-momentum summary.

### Data and feature engineering (`train_model.py`)
- **Data:** `matches.csv` (match-level) and `deliveries.csv` (ball-by-ball).
- **Cleaning:** first-innings total merged into the match table; old franchise names standardised (Delhi Daredevils → Delhi Capitals, Deccan Chargers → Sunrisers Hyderabad); only the 8 current teams kept; Duckworth–Lewis matches excluded.
- **Match-state features (per ball of the second innings):**

| Feature | Meaning |
|---------|---------|
| `batting_team`, `bowling_team`, `city` | Categorical context |
| `runs_left` | Target minus current score |
| `balls_left` | Balls remaining in the innings |
| `wickets` | Wickets in hand (10 minus wickets lost) |
| `total_runs_x` | Target set by the first innings |
| `crr` | Current run rate |
| `rrr` | Required run rate |

- **Target label:** `1` if the batting team won the match, else `0`.

### Model
- **Pipeline:** `ColumnTransformer` with `OneHotEncoder(drop='first')` for the three categorical columns, then `LogisticRegression(solver='liblinear', C=1.0)`.
- **Split:** 80/20 train/test.
- **Accuracy:** about **80.45%**, as reported in the project README. The trained pipeline is saved as `pipe.pkl`.
- Retrain any time with `python train_model.py`.

### Application features
- **Live chase centre:** runs needed, balls remaining, CRR and RRR update as inputs change.
- **Dynamic stadium theming** that adapts to the selected batting and bowling teams.
- **Animated radial gauges** and a **tug-of-war bar** for momentum.
- **Projected progression simulator**, an over-by-over win-probability curve.
- **Match presets** such as famous cliffhangers, high-scoring chases and low-scoring defences.
- **Three interfaces:** Flask web dashboard (`app.py`), Streamlit app (`streamlit_app.py`) and a Python/CLI module (`predict_cli.py`).

### REST API
| Route | Method | Description |
|-------|--------|-------------|
| `/api/health` | GET | Service and model-loaded status |
| `/api/predict` | POST | Win probability and match state |
| `/api/progression` | POST | Over-by-over probability trajectory (over 1 to 20) |

Request:
```json
{
  "batting_team": "Mumbai Indians", "bowling_team": "Chennai Super Kings",
  "city": "Mumbai", "target": 180, "score": 135, "overs": 15.0, "wickets": 3
}
```
Response (abridged):
```json
{
  "status": "success", "win_probability": 86.4, "loss_probability": 13.6,
  "runs_left": 45, "balls_left": 30, "wickets_remaining": 7,
  "crr": 9.0, "rrr": 9.0,
  "momentum": "Strong Advantage: Mumbai Indians"
}
```

### Use as a Python module
```python
from predict_cli import IPLPredictor

predictor = IPLPredictor()
result = predictor.predict(
    batting_team='Mumbai Indians', bowling_team='Chennai Super Kings',
    city='Mumbai', target=180, score=135, overs=15.0, wickets_out=3
)
print(f"Win Probability: {result['win_probability']}%")
```
Or run the interactive terminal console with `python predict_cli.py`.

### Run locally
```bash
git clone https://github.com/masumganvir/IPL_winning_match_team_system.git
cd IPL_winning_match_team_system
pip install -r requirements.txt
python app.py                       # http://127.0.0.1:5000
# or
streamlit run streamlit_app.py
```

### Deployment (Render)
Use the **Blueprint** (`render.yaml`) or a manual web service: build `pip install -r requirements.txt`, start `gunicorn app:app --bind 0.0.0.0:$PORT`, free instance type.

### Tech stack
Python, Flask, Gunicorn, Streamlit, scikit-learn, pandas, NumPy, SVG/CSS animations, vanilla JavaScript.

### Repository structure
```
IPL_winning_match_team_system/
├── app.py  streamlit_app.py  predict_cli.py   # Three interfaces
├── train_model.py                              # Data prep + training + export
├── pipe.pkl                                    # Trained pipeline
├── matches.csv  deliveries.csv                 # Datasets
├── templates/  static/                         # Web dashboard
├── Procfile  render.yaml  setup.sh  requirements.txt
├── Untitled.ipynb                              # Exploratory notebook
└── README.md
```

### Limitations and notes
- The model only covers the eight teams and the venues present in the training data, and it ignores player-level information such as who is batting.
- It works only for **second-innings** situations.
- Logistic regression treats features additively, so it can miss interactions such as the effect of wickets falling late in a close chase.

---

## Common Setup Guide

All six projects share the same basic workflow:

```bash
# 1. Clone the project you want
git clone https://github.com/masumganvir/<repository-name>.git
cd <repository-name>

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: .\venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the app (see the project's section above)
```

| Project | Start command | Default URL |
|---------|---------------|-------------|
| Book Recommender | `python app.py` | http://127.0.0.1:5000 |
| Mental Health | `python -m uvicorn main:app --port 2200 --reload` | http://127.0.0.1:2200 |
| Credit Risk | `python -m uvicorn main:app --reload` | http://127.0.0.1:8000 |
| Stock Prediction | `streamlit run streamlit_app.py` | http://localhost:8501 |
| NYC Airbnb | `uvicorn main:app --reload --port 8000` | http://127.0.0.1:8000 |
| IPL Predictor | `python app.py` | http://127.0.0.1:5000 |

**Tips**
- Use the Python version noted in each section (3.9+ to 3.12). Pickled scikit-learn models load most reliably with the same library versions pinned in `requirements.txt`.
- Free Render instances sleep when idle, so the first request after a pause can take a while.

## Shared Design Patterns

- **Train offline, serve online:** notebooks or training scripts produce `.pkl` artifacts that the web app loads once at start-up.
- **Pipelines keep preprocessing with the model** (credit risk, Airbnb, IPL), so the API accepts raw user input.
- **Validated inputs:** FastAPI projects use Pydantic schemas with range and enum checks.
- **CORS enabled** on the FastAPI services so the front end can be hosted separately.
- **Graceful degradation:** the mental health app falls back to a rule engine when Gemini is unavailable.
- **Health endpoints** (`/health`, `/api/health`) for hosting-platform monitoring.
- **Custom front ends** with no heavy framework: HTML, CSS and vanilla JavaScript (plus Chart.js for the stock dashboard).

## Disclaimer

These projects are for **educational and demonstration purposes**:

- The **stock forecasts** are not financial or investment advice.
- The **credit risk** outputs are model estimates and not a real lending decision.
- The **mental health** score is not a diagnosis. If you are struggling, please reach out to a qualified professional or someone you trust.
- The **IPL win probabilities** are statistical estimates and not betting guidance.

## Author

**Masum Ganvir** — [@masumganvir](https://github.com/masumganvir)

Feedback, issues and pull requests are welcome on each repository.
