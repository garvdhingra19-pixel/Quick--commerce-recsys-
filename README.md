# Quick-Commerce Recommendation Engine

## 📌 Project Overview

This project is a real-time, interactive recommendation system designed to mirror the machine learning architectures of modern quick-commerce platforms. It processes historical transactional data to generate highly personalized product carousels for end-users, dynamically balancing routine staples, cart-based associations, and novel algorithmic discoveries. The backend is fully integrated into a Streamlit web application for live, interactive deployment.

## 🧠 Core Recommendation Modules

### 1. Buy Again (Historical Frequency Baseline)

* **Methodology:** Analyzes past purchase logs to identify the user's highest-frequency consumer packaged goods (CPGs).
* **Functionality:** Acts as the primary retention driver by ensuring users can instantly reorder their essential staples.

### 2. Cart Add-ons (Lift-Based Association Rules)

* **Methodology:** Utilizes market basket analysis to recommend items that frequently co-occur in the same transaction as the item currently in the user's cart.
* **Algorithmic Edge:** Instead of relying on raw confidence or popularity bias (which disproportionately suggests generic items like bananas or milk), this module computes the **Lift metric** to identify statistically significant correlations between specific niche products.

### 3. New Discoveries (ALS Collaborative Filtering)

* **Methodology:** Implements Alternating Least Squares (ALS) matrix factorization to predict products a user has never bought but is highly likely to purchase.
* **Mathematical Foundation:** The algorithm factorizes a massive, sparse user-item interaction matrix into two lower-dimensional matrices representing hidden "latent traits" for users ($U$) and items ($V$). Predictions are calculated in real-time via the dot product of these vectors:
$\hat{r}_{ui} = \mathbf{u}_i^T \mathbf{v}_i$
* **Optimization:** Configured to utilize single-threaded OpenBLAS (`OPENBLAS_NUM_THREADS=1`) to prevent thread contention with the `implicit` library's native multi-threading, maximizing CPU performance.

## 🛠️ Technology Stack

* **Language:** Python 3.x
* **Data Manipulation:** Pandas, NumPy, SciPy
* **Machine Learning:** Implicit (Alternating Least Squares)
* **Web Framework & UI:** Streamlit
* **Version Control:** Git & GitHub

## 📂 Project Architecture

```text
quick-commerce-recsys/
│
├── app/
│   ├── main.py                # Core Streamlit web application and API logic
│   └── als_model.npz          # Trained collaborative filtering model weights
│
├── data/
│   ├── raw/                   # Original Instacart CSV datasets
│   └── cleaned/               # Pre-processed data ready for model ingestion
│
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   ├── 02_baselines.ipynb     # Statistical baseline modeling 
│   ├── 03_cart_suggestions.ipynb # Association rule mining
│   └── 04_collaborative_filtering.ipynb # ALS model training and evaluation
│
├── requirements.txt           # Production dependencies for cloud deployment
└── README.md                  # Project documentation

```

## 🚀 How to Run Locally

1. **Clone the repository:**
```bash
git clone https://github.com/garvdhingra19-pixel/Quick--commerce-recsys-.git
cd Quick--commerce-recsys-

```


2. **Install the dependencies:**
Ensure you have your preferred Python environment activated, then run:
```bash
pip install -r requirements.txt

```


3. **Launch the Streamlit Server:**
Navigate into the application folder and start the engine:
```bash
cd app
streamlit run main.py

```


The interactive dashboard will automatically open in your default web browser.

---

Copy this text, replace everything inside your current `README.md` file with it, and save the file. Run `git add README.md`, `git commit -m "Add formal README documentation"`, and `git push` to upload it to your repository.

Are you ready to head over to share.streamlit.io to instantly deploy this live to the internet?
