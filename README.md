# 📚 Student Performance Clustering & Hybrid Prediction Model

## 📝 Project Overview
This project introduces an intelligent analytical framework to monitor and predict student academic performance. Unlike traditional models, this system utilizes a **Hybrid Approach**: it first groups students based on behavioral patterns using **K-Means Clustering**, then feeds these insights into a **Random Forest Classifier** to predict final outcomes (Pass/Fail).

The goal is to provide educators with a data-driven tool to identify at-risk students early and understand the underlying behavioral factors affecting their grades.

---

## 🌟 Key Features

### 1. Behavioral Clustering (Unsupervised Learning)
* **Feature Selection:** Focuses on non-academic features (study time, absences, failures, social habits) to group students.
* **Algorithm:** **K-Means Clustering** with optimal 'K' determined by the **Elbow Method** and **Silhouette Score**.
* **Insight:** Automatically categorizes students into behavioral profiles (e.g., High Engagement vs. At-Risk groups).

### 2. Hybrid Predictive Modeling (Supervised Learning)
* **The Hybrid Edge:** The model doesn't just look at grades; it uses the `Cluster_ID` as a feature, significantly improving prediction accuracy.
* **Algorithm:** **Random Forest Classifier** optimized via **GridSearchCV**.
* **Performance:** Achieved an accuracy of **~88.6%**, outperforming standard models by incorporating behavioral cluster data.

### 3. Analytics & Deployment Dashboard
* **Interactive Streamlit UI:** A complete dashboard for:
    * **Batch Processing:** Uploading CSV files for large-scale analysis.
    * **Visual EDA:** Interactive charts for outlier detection and correlation maps.
    * **Individual Prediction:** Inputting a specific student's data to get a real-time Pass/Fail prediction with a confidence score.

---

## 📊 Methodology & Pipeline

1.  **Data Cleaning:** Handled semicolon-delimited data, removed duplicates, and performed outlier analysis.
2.  **Preprocessing:** Applied **StandardScaler** for clustering and **Label Encoding** for categorical variables.
3.  **Hybrid Logic:** * Step A: Perform K-Means to generate `cluster_id`.
    * Step B: Add `cluster_id` and `cluster_weight` to the original feature set.
    * Step C: Train Random Forest on the enriched dataset.
4.  **Evaluation:** Validation using Confusion Matrix, F1-Score, and Feature Importance analysis.



---

## 🛠 Tech Stack

* **Language:** Python
* **Data Science:** Pandas, NumPy, Scikit-learn
* **Visualization:** Plotly (Interactive), Matplotlib, Seaborn
* **Deployment:** Streamlit
* **Model Persistence:** Joblib (for saving/loading trained models)

---

## 🚀 Installation & Setup

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/mohamed-saied-1/Student-Performance-Clustering-Hybrid-Prediction-Model.git]
    ```
2.  **Install Dependencies:**
    ```bash
    pip install streamlit pandas numpy scikit-learn plotly matplotlib seaborn joblib
    ```
3.  **Run the Dashboard:**
    ```bash
    streamlit run streamlit_app.py
    ```

---

## 📁 Project Structure
* `streamlit_app.py`: The UI/Dashboard logic.
* `student_analysis.py`: Core logic for data processing, clustering, and ML training.
* `visualizations.py`: Custom class for professional dark-themed Plotly charts.
* `Student_Performance_main.ipynb`: The research phase and model development.
* `student-mat.csv`: The dataset (Portuguese secondary school students).

---
**Education Tech Analytics © 2026.**
*Empowering educators through Hybrid Machine Learning.*
