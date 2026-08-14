# Census Income Predictor 💼

An interactive Machine Learning web application built using **Streamlit** and **Scikit-Learn** to predict whether an individual's annual income exceeds $50K based on U.S. Census demographic data.

---

## 📁 Repository Structure

```
Census-Income-predictor/
├── app.py                      # Interactive Streamlit application (UI + model inference & EDA)
├── requirements.txt            # Pinned project dependencies
├── Decision_Tree_Practice.ipynb # Data cleaning, EDA, model training & hyperparameter tuning notebook
├── census-income.csv           # Cleaned source dataset (32,561 rows)
├── README.md                   # Project documentation & usage instructions
└── .gitignore                  # Git tracking rules
```

---

## ⚡ Quick Start

### 1. Prerequisites & Installation

Ensure you have Python 3.9+ installed. Clone or navigate to the project directory and install all required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run the Application

Launch the Streamlit dashboard locally:

```bash
streamlit run app.py
```

The web application will open automatically in your browser at `http://localhost:8501`.

---

## 🧠 Machine Learning Pipeline

- **Algorithm**: Decision Tree Classifier (`sklearn.tree.DecisionTreeClassifier`)
- **Dataset Size**: 32,561 records with 14 demographic and economic features.
- **Categorical Feature Encoding**: `LabelEncoder` fit across non-numerical features (`workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`).
- **Target Variable**: `annual_income` (`<=50K` or `>50K`).
- **Tuned Hyperparameters**: `max_depth=7`, `test_size=0.20`, `criterion='gini'`.
- **Model Accuracy**: ~85%+ test accuracy on Census evaluation data.

---

## 📊 Features & UI Highlights

1. **🔮 Live Income Predictor**:
   - Real-time form inputs for demographic, educational, and financial attributes.
   - Classification output badge (`<=50K` vs `>50K`) with prediction probability confidence scores.
2. **📊 Exploratory Data Analysis (EDA)**:
   - Target income class distribution bar & pie charts.
   - Age vs Income kernel density estimation (KDE).
   - Working hours per week distributions & Education vs Income correlation.
3. **🌲 Decision Tree Explorer**:
   - Confusion matrix heatmap and classification metrics table (Precision, Recall, F1-Score).
   - Feature importances ranking bar chart.
   - Interactive tree structure visualizer (`plot_tree`).

---

## 🛠️ Tech Stack

- **Frontend / Web App**: [Streamlit](https://streamlit.io/)
- **Machine Learning**: [Scikit-Learn](https://scikit-learn.org/)
- **Data Manipulation**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Data Visualization**: [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)
