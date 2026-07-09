# 🌸 Iris Flower Classification using Machine Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.14-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikitlearn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?style=for-the-badge&logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-blue?style=for-the-badge)
![Status](https://img.shields.io/badge/Project-Completed-success?style=for-the-badge)

### DecodeLabs Artificial Intelligence Internship – Project 2

A complete Machine Learning pipeline that classifies Iris flowers using multiple supervised learning algorithms, compares their performance, saves the best model, and provides real-time predictions.

</div>

---

# 📖 Project Overview

This project demonstrates the end-to-end workflow of a supervised Machine Learning classification problem using the famous **Iris Dataset**.

The application performs:

- Dataset loading and preprocessing
- Exploratory Data Analysis (EDA)
- Data visualization
- Feature scaling
- Training multiple ML models
- Model evaluation and comparison
- Saving the best-performing model
- Predicting flower species from user input

The trained model is stored using **Joblib**, allowing future predictions without retraining.

---

# 🎯 Objectives

- Understand supervised Machine Learning concepts
- Explore the Iris dataset
- Visualize feature relationships
- Compare multiple classification algorithms
- Evaluate model performance
- Save trained models for deployment
- Build a reusable prediction system

---

# 📊 Dataset Information

**Dataset:** Iris Dataset (Scikit-Learn)

### Total Samples

- **150 Flowers**

### Features

| Feature | Description |
|----------|-------------|
| Sepal Length | Length of Sepal (cm) |
| Sepal Width | Width of Sepal (cm) |
| Petal Length | Length of Petal (cm) |
| Petal Width | Width of Petal (cm) |

### Target Classes

- 🌸 Iris Setosa
- 🌺 Iris Versicolor
- 🌼 Iris Virginica

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Joblib
- VS Code

---

# 🤖 Machine Learning Algorithms

The project trains and compares multiple supervised learning models.

| Algorithm | Purpose |
|-----------|----------|
| Logistic Regression | Linear Classification |
| Decision Tree Classifier | Tree-Based Learning |
| K-Nearest Neighbors (KNN) | Distance-Based Classification |
| Random Forest Classifier | Ensemble Learning |
| Support Vector Machine (SVM) | Maximum Margin Classification |

---

# ⚙ Machine Learning Workflow

```text
Dataset Loading
       │
       ▼
Data Exploration
       │
       ▼
Data Visualization
       │
       ▼
Feature Selection
       │
       ▼
Train-Test Split
       │
       ▼
Feature Scaling
       │
       ▼
Model Training
       │
       ▼
Model Evaluation
       │
       ▼
Best Model Selection
       │
       ▼
Model Saving (.pkl)
       │
       ▼
Prediction System
```

---

# 📈 Visualizations Generated

The project automatically creates the following visualizations:

- 📌 Pair Plot
- 📌 Correlation Heatmap
- 📌 Box Plot
- 📌 Confusion Matrix
- 📌 Model Accuracy Comparison

These visualizations are saved inside the **screenshots** folder.

---

# 📂 Project Structure

```text
DecodeLabs_Project2
│
├── app
│   ├── predictor.py
│   └── streamlit_app.py
│
├── datasets
│   └── iris.csv
│
├── models
│   ├── iris_model.pkl
│   └── scaler.pkl
│
├── screenshots
│   ├── pairplot.png
│   ├── heatmap.png
│   ├── boxplot.png
│   ├── confusion_matrix.png
│   └── model_accuracy.png
│
├── notebooks
│
├── main.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

# ▶️ Installation

Clone the repository

```bash
git clone YOUR_GITHUB_REPO_LINK
```

Move into the project folder

```bash
cd DecodeLabs_Project2
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🚀 Run the Project

### Train Models

```bash
python main.py
```

### Predict Flower Species

```bash
python app/predictor.py
```

### Streamlit Interface (Optional)

```bash
py -m streamlit run app/streamlit_app.py
```

---

# 📸 Project Output

The project generates:

- Trained ML Models
- Dataset CSV
- Performance Graphs
- Confusion Matrix
- Accuracy Comparison
- Real-Time Prediction System

---

# 📷 Screenshots

Add screenshots of:

- Dataset Preview
- Pair Plot
- Correlation Heatmap
- Box Plot
- Confusion Matrix
- Accuracy Comparison
- Predictor Output
- Streamlit Interface (Optional)

---

# 💡 Key Features

- Complete Machine Learning Pipeline
- Exploratory Data Analysis (EDA)
- Feature Scaling
- Multiple Classification Models
- Automatic Model Comparison
- Saved Models using Joblib
- Prediction from User Input
- Visualization of Dataset
- Modular Project Structure
- Optional Streamlit Web Interface

---

# 📚 Learning Outcomes

Through this project I learned:

- Data preprocessing
- Feature engineering
- Exploratory Data Analysis
- Supervised Machine Learning
- Model evaluation
- Confusion Matrix analysis
- Saving and loading ML models
- Building reusable prediction systems

---

# 🔮 Future Improvements

- Hyperparameter tuning
- Cross-validation
- Feature importance analysis
- Interactive Streamlit dashboard
- Cloud deployment
- Docker support
- REST API integration

---

# 👩‍💻 Author

**Hiral Goyal**

B.Tech – Mathematics & Computing  
Madhav Institute of Technology & Science (MITS), Gwalior

- GitHub: https://github.com/hiral17234
- LinkedIn: https://linkedin.com/in/hiralgoyal17

---

# 🙏 Acknowledgement

This project was completed as part of the **DecodeLabs Artificial Intelligence Internship Program**. It focuses on building a complete Machine Learning workflow using Python and Scikit-Learn while strengthening practical skills in data preprocessing, visualization, model training, evaluation, and deployment.

---

<div align="center">

⭐ If you found this project useful, consider giving it a star!

</div>
