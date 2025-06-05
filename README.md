# Task-6-Elevate-labs
<br>
# Iris Dataset - KNN Classification

This repository demonstrates the implementation of a **K-Nearest Neighbors (KNN)** classification algorithm using the famous **Iris dataset**. The project includes normalization, model training, hyperparameter tuning (K values), model evaluation, and decision boundary visualization.

## 📊 Dataset

- **Name**: Iris Flower Dataset  
- **Source**: [UCI Machine Learning Repository](https://www.kaggle.com/datasets/uciml/iris)
- **Features**: 
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width
- **Target**: Iris species (Setosa, Versicolor, Virginica)

---

## 🔍 Tasks Performed

1. **Feature Normalization** using `StandardScaler`
2. **Model Training** with `KNeighborsClassifier` from `sklearn`
3. **K Tuning**: Tested multiple values of `K` to find the optimal one
4. **Evaluation**:
   - Accuracy Score
   - Confusion Matrix
5. **Decision Boundary Visualization** for better understanding of class separation

---

## 📁 Files

- `knn_iris.py` – Python script for the full analysis.
- `Iris.csv` – Dataset file (if not already available, download from the UCI repository).

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/knn-iris-classification.git
   cd knn-iris-classification
