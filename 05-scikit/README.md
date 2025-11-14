# What is **Scikit-learn (sklearn)?**

**Scikit-learn**, often imported as `sklearn`, is **the most popular and powerful open-source machine learning library in Python for classical ML tasks**.

It provides **simple and efficient tools** for:

* Data preprocessing (cleaning, scaling, encoding)
* Building machine learning models (classification, regression, clustering)
* Model evaluation and selection
* Pipelines for automating workflows

---

## Key Idea

Scikit-learn is built on top of:

* **NumPy** → for numerical computation
* **SciPy** → for scientific operations
* **Matplotlib** → for visualization

### The Estimator API

The central philosophy of Scikit-learn is its consistent, unified API (Application Programming Interface), known as the Estimator API. This consistency means that once you learn how to use one model (e.g., Linear Regression), you can easily use any other model (e.g., Support Vector Machines, Random Forests).

All models share three core methods:

| Method | Purpose | Description |
|--------|---------|-------------|
| `.fit(X, y)` | Training | Trains the model. `X` is the feature matrix (data), and `y` is the target variable (labels) |
| `.predict(X)` | Inference | Uses the trained model to make predictions on new data `X` |
| `.transform(X)` | Transformation | Used for preprocessing or dimensionality reduction steps (e.g., scaling data, extracting principal components). Not all estimators have this method. |

---

## Key Areas of Functionality

Scikit-learn is organized into modules that cover the primary tasks of classical machine learning:

That’s why the name **“Scikit”** stands for **“SciPy Toolkit.”**

---

## What You Can Do with Scikit-learn

| Task Type                 | Description                                               | Example Algorithms/Tools                               |
| ------------------------- | --------------------------------------------------------- | ------------------------------------------------------ |
| **Classification**        | Predicts which category or class a data point belongs to  | K-Nearest Neighbors (KNN), Support Vector Machines (SVM), Decision Trees, Random Forests, Logistic Regression |
| **Regression**            | Predicts continuous numerical values                      | Linear Regression, Ridge Regression, Lasso Regression, SVR (Support Vector Regression), Random Forest Regressor |
| **Clustering**            | Groups similar data points together without prior labels  | K-Means, DBSCAN, Hierarchical Clustering, PCA          |
| **Dimensionality Reduction** | Reduces feature count while preserving information     | Principal Component Analysis (PCA), Linear Discriminant Analysis (LDA) |
| **Preprocessing & Feature Engineering** | Prepares data for ML models (cleaning, scaling, encoding) | `SimpleImputer`, `StandardScaler`, `MinMaxScaler`, `OneHotEncoder`, `OrdinalEncoder` |
| **Model Selection & Evaluation** | Assesses model performance and selects optimal models | `train_test_split`, `GridSearchCV`, `cross_val_score`, `accuracy_score`, `mean_squared_error`, `roc_auc_score` |
| **Pipelines & ColumnTransformer** | Automate ML workflows and apply transformations to specific columns | `Pipeline`, `ColumnTransformer` |

---

## Example: Simple ML Workflow

```python
# Step 1: Import libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 2: Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Step 5: Predict
y_pred = clf.predict(X_test)

# Step 6: Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
```

---

## Why Use Scikit-learn
- Consistent API across all models
- Easy to switch algorithms (just change one line)
- Excellent documentation and community support
- Perfect for **rapid experimentation and prototyping**

---

## Summary

| Feature           | Scikit-learn                                       |
| ----------------- | -------------------------------------------------- |
| **Purpose**       | Classical machine learning                         |
| **Type**          | High-level library                                 |
| **Core Strength** | Ease of use, consistent API                        |
| **Best for**      | Data preprocessing, model training, evaluation     |
| **Not for**       | Deep learning (use PyTorch or TensorFlow for that) |



| Module            | Category                   | Example Function/Class                       |
| ----------------- | -------------------------- | -------------------------------------------- |
| `model_selection` | Data splitting, validation | `train_test_split`, `GridSearchCV`           |
| `preprocessing`   | Scaling, encoding          | `StandardScaler`, `OneHotEncoder`            |
| `linear_model`    | Regression, classification | `LinearRegression`, `LogisticRegression`     |
| `neighbors`       | Classification             | `KNeighborsClassifier`                       |
| `tree`            | Classification             | `DecisionTreeClassifier`                     |
| `ensemble`        | Boosting, bagging          | `RandomForestClassifier`, `GradientBoosting` |
| `cluster`         | Unsupervised learning      | `KMeans`, `AgglomerativeClustering`          |
| `decomposition`   | Dimensionality reduction   | `PCA`                                        |
