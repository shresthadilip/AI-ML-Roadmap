
# My Journey Learning AI/ML

Before I jumped into AI and machine learning, I realized it’s easy to get overwhelmed. So, I decided to tackle the most important Python libraries one by one, in a logical order.

| Hour | Library                     | Focus                   | Mini Project            |
| ---- | --------------------------- | ----------------------- | ----------------------- |
| 1    | **NumPy**                   | Math foundation         | House Price Predictor   |
| 2    | **Pandas**                  | Data cleaning           | Titanic data analysis   |
| 3    | **Matplotlib + Seaborn**    | Visualization           | Iris plots              |
| 4    | **Scikit-learn**            | Classical ML            | Iris classifier         |
| 5    | **PyTorch / TensorFlow**    | Deep learning basics    | MNIST classifier        |
| 6    | **SciPy + Plotly + MLflow** | Tooling & visualization | Model dashboard         |

Since I already have a solid programming background, I built myself a 6-hour crash course, an hour a day. Each hour, I focus on a different library, with clear goals and a mini project—so I learn by actually coding, not just reading.

---

## My 6-Hour AI/ML Python Library Crash Course

---

### Hour 1: NumPy – Building My Math Foundation

My goal: Get comfortable with arrays, vectorized math, and basic linear algebra.

I will be learning:
- How NumPy arrays differ from regular Python lists
- Creating arrays with `np.array`, `np.zeros`, `np.arange`, and random numbers
- Indexing, slicing, reshaping arrays
- Broadcasting and element-wise math
- Matrix multiplication (`dot`, `@`, `matmul`)
- Basic stats: mean, std, sum, argmax
- Linear algebra basics (`np.linalg.inv`, `np.linalg.det`)

**Mini Project:**
I will make a “House Price Predictor” using arrays for house sizes and prices, calculate mean and correlation, and manually implement simple linear regression.

---

### Hour 2: Pandas – Cleaning and Exploring Data

My goal: Learn to clean, explore, and prep data for ML.

I will be learning:
- Series and DataFrames
- Reading CSVs
- Selecting rows/columns with `loc` and `iloc`
- Handling missing data (`fillna`, `dropna`)
- Grouping and aggregation (`groupby`)
- Filtering and sorting
- One-hot encoding categorical data

**Mini Project:**
I will analyze Titanic survivors, calculate survival rates by gender/class, fill missing ages, create a “family_size” column, and export the cleaned data.

---

### Hour 3: Matplotlib & Seaborn – Visualizing Data

My goal: Visualize trends, distributions, and relationships.

I will be learning:
- Matplotlib basics (`plt.plot`, `plt.scatter`)
- Subplots and figure sizing
- Seaborn visualizations: histograms, boxplots, heatmaps, pairplots
- Customizing styles and color palettes
- Visualizing correlations and outliers

**Mini Project:**
I will visualize the Iris dataset with histograms, scatter plots colored by species, and a correlation heatmap.

---

### Hour 4: Scikit-learn – Classical Machine Learning

My goal: Train, test, and evaluate ML models.

I will be learning:
- ML workflow: split, train, predict, evaluate
- Linear & Logistic Regression
- Decision Trees and Random Forests
- KMeans clustering
- StandardScaler and normalization
- Model evaluation: accuracy, confusion matrix, cross-validation

**Mini Project:**
I will predict Iris species using Logistic Regression and Random Forest, evaluate accuracy, and visualize decision boundaries.

---

### Hour 5: PyTorch – Deep Learning Basics

My goal: Understand tensors, gradients, and simple neural networks.

I will be learning:
- Creating and manipulating tensors
- Automatic differentiation
- Building neural nets with `nn.Module`
- Loss functions and optimizers
- Training loops
Note: you can go with TensorFlow for deep learning. But for most beginners, PyTorch is often considered easier
**Mini Project:**
I will build a handwritten digit classifier for MNIST, train a simple network, and visualize predictions.

---

### Hour 6: Supporting Tools & Next Steps

My goal: Add tools for ML workflows and visualization.

I will be learning:
- SciPy for optimization and metrics
- Plotly for interactive charts
- MLflow/Joblib for saving models and tracking versions
- DVC for dataset version control
- Next steps: pipelines, feature engineering, deep learning frameworks

**Mini Project:**

I will train a model, save it, and build an interactive dashboard with Plotly.

---

As I progress through this learning plan, I will be sharing the notes and code generated during the process, including Jupyter notebooks, on GitHub as well as in Medium as a blog post.




