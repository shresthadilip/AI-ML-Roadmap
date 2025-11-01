# Matplotlib and Seaborn: A Comprehensive Guide

## Matplotlib: The Foundational Library

Matplotlib is a low-level Python library for creating static, animated, and interactive visualizations. Most other high-level visualization libraries, like Seaborn and the plotting functions in Pandas, are built on top of Matplotlib.

First, let's import the necessary libraries.
```python
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
```

### The Matplotlib Architecture: Figure, Axes, Artist
Understanding Matplotlib's structure is key to mastering it. Every plot consists of a hierarchy of components.

| Component | Description |
|---|---|
| **Figure** | The entire window or page the plot is drawn on. |
| **Axes** | The actual plot area containing the data, axes, labels, etc. A Figure can have multiple Axes. |
| **Axis** | The number-line-like objects that handle the data limits (the x-axis and y-axis). |
| **Artist** | Everything you can see on the figure, including the lines, text, and patches. |

The best practice is to use the object-oriented interface, which gives you explicit control over each element of your plot.

```python
fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4], [10, 20, 25, 30])
ax.set_xlabel("X-Axis")
ax.set_ylabel("Y-Axis")
ax.set_title("Basic Line Plot")
plt.show()
```

### Figures and Subplots
You can create multiple plots within a single figure using `plt.subplots()`. By passing the number of rows and columns, you can create a grid of Axes.

```python
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot([1, 2, 3], [1, 4, 9])
axes[0].set_title("Left Plot")

axes[1].plot([1, 2, 3], [1, 2, 3])
axes[1].set_title("Right Plot")

plt.tight_layout()
plt.show()
```

### Common Plot Types
Matplotlib supports a wide variety of plots. Here are a few common examples:

```python
# Line Plot
plt.plot([0, 1, 2, 3], [10, 20, 25, 30], color='blue', linestyle='--', marker='o')
plt.title("Line Plot")
plt.show()

# Scatter Plot
plt.scatter([1, 2, 3, 4], [10, 20, 25, 30], color='red')
plt.title("Scatter Plot")
plt.show()

# Bar Plot
plt.bar(["A", "B", "C"], [10, 15, 7], color='green')
plt.title("Bar Plot")
plt.show()

# Histogram
data = np.random.randn(1000)
plt.hist(data, bins=30, color='purple', edgecolor='black')
plt.title("Histogram")
plt.show()

# Pie Chart
labels = ['A', 'B', 'C']
sizes = [40, 35, 25]
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title("Pie Chart")
plt.show()
```

### Customizing Plots
Matplotlib's strength lies in its customizability. You can control nearly every aspect of your plot.

```python
# Titles and Labels
plt.plot([1, 2, 3], [4, 9, 16])
plt.title("Customized Plot")
plt.xlabel("X Axis Label")
plt.ylabel("Y Axis Label")
plt.show()

# Legends
plt.plot([1, 2, 3], [1, 4, 9], label='Squared')
plt.plot([1, 2, 3], [1, 8, 27], label='Cubed')
plt.legend()
plt.show()

# Gridlines
plt.plot([1, 2, 3], [1, 4, 9])
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
```

### Styling and Saving
You can use shorthand notation for styling lines and save your finished plot to a file.

```python
# Line Styles and Colors
plt.plot([1, 2, 3], [1, 4, 9], 'r--', label='Dashed Red Line')
plt.legend()
plt.show()

# Save to File
plt.plot([1, 2, 3], [1, 4, 9])
plt.title("Saving Example")
plt.savefig("my_plot.png", dpi=300, bbox_inches='tight')
plt.show()
```

### Object-Oriented Interface (Best Practice)
Using the object-oriented approach (`fig, ax = plt.subplots()`) is highly recommended as it provides greater control and clarity, especially for complex figures with multiple plots.

```python
fig, ax = plt.subplots(figsize=(6,4))
ax.plot([1, 2, 3], [1, 4, 9], label='Squared')
ax.set_title("Object-Oriented Plot")
ax.set_xlabel("X-Axis")
ax.set_ylabel("Y-Axis")
ax.legend()
plt.show()
```

---

## Seaborn: Statistical Data Visualization

### What is Seaborn?
**Seaborn** is a **high-level data visualization library** built on **Matplotlib**. It provides:
- Simplified syntax for common plots
- Beautiful default themes
- Tight integration with Pandas DataFrames
- Built-in statistical plotting capabilities

> Think of Seaborn as “Matplotlib + Intelligence + Style.”

### Seaborn vs. Matplotlib

| Feature | Matplotlib | Seaborn |
|---|---|---|
| Control | Manual, detailed | High-level, automated |
| Syntax | Verbose | Concise |
| Default Style | Basic | Beautiful |
| Integration | Numpy/Matplotlib | Pandas DataFrame |
| Ideal Use | Custom visualization | Quick data exploration |

### Importing and Loading Data
Seaborn comes with several built-in datasets, which are useful for examples. We'll use the `tips` dataset.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Load built-in dataset
tips = sns.load_dataset("tips")
tips.head()
```

### Relational Plots
These plots are used to understand the relationship between two variables.

#### `sns.scatterplot()` and `sns.lineplot()`
```python
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="time", style="smoker", palette="viridis", s=80)
plt.title("Tips vs Total Bill (Seaborn)")
plt.show()

flights = sns.load_dataset("flights")
sns.lineplot(data=flights, x="year", y="passengers")
plt.title("Passengers Over Years")
plt.show()
```

### Categorical Plots — Comparing Groups
Seaborn excels at creating plots that compare data across different categories.

```python
sns.barplot(data=tips, x="day", y="tip", hue="sex", palette="mako")
plt.title("Average Tip by Day and Gender")
plt.show()

sns.boxplot(data=tips, x="day", y="total_bill", hue="smoker", palette="cool")
plt.title("Bill Distribution by Day and Smoker Status")
plt.show()

sns.violinplot(data=tips, x="day", y="total_bill", hue="sex", split=True, palette="pastel")
plt.title("Violin Plot of Total Bill by Day and Gender")
plt.show()
```

### Distribution Plots — Data Distribution and Density
These plots help visualize the distribution of a single variable.

```python
sns.histplot(data=tips, x="total_bill", kde=True, color="skyblue")
plt.title("Histogram of Total Bills")
plt.show()

sns.kdeplot(data=tips, x="tip", fill=True, color="green")
plt.title("Density of Tip Amounts")
plt.show()
```

### Regression Plots — Relationship & Trends
These plots automatically add a linear regression line and a confidence interval to a scatter plot.

```python
sns.lmplot(data=tips, x="total_bill", y="tip", hue="sex", height=5, aspect=1.2)
plt.title("Regression of Tip vs Total Bill by Gender")
plt.show()

sns.regplot(data=tips, x="size", y="tip", ci=None, color="purple")
plt.title("Regression without Confidence Interval")
plt.show()
```

### Matrix and Heatmap Plots — Correlations
A heatmap is an excellent way to visualize a matrix of data, such as a correlation matrix.

```python
corr = tips.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Tips Dataset")
plt.show()
```

### Pairwise Relationships — `sns.pairplot()`
The `pairplot` function creates a grid of scatterplots for every pair of numerical columns in a DataFrame, with histograms or KDEs on the diagonal. It's a powerful tool for quickly spotting relationships.

```python
sns.pairplot(data=tips, hue="sex", diag_kind="kde", palette="husl")
plt.suptitle("Pairwise Relationships in Tips Dataset", y=1.02)
plt.show()
```

### Multi-Plot Grids — `sns.FacetGrid()`
For more customized multi-plot grids, `FacetGrid` allows you to map a plotting function across the rows and columns of a grid, conditioned on different variables.

```python
g = sns.FacetGrid(tips, col="sex", row="smoker", margin_titles=True)
g.map_dataframe(sns.scatterplot, x="total_bill", y="tip", color="steelblue")
g.add_legend()
plt.show()
```