**\[[🇧🇷 Português](README.pt_BR.md)\] \[**[🇬🇧 English](README.md)**\]**

<br>

# 16- [Data Mining]() / [LLM Tabular Preprocessing with Dictionary Groups]()

### ***Dictionary-Based Feature Grouping [for]() LLM/AI Pipelines***

<br><br>

[**Institution:**]() Pontifical Catholic University of São Paulo (PUC-SP)  
[**School:**]() Faculty of Interdisciplinary Studies  
[**Program:**]() Humanistic AI and Data Science  
[**Semester:**]() 2nd Semester 2025  
Professor: [***Professor Doctor in Mathematics Daniel Rodrigues da Silva***](https://www.linkedin.com/in/daniel-rodrigues-048654a5/)

<br><br>

---

## 📚 Table of Contents

1. [Overview](#overview)
2. [What is Dictionary-Based Feature Grouping?](#what-is-dictionary-based-feature-grouping)
3. [Why Use This Technique?](#why-use-this-technique)
4. [Key Concepts](#key-concepts)
5. [Installation](#installation)
6. [Quick Start](#quick-start)
7. [Basic Examples](#basic-examples)
8. [Advanced Usage with LLMs](#advanced-usage-with-llms)
9. [Real-World Applications](#real-world-applications)
10. [Project Structure](#project-structure)
11. [Notebooks](#notebooks)
12. [Dataset Resources](#dataset-resources)
13. [References](#references)
14. [Contact](#contact)

---

## 🎯 Overview

This repository demonstrates **dictionary-based feature grouping** for tabular data preprocessing, specifically designed for integration with **Large Language Models (LLMs)** and AI/ML pipelines.

The technique allows you to organize related columns (features) in a dataset using **dictionaries**, enabling:
- ✅ Semantic grouping of features
- ✅ Efficient preprocessing for LLM-based feature engineering
- ✅ Better interpretability of tabular data
- ✅ Streamlined data transformation pipelines

👉 **Perfect for**: Data Scientists, ML Engineers, AI Researchers, and Students working with tabular data and LLMs!

---

## 🤔 What is Dictionary-Based Feature Grouping?

**Dictionary-based feature grouping** is a data preprocessing technique where you use Python dictionaries to organize and group related columns (features) in a DataFrame based on their **semantic meaning** or **data type**.

### 💡 Simple Explanation (For Beginners)

Imagine you have a dataset about customers with many columns:
```
age, income, city, state, country, purchase_date, product_name, price, ...
```

Instead of processing all columns individually, you can **group them** by meaning:

```python
feature_groups = {
    'demographics': ['age', 'income'],
    'location': ['city', 'state', 'country'],
    'transaction': ['purchase_date', 'product_name', 'price']
}
```

This makes it easier to:
1. Apply specific transformations to each group
2. Feed organized data to LLMs
3. Understand your dataset structure
4. Create modular and maintainable code

---

## 🎯 Why Use This Technique?

### **For Traditional ML**
- 📦 **Organized Feature Engineering**: Group numerical, categorical, and text features separately
- ⚛️ **Pipeline Efficiency**: Apply different transformers to different feature groups
- 🧠 **Better Understanding**: Know which features belong together conceptually

### **For LLM Integration**
- 🤖 **Semantic Context**: LLMs perform better when features are semantically grouped
- 💬 **Prompt Engineering**: Create structured prompts with organized feature groups
- 🔗 **Hybrid Models**: Combine tabular data with LLM embeddings effectively
- 🚀 **Feature Generation**: Use LLMs to create new features from grouped columns

---

## 📝 Key Concepts

### 1. **Pandas GroupBy**
Core Python/Pandas functionality for splitting, applying, and combining data:
```python
df.groupby('category').agg({'value': 'mean'})
```

### 2. **Dictionary Mapping**
Using dictionaries to define feature relationships:
```python
column_mapping = {
    'group_name': ['col1', 'col2', 'col3']
}
```

### 3. **LLM Feature Engineering**
Leveraging LLMs to:
- Generate text embeddings from grouped text columns
- Create semantic features
- Enrich tabular data with contextual information

---

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/Quantum-Software-Development/16-DataMining_llm-tabular-preprocessing-dict-groups.git
cd 16-DataMining_llm-tabular-preprocessing-dict-groups

# Install requirements
pip install -r requirements.txt
```

### Docker Setup (Optional)

```bash
# Build Docker image
docker build -t dict-groups-preprocessing .

# Run container
docker run -p 8888:8888 dict-groups-preprocessing
```

---

## 🚀 Quick Start

### Basic Example

```python
import pandas as pd

# Sample dataset
data = {
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['NY', 'LA', 'Chicago'],
    'salary': [50000, 60000, 70000],
    'department': ['IT', 'HR', 'IT']
}
df = pd.DataFrame(data)

# Define feature groups
feature_dict = {
    'personal': ['name', 'age'],
    'location': ['city'],
    'professional': ['salary', 'department']
}

# Process by group
for group_name, columns in feature_dict.items():
    print(f"\nProcessing {group_name}:")
    print(df[columns].head())
```

### Output:
```
Processing personal:
      name  age
0    Alice   25
1      Bob   30
2  Charlie   35

Processing location:
       city
0        NY
1        LA
2   Chicago

Processing professional:
   salary department
0   50000         IT
1   60000         HR
2   70000         IT
```

---

## 💻 Basic Examples

### Example 1: Grouping by Data Type

```python
import pandas as pd
import numpy as np

# Create sample data
df = pd.DataFrame({
    'id': [1, 2, 3],
    'name': ['Product A', 'Product B', 'Product C'],
    'price': [10.5, 20.0, 15.75],
    'category': ['Electronics', 'Clothing', 'Electronics']
})

# Group columns by data type
type_groups = {
    'numeric': df.select_dtypes(include=[np.number]).columns.tolist(),
    'categorical': ['category'],
    'text': ['name']
}
```

---

## 🤖 Advanced Usage with LLMs

### LLM-Based Feature Generation

```python
# Example: Using grouped text features for LLM prompts
text_groups = {
    'product_info': ['product_name', 'description'],
    'user_feedback': ['reviews']
}

# Create structured prompt for LLM
def create_llm_prompt(row, group_dict):
    prompt = ""
    for group_name, cols in group_dict.items():
        prompt += f"{group_name}: {', '.join([str(row[col]) for col in cols])}\n"
    return prompt
```

---

## 🌐 Real-World Applications

1. **E-commerce**: Group product features, pricing, and reviews
2. **Healthcare**: Organize patient demographics, vitals, and medical history
3. **Finance**: Separate transaction data, customer info, and risk factors
4. **NLP**: Combine tabular + text data for hybrid models

---

## 📂 Project Structure

```
16-DataMining_llm-tabular-preprocessing-dict-groups/
│
├── Codes/
│   ├── notebooks_01_basic_example.ipynb
│   └── notebooks_02_llm_preprocessing.ipynb
│
├── requirements.txt
├── requirements-dev.txt
├── requirements-full.txt
├── requirements-lock.txt
├── pyproject.toml
├── .gitignore
├── LICENSE
├── README.md
└── README.pt_BR.md
```

---

## 📓 Notebooks

### 1. `notebooks_01_basic_example.ipynb`
- Introduction to dictionary-based grouping
- Basic Pandas operations
- Simple examples with sample data

### 2. `notebooks_02_llm_preprocessing.ipynb`
- Advanced LLM integration
- Feature generation using grouped data
- Real-world dataset examples

👉 **Open in Colab**: [Basic Example](https://colab.research.google.com) | [LLM Preprocessing](https://colab.research.google.com)

---

## 📊 Dataset Resources

The notebooks use publicly available datasets:

- **UCI Machine Learning Repository**: https://archive.ics.uci.edu/ml/index.php
- **Kaggle Datasets**: https://www.kaggle.com/datasets
- **Hugging Face Datasets**: https://huggingface.co/datasets

---

## 📚 References

- **Chen, X., et al.** (2024). LLM-based feature generation from text for interpretable machine learning. *arXiv preprint*. Retrieved from [arxiv.org/html/2409.07132v2](https://arxiv.org/html/2409.07132v2)

- **DataCamp.** (2024). Pandas GroupBy Explained: Syntax, Examples, and Tips. Retrieved from [datacamp.com/tutorial/pandas-groupby](https://www.datacamp.com/tutorial/pandas-groupby)

- **GeeksforGeeks.** (2024). Pandas dataframe.groupby() Method. Retrieved from [geeksforgeeks.org](https://www.geeksforgeeks.org/pandas-groupby/)

- **Machine Learning Mastery.** (2024). Feature Engineering with LLM Embeddings: Enhancing Scikit-learn Models. Retrieved from [machinelearningmastery.com](https://machinelearningmastery.com)

- **McKinney, W.** (2017). *Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython* (2nd ed.). O'Reilly Media.

- **Pandas Documentation.** (2024). Group by: split-apply-combine. Retrieved from [pandas.pydata.org/docs/user_guide/groupby.html](https://pandas.pydata.org/docs/user_guide/groupby.html)

- **VanderPlas, J.** (2016). *Python Data Science Handbook: Essential Tools for Working with Data*. O'Reilly Media.

---

<br><br>

### ==================================

### Yupiiii 👩🏻‍🚀 Still Building ๋ ⭑🛸๋⭑

### =================================

<br><br>

#### 🌟 [Sponsor Quantum Software Development](https://github.com/sponsors/Quantum-Software-Development)


> [!Important]  
> ⚠️  
> Heads Up
> - • Projects and deliverables may be made [publicly available]() whenever possible.
> - • The course emphasizes [**practical, hands-on experience**]() with real datasets to simulate professional consulting scenarios in the fields of **Data Analysis and Data Mining** for partner organizations and institutions affiliated with the university.
> - • All activities comply with the [**academic and ethical guidelines of PUC-SP**]().
> - • Any content not authorized for public disclosure will remain [**confidential**]() and securely stored in [private repositories]().

<br><br>

##### 🎶 Prelude Suite no.1 (J. S. Bach) - [Sound Design Remix]()

https://github.com/user-attachments/assets/62dc99f6-e40f-4f48-8fa7-b00b3d443866

#### 📺 For better resolution, watch the video on [YouTube.](https://youtu.be/_ytC6S4oDbM)

> [!Tip]
> This repository is a review of the Statistics course from the undergraduate program Humanities, AI and Data Science at PUC-SP.

### ☞ **Access Data Mining [Main Repository](https://github.com/Quantum-Software-Development/1-Main_DataMining_Repository)**

---

## 💬 Contact

### 💌 [Let the data flow... Ping Me !](mailto:fabicampanari@proton.me)

#### 🛸๋ My Contacts [Hub](https://linktr.ee/fabianacampanari)

---

<br>

<p align="center">
  <img src="https://private-user-images.githubusercontent.com/113218619/368403872-517fc573-7607-4c5d-82a7-38383cc0537d.GIF?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzA4NzIxNjUsIm5iZiI6MTczMDg3MTg2NSwicGF0aCI6Ii8xMTMyMTg2MTkvMzY4NDAzODcyLTUxN2ZjNTczLTc2MDctNGM1ZC04MmE3LTM4MzgzY2MwNTM3ZC5HSUY_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQxMTA2JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MTEwNlQwNDM3NDVaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0zM2RiYjNlZTZiZTlhOTZjMWY4ZjE3MzY3MGVhNjc5NTE5ZWJiMmZiOGZiOWE2ZDkzYjZiNmI0NDVjMmYxZjg3JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.4HrEDrGYlnW7OTDHckH9Rt3uxHX8wQzCWgcZvRTKzQU" width="200">
</p>

<br>

<p align="center">
  ────────────── 🔭⬆──────────────
</p>

<p align="center">
  <a href="#top">➣➢➤ Back to Top</a>
</p>

<br>

###### Copyright 2026 Quantum Software Development. Code released under the [MIT License license.](https://github.com/Quantum-Software-Development/Math/blob/3bf8270ca09d3848f2bf22f9ac89368e52a2fb66/LICENSE)
