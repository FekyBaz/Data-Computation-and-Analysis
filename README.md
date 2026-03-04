# Application of Dimensionality Reduction for Multi-Class LDA

This repository is for **Project 1** on the application of dimensionality reduction using **filter-based feature selection** and **Multi-Class Linear Discriminant Analysis (LDA)**.

The assignment studies how different filter ranking criteria affect classifier performance and feature selection stability.

---

## Project Objective

Investigate and compare multiple filter feature ranking metrics, then train a **multi-class LDA** model using top-ranked features.

You are required to:

1. Implement feature ranking metrics:
   - Pearson Correlation
   - Spearman Rank Correlation
   - Gini Index
   - Chi-Square Statistic
   - Information Gain (Entropy-based)
2. Rank features using each metric.
3. Train a Multi-class LDA model using top-\(k\) selected features.
4. Compare model performance and selection stability.
5. Analyze theoretical differences between criteria.

---

## Phases (Required Workflow)

### Phase 1: Data Understanding and Basic Analysis
- Perform EDA (Exploratory Data Analysis).
- Clean and prepare the dataset.
- Build initial report tables and plots.
- Implement:
  - Pearson Correlation
  - Chi-Square

### Phase 2: Additional Filter Criteria
- Implement:
  - Spearman Rank Correlation
  - Gini Index

### Phase 3: Information-Theoretic Selection
- Implement Information Gain.
- Produce ranked feature lists for all metrics.

### Phase 4: Classification
- Train Multi-class LDA using top-\(k\) features from each ranking method.
- Evaluate model performance.

### Phase 5: Comparison and Discussion
- Compare feature-selection stability across methods.
- Analyze theoretical differences and practical impact.

---

## Dataset Requirements

Use a **multi-class classification dataset** with:

- At least **6 features**
- At least **3 classes**
- **No missing values**

---

## Team and Delivery Notes

- Maximum group size: **5 students**.
- The assignment will be discussed in lecture.

### Deadline Policy
- Submission expected by **next Saturday (in lecture)**.
- **No excuses policy**:
  - No late submissions accepted.
  - No deadline extensions.

---

## Submission Format

Students must submit:

1. **Python implementation** (`.ipynb` or `.py`)
2. **PDF Report** (5–8 pages)
3. **All plots** with proper labels
4. **Clear written explanations**
   - Do not submit screenshots of code without explanation.


---

## Recommended Evaluation Outputs

To support the report, include:

- Ranking tables for each metric
- Top-\(k\) feature subsets tested (e.g., \(k = 2, 4, 6\), etc.)
- LDA performance metrics (accuracy, confusion matrix, class-wise scores)
- Stability comparison between ranking methods
- Final discussion linking theory to observed results
