# Used-Vehicle-Auctions!

![Used-Vehicle-Auctions](/Images/605.png)

## Project Overview

The objective of this project is to predict the likelihood of a car purchased at an auction being a "kick" (bad buy). This prediction can aid auto dealerships in implementing cost-effective retention strategies to prevent undesirable purchases and potential losses.

![Description](/Images/14-745.png)

## Dataset

![Download](/Data/)

- **Structured:** Yes
- **Format:** Single .csv file
- **Number of Features:** [33]
- **Target Feature (Vector):** `IsBadBuy`, imbalanced, [12:88]

![Distribution](/Images/Distribution-of-Good-and-Bad-Buys-in-df.png)

- **Duplicates:** [0]

## Problem Space

- Binary classification problem
- Modertate size dataset size for ML model training
- Imbalanced target class distribution
- Robust model development with independent performance metrics
- Custom scoring using F1 score (same emphasis on recall and precision)

## Best Models

- Notebook 4. LR Model (Logistic Regression)
- Notebook 5. DT-RF Model (Decision-Tree / Random Forest)

## Process

### 1. Exploratory Data Analysis (EDA)

The EDA.ipynb file encapsulates the process of Exploratory Data Analysis (EDA), where I thoroughly explore and analyze datasets to extract valuable insights and discern patterns
Below are the boxplots and histograms of the numerical features
![Numerical Features](/Images/output.png)

As for the categorical features, here are a few of them that is below a certain number of unique values
![Categorical Features](/Images/categorical-distribution.png)

### 2. Models

Developed base classification algorithms: Support Vector Machines (SVC), Logistic Regression, K-Nearest Neighbors, Decision Tree / Random Forest, Artificial Neural Networks, Ensemble.

### 3. Pipeline

#### Polynomial Feature Expansion with PCA

A specialized pipeline is designed for feature engineering, combining polynomial feature expansion, standardization, and PCA (Principal Component Analysis). This pipeline is particularly tailored for enhancing the performance of a classification model. Below is a breakdown of the key components:

#### Pipeline Components:

**Polynomial Feature Expansion:**

The pipeline initiates with polynomial feature expansion using PolynomialFeatures with a degree of 2.

**Numerical Feature Standardization:**

StandardScaler is employed to standardize numerical features, ensuring consistent scaling for optimal model training.

**Principal Component Analysis (PCA):**

A dimensionality reduction step is applied using `PCA to retain 99%` of the variance in the data.

**Categorical Feature Encoding:**

Categorical features are processed using OneHotEncoder to transform them into a machine-learning-compatible format. The handle_unknown='ignore' parameter is employed to gracefully handle unknown categories during encoding.

**Classification Model:**

A `classification` model is incorporated into the pipeline.

#### Pipeline Execution:

**Initialization:**

Components such as polynomial feature expansion, numerical feature scaler, PCA, categorical feature encoder, and the classification model are initialized.

**Column Transformation:**

The ColumnTransformer orchestrates the application of appropriate preprocessing steps to numerical and categorical features.

**Model Training:**

The pipeline is trained on the provided training data, adapting to the classification model.

**Model Prediction:**

The trained model is employed to make predictions on new data, generating outcome predictions.

#### Execution of the Pipeline:

To execute this specialized pipeline, ensure the necessary Python packages are installed. Execute the provided code within the relevant environment, adjusting the pipeline or model parameters as needed. This pipeline is crafted to enhance the performance of a classification model through advanced feature engineering.

### 4. Hyperparameter Optimization

Model-specific search spaces were carefully defined and fed into GridSearches to facilitate hyperparameter fine-tuning. Performance metrics were compared, and potential improvements and model-specific
issues were discussed in detail.

### 5. Feature Engineering

Incorporated Polynomial Features and K-Means Clusters Feature based on model interpretation. Adjusted the main pipeline to seamlessly integrate the feature engineering function.

### 6. Model Interpretation

Derived insights into feature importance through model-specific methodologies, including analyses of feature importance, and decision tree visualizations.

## Models' Performance Metrics

|               | Support Vector Machine | K-Nearest Neighbors | Decision Tree | Logistic Regression | Random Forest | ANN    |
| :------------ | ---------------------- | ------------------- | ------------- | ------------------- | ------------- | ------ |
| **F1 Score**  | 37.53%                 | 22.69%              | 35.87%        | 43.04%              | 43.48%        | 37.73% |
| **Recall**    | 41.25%                 | 20.92%              | 27.07%        | 42.78%              | 42.20%        | 28.72% |
| **Precision** | 82.29%                 | 24.78%              | 53.13%        | 43.30%              | 44.85%        | 54.98% |
| **Accuracy**  | 34.42%                 | 81.62%              | 87.52%        | 85.40%              | 85.86%        | 87.78% |

## Choosing the Best Model

The best-performing model is `Random Forest Classifier`. It's tuned toward the F1 score

The confusion matrix of best scoring model is:

![Confusion Matrix](/Images/Best-Model-RF.png)

## Model Interpretation

An evaluation of the constructed predictive model has been conducted using various performance metrics, aiming to ascertain its effectiveness in classifying auctioned vehicles into categories of good or bad buys. The model's performance has been quantified using several metrics, defined as follows:

- F1 Score: A metric that is measured to balance precision and recall, providing a single performance indicator, especially valuable when dealing with imbalanced classes.
- Recall: A metric reflecting the ability of the model to accurately identify and label the relevant (positive) cases.
- Precision: A representation of the proficiency of the model to ensure the relevancy of labeled cases.
- Accuracy: A metric showing the proportion of total predictions (both positive and negative) that were determined to be correct.
  Subsequent to the performance evaluation, an analysis of feature importance was performed to identify the most influential features in the predictions made by the model.

## Feature Importance

![Feature Importance](/Images/Feature-Importance.png)

## Repository description

- how to install
- files description

### Data Set:

Obtained from: [kaggle.com](https://www.kaggle.com/c/DontGetKicked/data)
