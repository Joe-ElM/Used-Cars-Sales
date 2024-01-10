# Used-Vehicle-Auctions!

## Project Overview

The objective of this project is to predict the likelihood of a car purchased at an auction being a "kick" (bad buy). This prediction can aid auto dealerships in implementing cost-effective retention strategies to prevent undesirable purchases and potential losses.

## Dataset

[Kaggle - competetion !](https://www.kaggle.com/c/DontGetKicked/data)

- **Structured:** Yes
- **Format:** Single .csv file
- **Size:** [Insert size] observations (unique: [Insert unique count])
- **Number of Features:** [33]
- **Target Feature (Vector):** `IsBadBuy`, imbalanced, [12:88]

- **Duplicates:** [0]

## Problem Space

- Binary classification problem
- Modertate size dataset size for ML model training
- Imbalanced target class distribution
- Robust model development with independent performance metrics
- Custom scoring using F1 score (same emphasis on recall and precision)

## Process

### 1. Exploratory Data Analysis (EDA)

The EDA.ipynb file encapsulates the process of Exploratory Data Analysis (EDA), where I thoroughly explore and analyze datasets to extract valuable insights and discern patterns

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

A dimensionality reduction step is applied using PCA to retain 99% of the variance in the data.

**Categorical Feature Encoding:**

Categorical features are processed using OneHotEncoder to transform them into a machine-learning-compatible format. The handle_unknown='ignore' parameter is employed to gracefully handle unknown categories during encoding.

**Classification Model:**

A classification model is incorporated into the pipeline.

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

Incorporated Polynomial Features and K-Means Cluster Features based on model interpretation. Adjusted the main pipeline to seamlessly integrate the feature engineering function.

### 6. Model Interpretation

Derived insights into feature importance through model-specific methodologies, including analyses of feature importance, and decision tree visualizations.

## Models' Performance Metrics

|               | Support Vector Machine | K-Nearest Neighbors | Decision Tree      | Logistic Regression | Random Forest      | XGBoost            |
| :------------ | ---------------------- | ------------------- | ------------------ | ------------------- | ------------------ | ------------------ |
| **F2 Score**  | [Insert score]         | [Insert score]      | [Insert score]     | [Insert score]      | [Insert score]     | [Insert score]     |
| **Recall**    | [Insert recall]        | [Insert recall]     | [Insert recall]    | [Insert recall]     | [Insert recall]    | [Insert recall]    |
| **Precision** | [Insert precision]     | [Insert precision]  | [Insert precision] | [Insert precision]  | [Insert precision] | [Insert precision] |

## Choosing the Best Model

The best-performing model is [Insert Best Model]. It's tuned toward the F2 score, with more weight on recall, aligning with the project's goal.

## Model Interpretation

### Feature Importance

[Insert Feature Importance Visualization]

### Retention Strategies

Address observations listed in the Feature Importance section, prioritizing from top to bottom. Suggestions provided for specific features.

## Project Highlights

- Shared files for functions and global settings
- Main pipeline with external model-definitions .py file
- Advanced confusion matrix with meaningful color-coding
- Bulk generation of data visualizations for streamlined EDA

## Repository Description

- How to Install
- Files Description

# Auction Car Purchses!

## Project Overview

In this project, I aim to predict the likelihood of a car turning into a "kick" (a problematic purchase) using machine learning classification algorithms. The primary goal is to assist auto dealerships in making informed decisions, mitigating potential losses, and implementing effective retention strategies.

## Dataset

[Kaggle - Don't Get Kicked!](https://www.kaggle.com/c/DontGetKicked/data)

- **Structure:** Tabular (CSV)
- **Size:** [65620] records
- **Features:** [PurchDate Auction VehYear VehicleAge Make Model Trim SubModel Color Transmission WheelTypeID WheelType VehOdo Nationality Size TopThreeAmericanName MMRAcquisitionAuctionAveragePrice MMRAcquisitionAuctionCleanPrice MMRAcquisitionRetailAveragePrice MMRAcquisitonRetailCleanPrice MMRCurrentAuctionAveragePrice MMRCurrentAuctionCleanPrice MMRCurrentRetailAveragePrice MMRCurrentRetailCleanPrice PRIMEUNIT AUCGUART BYRNO VNZIP1 VNST VehBCost IsOnlineSale WarrantyCost]
- **Target Feature:** `IsBadBuy`, imbalanced, [12/88]
-
- **Duplicates:** [No dublicates]

## Problem Space

- Binary classification challenge
- Limited dataset size for model training
- Imbalanced distribution of target classes
- Focus on building a robust model with unbiased performance metrics
- Custom scoring using F2 score, emphasizing recall over precision

## Project Journey

### 1. Establishing Conventions

I set coding standards, variable naming, randomization parameters, JupyterLab notebook structure, and Git repository organization for consistency and efficient collaboration.

### 2. Exploratory Data Expedition (EDE)

I independently conducted EDE to foster unbiased insights. Insights were shared, discussed, and integrated for comprehensive analysis.

### 3. Model Expedition

I developed foundational classification algorithms: Support Vector Machines (SVC), Logistic Regression, K-Nearest Neighbors, Decision Tree / Random Forest. Considered additional models for future exploration.

### 4. Model Expedition Pipeline

I created the main pipeline while ensuring flexibility for model-specific variations introduced later.

### 5. Fine-Tuning Hyperparameters

I defined model-specific search spaces and executed GridSearches. Evaluated and compared performance metrics, fostering insightful discussions for improvement.

### 6. Decoding Model Insights

I interpreted feature importance utilizing model-specific methods such as feature importance, permutation feature importance, and decision tree visualization.

### 7. Feature Evolution

I generated new features based on model interpretations. Adapted the main pipeline to accommodate evolving feature engineering needs.

## Performance Metrics

| Model               | F2 Score       | Recall          | Precision          |
| :------------------ | -------------- | --------------- | ------------------ |
| SVM                 | [Insert score] | [Insert recall] | [Insert precision] |
| K-NN                | [Insert score] | [Insert recall] | [Insert precision] |
| Decision Tree       | [Insert score] | [Insert recall] | [Insert precision] |
| Logistic Regression | [Insert score] | [Insert recall] | [Insert precision] |
| Random Forest       | [Insert score] | [Insert recall] | [Insert precision] |
| XGBoost             | [Insert score] | [Insert recall] | [Insert precision] |

## Optimal Model Selection

The most effective model, considering the F2 score, is [Insert Best Model]. This model is finely tuned for recall, aligning perfectly with my project's mission.

## Interpreting Model Insights

### Feature Impact

[Insert Visualization of Feature Impact]

### Strategies for Retention

I will address key observations from the feature impact analysis. Prioritize initiatives based on impact, considering unique challenges and strengths of each feature.

## Project Highlights

- Shared files promoting consistency and efficient collaboration
- Main pipeline adaptable to model-specific variations
- Advanced confusion matrix with purposeful color-coding
- Automated data visualization for streamlined EDE

## Repository Guide

- Installation instructions
- File descriptions
