# Don't Get Kicked!

## Project Overview

The objective of this project is to predict the likelihood of a car purchased at an auction being a "kick" (bad buy). This prediction can aid auto dealerships in implementing cost-effective retention strategies to prevent undesirable purchases and potential losses.

## Dataset

[Kaggle - Don't Get Kicked!](https://www.kaggle.com/c/DontGetKicked/data)

- **Structured:** Yes
- **Format:** Single .csv file
- **Size:** [Insert size] observations (unique: [Insert unique count])
- **Number of Features:** [Insert feature count]
- **Target Feature (Vector):** `IsBadBuy`, imbalanced, [Insert class distribution]
- **NA Values:** [Insert count] / [Insert total observations]
- **Duplicates:** [Insert duplicate count]

## Problem Space

- Binary classification problem
- Small dataset size for ML model training
- Imbalanced target class distribution
- Robust model development with independent performance metrics
- Custom scoring using F2 score (more weight on recall than precision)

## Process

### 1. Project Conventions

Established coding conventions, variable names, random parameters, JupyterLab structure, and Git repository folder structure to ensure consistent and unbiased model performance comparisons.

### 2. Exploratory Data Analysis (EDA)

Independently conducted EDA by each team member to prevent bias. Insights exchanged and compared for comprehensive analysis.

### 3. Models

Developed base classification algorithms: Support Vector Machines (SVC), Logistic Regression, K-Nearest Neighbors, Decision Tree / Random Forest. Additional models (Artificial Neural Networks, XGBoost, Adaboost, Gradient Boost, VotingClassifier) considered for later development.

### 4. Pipeline

Main pipeline created by one team member while other models followed its JupyterNotebook structure.

### 5. Hyperparameters Fine-Tuning

Defined model-specific search spaces and fed them into GridSearches. Compared performance metrics and discussed potential improvements and model-specific issues.

### 6. Model Interpretation

Interpreted feature importance using model-specific methods (feature importance, permutation feat. importance, decision tree visualization).

### 7. Feature Engineering

Developed new features based on model interpretation. Main pipeline readjusted to accommodate feature engineering function.

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
