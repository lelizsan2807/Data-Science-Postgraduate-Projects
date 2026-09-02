"""
# EasyVisa Case Status Prediction

## Project Overview

This project focuses on predicting the outcome of visa applications using machine learning classification models. The objective is to identify whether an EasyVisa application will be **Certified** or **Denied** based on applicant, employer, job, and application characteristics.

The project follows a complete machine learning workflow, including data preparation, exploratory analysis, model development, class imbalance treatment, hyperparameter tuning, model comparison, evaluation, and interpretation of the final model.

The primary target variable is `case_status`, which is transformed into a binary classification problem:

- **1 = Certified**
- **0 = Denied**

Because both false positives and false negatives are important in this business problem, **F1-score** is used as the primary model evaluation metric. Accuracy, precision, recall, and confusion matrices are also considered to provide a more complete assessment of model performance.

---

## Dataset

The dataset contains information related to visa applications and includes variables describing characteristics such as:

- Applicant education
- Applicant experience
- Job requirements
- Offered wage
- Employer information
- Employer size
- Geographic location
- Application characteristics
- Visa case status

The categorical variables were converted into numerical features using one-hot encoding with `drop_first=True`.

The data was divided into three stratified datasets:

- **Training set:** 15,288 observations
- **Validation set:** 5,096 observations
- **Test set:** 5,096 observations

Stratification was used to preserve the original proportion of Certified and Denied cases across the datasets.

---

## Data Preparation

The data preparation process included:

1. Identifying the target variable.
2. Converting `case_status` into a binary variable.
3. Separating predictor variables from the target.
4. Encoding categorical variables using one-hot encoding.
5. Splitting the dataset into training, validation, and test sets using stratification.
6. Evaluating the class distribution.
7. Applying class imbalance techniques to the training data.

The training data contains approximately:

- **66.78% Certified cases**
- **33.22% Denied cases**

Since the classes are not perfectly balanced, both oversampling and undersampling were evaluated to determine whether balancing the training data would improve model performance.

---

## Model Development

Five classification models were initially developed and compared:

1. Bagging Classifier
2. Random Forest Classifier
3. AdaBoost Classifier
4. Gradient Boosting Classifier
5. XGBoost Classifier

The models were first evaluated using the original training data. Additional experiments were then performed using SMOTE oversampling and Random Under Sampling.

This approach allowed the project to compare not only different algorithms, but also different approaches to handling class imbalance.

---

## Baseline Model Comparison

The baseline models produced different levels of performance on the validation dataset.

| Model | Training F1 | Validation F1 |
|---|---:|---:|
| Bagging | 0.9892 | 0.7737 |
| Random Forest | 1.0000 | 0.8050 |
| AdaBoost | 0.8204 | 0.8180 |
| Gradient Boosting | 0.8291 | **0.8266** |
| XGBoost | 0.8949 | 0.8104 |

Gradient Boosting achieved the highest validation F1-score among the baseline models. Random Forest achieved a perfect training F1-score but a lower validation F1-score, which indicates that the model was likely overfitting the training data.

The comparison demonstrates why validation performance is more useful than training performance when selecting a model.

---

## Class Imbalance Experiments

### SMOTE Oversampling

SMOTE was applied to the training data to create a balanced dataset.

Before SMOTE:

- Certified: 10,210
- Denied: 5,078

After SMOTE:

- Certified: 10,210
- Denied: 10,210

The validation F1-scores after oversampling were:

| Model | Validation F1 |
|---|---:|
| Bagging | 0.7665 |
| Random Forest | 0.7965 |
| AdaBoost | **0.8195** |
| Gradient Boosting | 0.8173 |
| XGBoost | 0.8129 |

SMOTE did not improve validation performance compared with the original training data for the strongest models. Therefore, balancing the classes through synthetic oversampling was not sufficient to produce a better final model.

### Random Undersampling

Random undersampling reduced the majority class to match the minority class.

After undersampling:

- Certified: 5,078
- Denied: 5,078

The validation F1-scores were:

| Model | Validation F1 |
|---|---:|
| Bagging | 0.7057 |
| Random Forest | 0.7417 |
| AdaBoost | 0.7660 |
| Gradient Boosting | **0.7766** |
| XGBoost | 0.7459 |

Undersampling resulted in lower validation performance across all models. This suggests that removing a substantial number of Certified observations caused the models to lose useful information.

Based on these experiments, the original training data provided the strongest overall validation performance.

---

## Hyperparameter Tuning

After comparing the baseline models and class imbalance strategies, four models were selected for hyperparameter tuning:

- Random Forest
- AdaBoost
- Gradient Boosting
- XGBoost

### Why These Four Models Were Tuned

These models were selected because they demonstrated stronger validation performance and/or provided meaningful potential for improvement through hyperparameter optimization.

Random Forest was selected because it had strong baseline validation performance and demonstrated high predictive capacity, although its perfect training F1-score indicated potential overfitting.

AdaBoost and Gradient Boosting were selected because they showed relatively consistent training and validation performance and were competitive with the strongest baseline models.

XGBoost was selected because it achieved strong validation performance and provided an opportunity to optimize the balance between precision and recall through parameters such as `scale_pos_weight`, learning rate, subsampling, and regularization.

### Why Bagging Was Not Tuned

Bagging was not selected for hyperparameter tuning because its validation performance was consistently weaker than the other models:

- Original data: F1 = 0.7737
- SMOTE: F1 = 0.7665
- Undersampling: F1 = 0.7057

Although Bagging achieved a very high training F1-score, this did not translate into comparable validation performance. This difference suggested overfitting and made Bagging a lower-priority candidate for additional tuning.

The purpose of tuning was therefore to focus computational effort on models with stronger validation performance and greater potential to provide a competitive final solution.

---

## Hyperparameter Optimization Method

GridSearchCV with **5-fold cross-validation** was used for hyperparameter tuning.

The primary scoring metric was **F1-score**, consistent with the project's objective of balancing precision and recall.

The tuning process evaluated combinations of relevant model parameters rather than relying on a single default configuration.

---

## Random Forest

The Random Forest search evaluated:

- `max_depth`
- `max_features`
- `min_samples_split`
- `n_estimators`

The selected model used:

- `max_depth = 10`
- `min_samples_split = 5`
- `n_estimators = 30`

The tuned Random Forest achieved a validation F1-score of **0.8262**, improving over its baseline validation F1-score of **0.8050**.

Its training F1-score decreased from 1.0000 to approximately 0.8449, which is a positive sign because the model became less dependent on the training data while maintaining strong validation performance.

---

## AdaBoost

AdaBoost was tuned by varying:

- Decision tree depth
- Number of estimators
- Learning rate

The best configuration used:

- Decision tree depth = 3
- `n_estimators = 100`
- `learning_rate = 0.06`

The tuned model achieved:

- Training F1: **0.8235**
- Validation F1: **0.8231**

The relatively small difference between training and validation performance suggests that the tuned model generalizes consistently.

---

## Gradient Boosting

Gradient Boosting was tuned using:

- Number of estimators
- Subsample
- Maximum features
- Learning rate

The best configuration used:

- `n_estimators = 50`
- `subsample = 0.7`
- `max_features = 0.7`
- `learning_rate = 0.05`

The model achieved:

- Training F1: **0.8276**
- Validation F1: **0.8269**

The validation performance was slightly higher than the baseline Gradient Boosting result of 0.8266.

The very small difference between training and validation F1 also indicates stable generalization.

---

## XGBoost

XGBoost was tuned using:

- `n_estimators`
- `scale_pos_weight`
- `subsample`
- `learning_rate`
- `gamma`

The selected configuration included:

- `n_estimators = 100`
- `learning_rate = 0.1`
- `gamma = 1`

The tuned XGBoost model achieved:

- Training F1: **0.8464**
- Validation F1: **0.8218**

XGBoost produced the highest validation recall among the tuned models at approximately **0.9468**. This means it was particularly effective at identifying Certified cases, although its precision was lower than the other leading models.

This demonstrates why F1-score was used instead of recall alone: maximizing recall can increase the number of correctly identified positive cases while also increasing false positives.

---

## Final Tuned Model Comparison

The final validation results were:

| Model | Accuracy | Recall | Precision | F1 |
|---|---:|---:|---:|---:|
| Random Forest | 0.7498 | 0.8904 | 0.7707 | 0.8262 |
| AdaBoost | 0.7494 | 0.8725 | 0.7789 | 0.8231 |
| Gradient Boosting | 0.7490 | 0.8975 | 0.7666 | **0.8269** |
| XGBoost | 0.7257 | **0.9468** | 0.7259 | 0.8218 |

Gradient Boosting achieved the highest validation F1-score at **0.8269**, making it the numerical winner when F1-score is used as the sole selection criterion.

Random Forest was extremely competitive, with an F1-score of **0.8262**, while also achieving slightly higher validation accuracy and precision than Gradient Boosting.

Therefore, the difference between the two leading models is very small. The final model selection should consider the business importance of the different types of classification errors rather than relying only on a single metric.

---

## Model Evaluation

The models were evaluated using:

### Accuracy

Measures the overall proportion of correctly classified observations.

### Precision

Measures the proportion of predicted positive cases that were actually positive.

### Recall

Measures the proportion of actual positive cases that were correctly identified.

### F1-Score

The harmonic mean of precision and recall.

F1-score is particularly useful for this project because both false positives and false negatives have meaningful consequences.

### Confusion Matrix

Confusion matrices were also used to examine the number of:

- True Positives
- True Negatives
- False Positives
- False Negatives

This provides additional insight into how each model makes classification errors.

---

## Key Findings

Several important findings emerged from the model comparison:

1. **Gradient Boosting was the strongest baseline model** based on validation F1-score.
2. **Random Forest showed signs of overfitting** in its baseline configuration because it achieved a perfect training F1-score but substantially lower validation performance.
3. **SMOTE did not improve overall validation F1-score** for the strongest models.
4. **Random undersampling produced the weakest validation performance**, suggesting that removing majority-class observations discarded useful information.
5. **Hyperparameter tuning improved Random Forest substantially**, increasing validation F1 from 0.8050 to 0.8262.
6. **Gradient Boosting remained highly competitive after tuning**, achieving the highest validation F1-score of 0.8269.
7. **XGBoost achieved the highest recall**, but this came with lower precision and therefore did not produce the highest F1-score.
8. **Bagging was excluded from tuning** because its validation performance was consistently lower than the other candidate models despite very high training performance.

---

## Business Insights

The analysis also identified several characteristics associated with visa certification outcomes.

The exploratory analysis and model interpretation indicate that factors such as:

- Education level
- Years of experience
- Offered wage
- Employer size
- Employer geographic location

can contribute to differences in predicted visa outcomes.

The analysis suggests that higher education and greater professional experience can be associated with more favorable certification outcomes. Compensation and employer characteristics also appear to contribute to the model's predictions.

Geographic patterns were also observed, with differences across regions such as the Midwest and South.

These findings can help organizations better understand which applicant and employer characteristics are associated with visa certification outcomes.

---

## Conclusion

This project demonstrates an end-to-end machine learning classification workflow for predicting EasyVisa case outcomes.

Five ensemble learning algorithms were initially compared, followed by experiments with SMOTE oversampling and random undersampling. The results showed that the original training data produced the strongest overall validation performance.

Four models were then selected for hyperparameter tuning based on their competitive baseline performance and potential for improvement. GridSearchCV with 5-fold cross-validation and F1-score optimization was used to identify stronger model configurations.

The final comparison showed that Gradient Boosting achieved the highest validation F1-score at **0.8269**, with Random Forest very close behind at **0.8262**.

Overall, the analysis emphasizes the importance of evaluating models based on validation performance, understanding overfitting, comparing multiple evaluation metrics, and selecting models according to the business consequences of classification errors.
