# Cirrhosis Patient Survival Prediction

This repository contains the code and analysis for predicting patient survival status (death or censored) using the **Cirrhosis Patient Survival** dataset from the UCI Machine Learning Repository. This project explores the predictive power of clinical and laboratory measurements using **Decision Trees** and **Naive Bayes** models, providing insights into patient outcomes that can guide clinical decision-making.

## 📁 Project Structure

```
Cirrhosis-Patient-Survival-Prediction/
│
├── data/
│   └── cirrhosis.csv
│
├── scripts/
│   └── cirrhosis_classification.R
│
├── results/
│   └── decision_tree.png
|   └── cv_decision_tree.png
│   └── naive_bayes.png
|   └── cv_naive_bayes.png
│   └── correlation_plot.png
│   └── model_accuracy_comparison.png
│
├── report/
|   └── Cirrhosis Patient Survival Prediction.png
|
├── README.md
│
└── LICENSE
```

## 🚀 Project Overview

Chronic liver cirrhosis is a serious, irreversible condition that can lead to liver failure and death. Early prediction of patient outcomes can assist in personalized treatment planning and risk assessment. This project aims to classify patient survival status using machine learning techniques, focusing on the following objectives:

* Data cleaning and preprocessing for reliable model training
* Exploratory data analysis to identify key features
* Building and tuning decision tree and naive Bayes classifiers
* Evaluating model performance using cross-validation and test data

## 📊 Dataset

The dataset contains **418 patient records** with **17 features** including:

* Demographic data (Age, Sex)
* Clinical observations (Ascites, Hepatomegaly, Spiders, Edema)
* Laboratory measurements (Bilirubin, Cholesterol, Albumin, Copper, Platelets, Prothrombin)
* Outcome status (Censored, Death)

The original dataset can be found here: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/878/cirrhosis+patient+survival+prediction+dataset-1)

## 🔄 Data Preprocessing

Key preprocessing steps included:

* **Column Removal**: Dropped non-predictive ID and N\_Days columns
* **Label Encoding**: Converted categorical variables to factors
* **Data Cleaning**: Removed patients with liver transplants (Status = 'CL')
* **Imputation**: Replaced missing values with column means for critical variables
* **Log Transformation**: Applied to numeric variables to reduce variance and improve model stability

## 📐 Model Training and Evaluation

### Decision Tree

* Built using the `rpart()` function
* Tuned with cross-validation to optimize the complexity parameter (cp)
* Achieved **84.75% accuracy** on the test set (cp = 0.001)

### Naive Bayes

* Implemented using `naiveBayes()`
* Tuned with grid search to avoid zero-probability issues
* Achieved **81.36% accuracy** on the test set

### Cross-Validation

* 10-fold cross-validation was conducted to find the optimal hyperparameters for both models

## 📊 Results

* Decision Tree (cp = 0.001): **84.75% Accuracy**
* Naive Bayes: **81.36% Accuracy**

## 📈 Key Insights

* Decision trees provided higher overall accuracy, indicating better performance for this dataset.
* Naive Bayes models showed higher sensitivity but more false positives, suggesting they may be more prone to overfitting.
* Log transformation of numeric variables significantly improved both models' stability.

## 📅 Future Improvements

* Explore ensemble methods like Random Forests or Gradient Boosting
* Include additional clinical features for richer analysis
* Consider feature engineering and more advanced hyperparameter tuning

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 💬 Acknowledgments

Special thanks to Amrutha Karuturi for guidance on Laplace smoothing and log transformations. AI tools were used to improve explanations and debug R code.

## 📚 References

* [NIDDK - Definition & Facts for Cirrhosis](https://www.niddk.nih.gov/health-information/liver-disease/cirrhosis/definition-facts)
* Baki, J. A., & Tapper, E. B. (2019). Contemporary Epidemiology of Cirrhosis. *Current Treatment Options in Gastroenterology*. [https://doi.org/10.1007/s11938-019-00228-3](https://doi.org/10.1007/s11938-019-00228-3)
