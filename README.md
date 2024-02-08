# deep-learning-challenge

## Background Information
The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. This project uses neural network model to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.
![image](https://github.com/gialiubc/credit-risk-classification/assets/141379548/987401cc-0c1a-4181-91c0-2e71e7290f6b)

## Summary Report

### Overview of the Analysis
* The purpose of this analysis is to create a neural network model to predict whether applicants will be successful if funded by Alphabet Soup.

* The dataset contains 77,535 instances and 8 features, the features includes:

  * 0   loan_size         77536 non-null  float64 ---the total amount of the loan
  * 1   interest_rate     77536 non-null  float64 ---the interest rate
  * 2   borrower_income   77536 non-null  int64   ---the borrower's annual income
  * 3   debt_to_income    77536 non-null  float64 ---the ratio of debt to income
  * 4   num_of_accounts   77536 non-null  int64   ---the total number of accounts
  * 5   derogatory_marks  77536 non-null  int64   ---a negative item, such as a late payment or foreclosure. 
  * 6   total_debt        77536 non-null  int64   ---total amount of debt
  * 7   loan_status       77536 non-null  int64   ---0 for healthy loan and 1 for high-risk loan

  Given the information above, we want to predict the loan_status of a future entry.

* In this model, `value_counts` will be the prediction factor `y` and has a value of 0 and 1, whereas 0 represents healthy loan and 1 represents high-risk loan.

* The analysis is conducted in the following stages:
  * Stage 1, we prepare the datasets. We label the `value_counts` column as `y` and the rest of the features as `X`. We check for non NAN values and how the data is stratified on `y`. Then, we split the dataset into training set and testing set;
  * Stage 2, we set up the prediction model. Since we want to perform a classification model, we choose Logistic Regression in this case. We set up the model with `solver="lbfgs" and  random_state=1`. We fit in the training dataset `X_train` and `y_train`. Then, we make prediction base on `X_test`;
  * Stage 3, we evaluate the model's performance. We calculate the `balanced_accuracy_score`, generate the `confusion_matrix`, and print the `classification_report`;
  * Stage 4, we resampled training data using RandomOverSampler. Since we are dealing with imbalanced data, we oversample the minority class to reduce class imbalance. We repeat stage 1 to 3 using the resampled data and evaluate the model's perfromance.

### Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1: Logistic Regression 
  * Description of Model 1 Accuracy, Precision, and Recall scores.
  
    ![image](https://github.com/gialiubc/credit-risk-classification/assets/141379548/edd532e0-677c-4b56-8e67-4b1e9a6767ba)

    * The balanced accuracy score of Model 1 is 0.952;
    * The precision rate is 1.00 for `0` and 0.85 for `1`;
    * The recall rate is 0.99 for `0` and 0.91 for `1`.
    
* Machine Learning Model 2: Logistic Regression with Resampled Data
  * Description of Model 2 Accuracy, Precision, and Recall scores.

    ![image](https://github.com/gialiubc/credit-risk-classification/assets/141379548/6760e9ae-92c7-43ab-bfa4-76458a97ca0e)
  
    * The balanced accuracy score of Model 2 is 0.994;
    * The precision rate is 1.00 for `0` and 0.84 for `1`;
    * The recall rate is 0.99 for `0` and 0.99 for `1`.

### Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Our purpose is to accurately detect high-risk loans, so we care more about the return of false negative than false positive, because we don't want to overlook any risky loan. As we can see from the performance evaluation results, Model 2 performs better than Model 1 with higher recall rate, indicating lower false negative prediction. However, we need to take into account the set back of resampling the data, which can cause overfitting by replicating observations from the minority class. Therefore, it is hard to say which model is better at this point.
* In order to have a better call on which model performs better with the given dataset, we should conduct a validation test to see if the model is overfitting. Also, we should standardize the dataset in order to have the values in the similar scales. In conclusion, we shouldn't deploy the model at this point.

## File Paths

- Code: credit_risk_classification.ipynb
- Data: Resources ---> lending_data.csv

## Notes

This data analysis is conducted as an individual project by Bochao Gia Liu. 

## References & Data Source

1. Investigating the effects of resampling imbalanced datasets with data validation techniques: https://medium.com/geekculture/investigating-the-effects-of-resampling-imbalanced-datasets-with-data-validation-techniques-f4ca3c8b2b94
2. Image: guide_risk-management-banking_featured-img_1127x340 https://reciprocity.com/wp-content/uploads/2022/05/guide_risk-management-banking_featured-img_1127x340.jpg
3. Data Source: data for this dataset was generated/provided by edX Boot Camps LLC, and is intended for educational purposes only.
