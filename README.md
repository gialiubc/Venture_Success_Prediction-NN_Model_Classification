# deep-learning-challenge

## Background Information
The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. This project uses neural network model to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.


![image](https://spectrum.ieee.org/media-library/a-photo-illustration-showing-an-artificial-network-and-computer-code.jpg?id=29854895&width=1200&height=701)

## Summary Report

### Overview of the Analysis
* The purpose of this analysis is to create a neural network model to predict whether applicants will![image](https://github.com/gialiubc/deep-learning-challenge/assets/141379548/12ec5d6e-305a-4b5d-9d5d-f1eb9b36ba93)
 be successful if funded by Alphabet Soup.

* The dataset contains 34,299 instances and 12 features, the features includes:
* ![image](https://github.com/gialiubc/deep-learning-challenge/assets/141379548/88a86487-3083-4be8-b1ec-eaf58af74d9c)


* In this model, `IS_SUCCESSFUL` will be the prediction factor `y` and has a value of 0 and 1, whereas 0 represents unsuccess and 1 represents success.

* The analysis is conducted in the following stages:
  * Stage 1, processing the data. We study the dataset, including the data type, the features and each feature's unique count etc. There are eight out of twelve object data type and we use `get_dummies` need to transfrom them into binary symbols. We also reduce the unique value of each feature by cutting and bining those as `other`. Then we concate the feature dummies together with the numerical features and create the new data frame. We label the `IS_SUCCESSFUL` column as `y` and the rest of the features as `X`. Then, we scale and split the dataset into training set and testing set;
  * 
  * Stage 2, we set up the neural network model. We set up the initial model as follow:
  *![image](https://github.com/gialiubc/deep-learning-challenge/assets/141379548/97678ca8-686d-440d-97af-a7f5b68e73d1)
  * we use `relu` and  `sigmoid` as activation function for hidden layer and output layer respectively. We fit and train the model with X scaled and y training set.
  * 
  * Stage 3, we evaluate the model's performance. We calculate the `model_loss` and the `model_accuracy`. 
  * 
  * Stage 4, we save the model.

### Results
* Data Preprocessing
  - What variable(s) are the target(s) for your model?
    - The target is `IS_SUCCESSFUL`.
  - What variable(s) are the features for your model?
    - The features are `STATUS`, `ASK_AMT`, `APPLICATION_TYPE`, `AFFILIATION`, `CLASSIFICATION`, `USE_CASE`, `ORGANIZATION`, `INCOME_AMT`, `SPECIAL_CONSIDERATIONS`.
  - What variable(s) should be removed from the input data because they are neither targets nor features?
    - `EIN` and `NAME` are removed.
* Compiling, Training, and Evaluating the Model
  - How many neurons, layers, and activation functions did you select for your neural network model, and why?
    - The first layer has 80 neurons with `relu` as activation function, because we want to checkout more connections between neurons and we use `relu` because it's simple and fast for non-linear model.
    - The second layer has 30 neurons with `relu` as activation function, because we want to focus on more weighted connections;
    - The output layer has 1 neuron with `sigmoid` as activation function, because we want a single output and it should be 0 or 1.
    - 
  - Were you able to achieve the target model performance?
    - No.
  - What steps did you take in your attempts to increase model performance?
    - We stratified the data when spliting. For the first attempt we add one more layer; the second attempt we search for the best hyperparameters; the third attempt we reduced cleaned the data by deleting outliers and bining for each feature.
      
### Model
* Initial NN Model: AlphabetSoupCharity 
  * The loss and accuracy results as follow:
  * ![image](https://github.com/gialiubc/deep-learning-challenge/assets/141379548/60bde007-93b3-4ffc-86d2-3d77fae6920e)
    * Epochs is set to 100, optimizer `adam`, loss `binary_crossentropy`
    * The accuracy is 72.44%.
  
### Optimize
* Optimize 1
  * Stratify + add one more layer
  * In this model, we changed two things one by one. We first add one more hidden layer and the result accuracy is slightly improved. Then we stratified when spliting the data and the accuracy increased from 72.44% to 73.11%. Therefore we see that stratifying the data improves the model more so we will implement this in further testing.
* Optimize 2
  * Stratify + random search hyperparameters
  * For the second attempt, we used `random_search_tuner` to optimize the hyperparameters. After 10 trials and 100 epochs each, we get the following result:
  * ![image](https://github.com/gialiubc/deep-learning-challenge/assets/141379548/a33b905b-6f97-46fc-9475-e41fef1c5a6b)
  * with the best parameters as follow:
  * ![image](https://github.com/gialiubc/deep-learning-challenge/assets/141379548/3b137413-6adb-4ffa-b4f0-a445145bd303)
  * As we can see the accuracy has increased to 73.57%.
 
* Optimize 3
  * Stratify + random search hyperparameters + reduce data
  * For the third attempt, we used `random_search_tuner` to optimize the hyperparameters, and also adjusted the data as follow:
    * Delete 4 outliers of the `ASK_AMOUNT`;
    * Delete the following features: `STATUS`, `SPECIAL_CONSIDERATIONS`;
    * Cut and bin the following features: `AFFLILIATION`, `USE_CASE`, `ORGANIZATION`, `INCOME_AMT`;
  * So now we have 34 dimensions and 34,295 instances in total, and the best model is as follow:
  * ![image](https://github.com/gialiubc/deep-learning-challenge/assets/141379548/0b56edf6-1e6f-4907-b2a4-bbc9dc76e5d9)
  * As we can see the accuracy has increased to 73.68%.


### Summary

* Our best attempt is 'Optimize 3' where we have an accuracy of 73.68%, followed by 'Optimize 2' with accuracy 73.57%, the last is 'Optimize 1' with accuracy 73.11%. All of them are better than the initial model with accuracy of 72.44%. 
* In this case, we could use a Support Vector Machine model to solve the problem. Since SVMs shine with small to medium-sized nonlinear datasets, it might give us a better result.

## File Paths

- Code for initial model: AlphabetSoupCharity.ipynb
- Code for Optimize 1: Optimize1_ASoup-AddLayer.ipynb
- Code for Optimize 2: Optimize2_ASoup_Optimize_RandomSearch.ipynb
- Code for Optimize 3: Optimize3_ASoup_ReduceData.ipynb

## Notes

This data analysis is conducted as an individual project by Bochao Gia Liu. 

## References & Data Source

1. Investigating the effects of resampling imbalanced datasets with data validation techniques: https://medium.com/geekculture/investigating-the-effects-of-resampling-imbalanced-datasets-with-data-validation-techniques-f4ca3c8b2b94
2. Image: https://spectrum.ieee.org/media-library/a-photo-illustration-showing-an-artificial-network-and-computer-code.jpg?id=29854895&width=1200&height=701
3. Data Source: data for this dataset was generated/provided by edX Boot Camps LLC, and is intended for educational purposes only.
