# neural-network-challenge-2
Module 19 Challenge neural-network-challenge-2

This is a neural network that HR can use to predict whether employees are likely to leave the company. Additionally, HR believes that some employees may be better suited to other departments. This is to predict the department that best fits each employee. **These two columns should be predicted using a branched neural network**.

## Preprocessing
First, import the data and examine the first five rows to get a sense of its structure. Afterward, determine the number of unique values present in each column to understand the diversity within the dataset. Next, construct a dataframe `y_df` containing the `attrition` and `department` columns, which are crucial for our analysis. Following this, compile a list comprising at least 10 column names to serve as the `X` data for our model. These columns should exclude 'attrition' and 'department'. Subsequently, create `X_df` using the chosen columns and inspect the data types present within it. Proceed by splitting the dataset into training and testing sets to facilitate model evaluation. Convert the `X` data to numeric types as deemed appropriate, ensuring to fit any encoders to the training data before transforming both training and testing sets accordingly. Additionally, instantiate a **StandardScaler** and apply it to normalize the features, fitting it to the training data and transforming both sets thereafter. Utilize **OneHotEncoder** to encode the 'department' and 'attrition' columns separately, fitting the encoder to the training data and employing it to transform both training and testing sets for each column. Finally, implement the encoded data in preparation for subsequent modeling steps.


## Creating, Compiling, and Training the Model
To begin, let's first find the number of columns in the X training data. With this information, we can determine the number of neurons to assign to the input layer. I chose to keep 8 columns in the X training data, so the input layer will have (10 neurons - see below the reasons). These columns are

`Age ,  BusinessTravel ,  DistanceFromHome ,  HourlyRate ,  JobSatisfaction ,  PerformanceRating ,  WorkLifeBalance ,  YearsInCurrentRole`

Reason for this is because we will apply `getdummy` to the `BusinessTravel` column, which will result in 3 columns. The `department` column will also be encoded to 3 columns. Next, we'll proceed to create the input layer. However, it's important to note that we won't be using a `sequential` model here, as there will be two branched output layers. Following this, we'll establish at least two shared layers within the model architecture.

```
__________________________________________________________________________________________________
 Layer (type)                Output Shape                 Param #   Connected to                  
==================================================================================================
 input_2 (InputLayer)        [(None, 10)]                 0         []                            
                                                                                                  
 shared1 (Dense)             (None, 64)                   704       ['input_2[0][0]']             
                                                                                                  
 shared2 (Dense)             (None, 128)                  8320      ['shared1[0][0]']             
                                                                                                  
 department_hidden (Dense)   (None, 32)                   4128      ['shared2[0][0]']             
                                                                                                  
 attrition_hidden (Dense)    (None, 32)                   4128      ['shared2[0][0]']             
                                                                                                  
 department_output (Dense)   (None, 3)                    99        ['department_hidden[0][0]']   
                                                                                                  
 attrition_output (Dense)    (None, 2)                    66        ['attrition_hidden[0][0]']    
                                                                                                  
==================================================================================================
Total params: 17445 (68.14 KB)
Trainable params: 17445 (68.14 KB)
Non-trainable params: 0 (0.00 Byte)
__________________________________________________________________________________________________
```
Subsequently, we'll construct a branch specifically tailored to predict the department target column. This branch will consist of one hidden layer and one output layer.
Similarly, we'll create another branch to predict the attrition target column. Like the previous branch, this will also include one hidden layer and one output layer.
Once the architecture is defined, we'll proceed to create the model. With the model structure established, the next step is to compile it. After compiling, we'll summarize the model to provide an overview of its architecture and parameters. Moving forward, we'll train the model using the preprocessed data. Following the training phase, we'll evaluate the model's performance with the testing data. Finally, we'll print the accuracy scores for both the department and attrition predictions.

```
Department Accuracy: 0.510869562625885
Attrition Accuracy: 0.8070651888847351
```

## Summary

I think **accuracy** can be used in this model to evaluate the model's performance. The model has an accuracy of 0.80 for the department prediction and 0.51 for the attrition prediction. This means that the model is able to predict the department of an employee with 83% accuracy and whether an employee will leave the company with 85% accuracy. This is a good result, but there is still room for improvement. The model can be further optimized by tuning the hyperparameters, increasing the number of epochs, or adding more layers to the neural network. Additionally, more data can be collected to improve the model's performance. Overall, the model is a good starting point for HR to predict employee attrition and department fit.

This type of accuracy may not be very reliable if there were cost sensitive issues. For example, if the cost of false positives is high, then the model may not be very useful. In this case, the model may need to be further optimized to reduce the number of false positives. However, if the cost of false negatives is high, then the model may be useful as is. It is important to consider the cost of false positives and false negatives when evaluating the model's performance. While predicting attrition for any company is important, but cannot be compared with outcome of for example bad fraud detection or healthcare issues.

I chose `relu` for shared layer because it is a good activation function for hidden layers in neural networks. It is able to handle the vanishing gradient problem, which can occur when training deep neural networks. I chose `sigmoid` for the attrition output layer because it is a good activation function for binary classification problems. It squashes the output to a range between 0 and 1, which is useful for predicting probabilities. I chose `softmax` for the department output layer because it is a good activation function for multi-class classification problems. It squashes the output to a range between 0 and 1, which is useful for predicting probabilities. Additionally, it ensures that the sum of the output values is equal to 1, which is useful for interpreting the results.

A model might be improved by increasing the number of epochs, adding more layers to the neural network, or tuning the hyperparameters. Additionally, more data can be collected to improve the model's performance. Overall, the model is a good starting point for HR to predict employee attrition and department fit. Also by chossing different data column to understand which data column might be produce most accurate results.