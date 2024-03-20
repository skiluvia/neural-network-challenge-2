# neural-network-challenge-2
Module 19 Challenge neural-network-challenge-2

This is a neural network that HR can use to predict whether employees are likely to leave the company. Additionally, HR believes that some employees may be better suited to other departments. This is to predict the department that best fits each employee. These two columns should be predicted using a branched neural network.

## Preprocessing
Firstly, import the data and examine the first five rows to get a sense of its structure. Afterward, determine the number of unique values present in each column to understand the diversity within the dataset. Next, construct a dataframe `y_df` containing the 'attrition' and 'department' columns, which are crucial for our analysis. Following this, compile a list comprising at least 10 column names to serve as the `X` data for our model. These columns should exclude 'attrition' and 'department'. Subsequently, create `X_df` using the chosen columns and inspect the data types present within it. Proceed by splitting the dataset into training and testing sets to facilitate model evaluation. Convert the `X` data to numeric types as deemed appropriate, ensuring to fit any encoders to the training data before transforming both training and testing sets accordingly. Additionally, instantiate a StandardScaler and apply it to normalize the features, fitting it to the training data and transforming both sets thereafter. Utilize OneHotEncoder to encode the 'department' and 'attrition' columns separately, fitting the encoder to the training data and employing it to transform both training and testing sets for each column. Finally, implement the encoded data in preparation for subsequent modeling steps.


## Creating, Compiling, and Training the Model
To begin, let's first find the number of columns in the X training data.

Next, we'll proceed to create the input layer. However, it's important to note that we won't be using a sequential model here, as there will be two branched output layers.

Following this, we'll establish at least two shared layers within the model architecture.

Subsequently, we'll construct a branch specifically tailored to predict the department target column. This branch will consist of one hidden layer and one output layer.

Similarly, we'll create another branch to predict the attrition target column. Like the previous branch, this will also include one hidden layer and one output layer.

Once the architecture is defined, we'll proceed to create the model.

With the model structure established, the next step is to compile it.

After compiling, we'll summarize the model to provide an overview of its architecture and parameters.

Moving forward, we'll train the model using the preprocessed data.

Following the training phase, we'll evaluate the model's performance with the testing data.

Finally, we'll print the accuracy scores for both the department and attrition predictions.
