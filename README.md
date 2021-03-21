# Create-and-Deploy-Machine-Learning-Model-using-AWS-SageMaker-for-Spam-Classification

Used XGBoost estimator in AWS Sagemaker to classify spam text stored in AWS S3 after applying TF-IDF transformation.

##  Data: Spam Messages file
 Each record represents a single message having two attributes/columns: 'text', 'label'

1. 'text' column consists of the text message

2. 'label' column indicates whether the message is a spam or not 


## Data Pre-Processing:
1. Transformed text into a sparse matrix of n-gram counts using Count Vectorizer

2. Performed TF-IDF transformation from a provided matrix of counts. using TF-IDF Transformer

Created the S3 bucket. Uploaded the pre-processed data as CSV file into AWS S3 buckets so that AWS Sagemaker Jupyter instance could access the data for model training.

## Data Modeling
Downloaded the data from the S3 bucket. Used AWS SageMaker in-built XGBoost algorithm to train the model using training dataset and classify the messages. Set up a session and created an instance of the XGB model (estimator) and defined hyperparameters for the same.
The job trained the model using gradient optimization on a ml.m4.xlarge instance.

## Evaluation
Deployed the model and predictions were made using batch transform (one-time batch inference job). Created a transformer object and initiated the job by executing the transform() method of the transformer object. Post this, we downloaded the predictions (test.out.csv) file created by the Sagemaker and evaluated the model performance using metrics like classification accuracy, precision, recall, etc.

## Clean up
Deleted all the AWS resources (notebook instances, model, endpoints, S3 buckets, Cloudwatch logs) utilized.
