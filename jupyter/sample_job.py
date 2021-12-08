## Import Libraries
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.regression import LinearRegression, LinearRegressionModel

sc = SparkContext()
spark = SparkSession(sc)

## Load model
lrModel = LinearRegressionModel.load('gs://spark-training-data/ml_models/sample_model.model')

## Read in the data from model_test_jc
df = spark.read.format('libsvm').load('gs://spark-training-data/datasets/sample_linear_regression_data.txt')
df.show(5)

## Predict Results
predictions = lrModel.transform(df)
predictions.show(5)
