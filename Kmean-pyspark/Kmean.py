import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator


os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-amd64"

spark = SparkSession.builder.appName("PySPARK KMeans Clustering").master("local[*]") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

dataset = spark.read.option("inferSchema", True).option("header", True).csv("Mall_Customers.csv")
vectorAssembler = VectorAssembler(inputCols=["Age", "Annual Income (k$)", "Spending Score (1-100)"], outputCol="Features")
assambledDataset = vectorAssembler.transform(dataset)