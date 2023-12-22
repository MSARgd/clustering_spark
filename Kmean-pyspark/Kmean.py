import os
from pyspark.sql import SparkSession
from datetime import datetime

date_today = datetime.now().date()
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-amd64"

spark = SparkSession.builder.appName("PySPARK").master("local[*]") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")







