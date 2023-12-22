package ma.enset;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.evaluation.ClusteringEvaluator;
import org.apache.spark.ml.feature.MinMaxScaler;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class KMeanApp {
    public static void main(String[] args) {
        Logger.getLogger("org").setLevel(Level.ERROR);
        Logger.getLogger("akka").setLevel(Level.ERROR);

        SparkSession spark = SparkSession
                .builder()
                .appName("LinearRegressionModel Spark")
                .master("local[*]")
                .getOrCreate();
        Dataset<Row> dataset = spark.read().option("inferSchema",true).option("header",true).csv("Mall_Customers.csv");
        VectorAssembler vectorAssembler = new VectorAssembler().setInputCols(new String[]{"Age","Annual Income (k$)","Spending Score (1-100)"}).setOutputCol("Feautres");
        Dataset<Row> assambledDS = vectorAssembler.transform(dataset);
        MinMaxScaler scaler = new MinMaxScaler().setInputCol("Feautres").setOutputCol("normalizeFeautres");

        Dataset<Row> normalizedDS = scaler.fit(assambledDS).transform(assambledDS);
        normalizedDS.printSchema();
        KMeans kMeans = new KMeans().setK(5).setSeed(123).setFeaturesCol("normalizeFeautres").setPredictionCol("prediction");
        KMeansModel model = kMeans.fit(normalizedDS);
        Dataset<Row> predictions = model.transform(normalizedDS);
        predictions.show(20);
        ClusteringEvaluator evaluator=  new ClusteringEvaluator();
        double score = evaluator.evaluate(predictions);
        System.out.println(score);

    }


}