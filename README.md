bin/spark-submit --class org.apache.spark.examples.SparkPi \
    --master yarn-cluster \
    --num-executors 8 \
    --driver-memory 1g \
    --executor-memory 1g \
    --executor-cores 4 \
    --queue default \
    examples/jars/spark-examples*.jar \
    10
