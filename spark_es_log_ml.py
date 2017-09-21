# coding: utf-8
"""offline log entries analysis with spark on hadoop

"""

from __future__ import print_function

import sys
#import pandas
import time
import pyspark
from pyspark import SparkContext
from pyspark.ml.feature import IDF as MLIDF
from pyspark.ml.feature import HashingTF as MLHashingTF
from pyspark.mllib.clustering import KMeans
#from pyspark.mllib.linalg import SparseVector, VectorUDT
from pyspark.mllib.util import MLUtils
from pyspark.sql.functions import col, explode, udf
from pyspark.sql.types import DoubleType, IntegerType, StringType
import argparse

localtime = time.localtime()
strdate = time.strftime("%Y.%m.%d", localtime)
sc = SparkContext(appName="Log Analysis")
sqlContext = pyspark.sql.SQLContext(sc)

es_load_filter = """
{ "_source": ["source", "message"],
    "query": {
        "query_string": {
        "fields": [
            "message"
        ],
        "query": "exception",
        "analyze_wildcard": true
        }
    }
}
"""
es_load_conf = {
    "es.nodes": "node2",
    "es.port": "39200",
    "es.resource": "filebeat-" + strdate,
    "es.query": es_load_filter,
    "es.net.http.auth.user": "admin",
    "es.net.http.auth.pass": "Hz123456"
}

es_conf = {"es.nodes": "node2",
           "es.port": "39200",
           "es.resource": "log_ml-%s/doc" % strdate,
           "es.net.http.auth.user": "admin",
           "es.net.http.auth.pass": "Hz123456"}


def load_es():
    """Load data from ES
    """
    es_rdd_load = sc.newAPIHadoopRDD(
        inputFormatClass="org.elasticsearch.hadoop.mr.EsInputFormat",
        keyClass="org.apache.hadoop.io.NullWritable",
        valueClass="org.elasticsearch.hadoop.mr.LinkedMapWritable",
        conf=es_load_conf)
    es_df = sqlContext.createDataFrame(es_rdd_load)

    keys = (es_df
            .select(explode("_2"))
            .select("key")
            .distinct()
            .rdd.flatMap(lambda x: x)
            .collect())

    exprs = [col("_2").getItem(k).alias(k) for k in keys]
    es_df_exploded = es_df.select(*exprs)

    # select two fields
    lean_df = es_df_exploded[['source', 'message']]
    lean_df = (lean_df
               .rdd
               .map(lambda x: (x.source, x.message.split(" ")))
               .toDF()
               .withColumnRenamed("_1", "source")
               .withColumnRenamed("_2", "message"))
    return lean_df


def analysis(df):
    """ML in Spark
    """
    htf = MLHashingTF(inputCol="message", outputCol="tf")
    tf = htf.transform(df)
    idf = MLIDF(inputCol="tf", outputCol="idf")
    tfidf = idf.fit(tf).transform(tf)
    #tfidf.show(truncate=True)

    #sum_ = udf(lambda v: float(v.values.sum()), DoubleType())
    #res_df = tfidf.withColumn("idf_sum", sum_("idf"))
    res_df = MLUtils.convertVectorColumnsFromML(tfidf, 'idf')
    ml_dataset = res_df.rdd.map(lambda x: x.idf).collect()
    model = KMeans.train(sc.parallelize(ml_dataset), 5, 50)

    return res_df, model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', action='store', dest='es_index', help='elasticsearch index for input')
    parser.add_argument('-s', action='store', dest='save_es_index', help='elasticsearch index for save result')
    args = parser.parse_args()
    if args.es_index != None:
        es_load_conf["es.resource"] = args.es_index;

    if args.save_es_index != None:
        es_conf["es.resource"] = args.save_es_index + "/doc"

    res_df, model = analysis(load_es())
    df_with_cat = res_df.withColumn("cate", udf(
        lambda x: model.predict(x), IntegerType())("idf"))

    lean_res = df_with_cat.withColumn("messages", udf(lambda x: ' '.join(
        x), StringType())('message'))[['source', 'messages', 'cate']]

    lean_res = lean_res.rdd.map(lambda x: (
        '2', {'source': x.source, 'message': x.messages, 'cate': x.cate}))

    lean_res.saveAsNewAPIHadoopFile(
        path='-',
        outputFormatClass="org.elasticsearch.hadoop.mr.EsOutputFormat",
        keyClass="org.apache.hadoop.io.NullWritable",
        valueClass="org.elasticsearch.hadoop.mr.LinkedMapWritable", conf=es_conf)

sc.stop()
