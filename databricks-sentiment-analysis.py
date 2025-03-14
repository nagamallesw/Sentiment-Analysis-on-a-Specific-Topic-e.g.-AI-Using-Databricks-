# Databricks notebook source
# MAGIC %pip install requests textblob

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, date_trunc, current_timestamp
from pyspark.sql.types import FloatType, StringType  # Added StringType import
from textblob import TextBlob
import requests

spark = SparkSession.builder.appName("TopicSentimentAnalysis").getOrCreate()

# COMMAND ----------

api_key = "188df6befdb5416e9ff5949809b7f412"

# COMMAND ----------

topic = "AI"  
url = f"https://newsapi.org/v2/everything?q={topic}&apiKey={api_key}&language=en"


# COMMAND ----------


response = requests.get(url)
news_data = response.json()["articles"]
display(news_data)

# COMMAND ----------

news_df = spark.createDataFrame(
    [(article["title"] + " " + (article["description"] or "")) for article in news_data],
    "string"
).toDF("text")

# COMMAND ----------

def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return "positive"
    elif polarity < 0:
        return "negative"
    else:
        return "neutral"

sentiment_udf = udf(get_sentiment, StringType())

# COMMAND ----------

processed_df = news_df.withColumn("sentiment", sentiment_udf("text")) \
                     .withColumn("timestamp", current_timestamp())

# COMMAND ----------

delta_path = "/delta/topic_sentiment"
processed_df.write.format("delta").mode("append").save(delta_path)

# COMMAND ----------

try:
    sentiment_table = spark.read.format("delta").load(delta_path)
    print("Delta table loaded. Row count:", sentiment_table.count())
    display(sentiment_table.limit(5))  
except Exception as e:
    print(f"Error loading Delta table: {e}")
    raise


# COMMAND ----------

sentiment_trends = sentiment_table.groupBy(
    date_trunc("hour", "timestamp").alias("hour"),
    "sentiment"
).count()

# COMMAND ----------

print("Displaying sentiment trends...")
display(sentiment_trends)
