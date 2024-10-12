from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, FloatType
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import udf
from pyspark.ml.classification import DecisionTreeClassifier
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.ml.tuning import TrainValidationSplit
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder
from pprint import pprint
from pyspark.sql.functions import split, min, max
from pyspark.sql.types import IntegerType, StringType
from pyspark.sql.functions import broadcast, when
import sys,logging
from datetime import datetime
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.classification import RandomForestClassifier


AppName="FindInvestigatable"

formatter = logging.Formatter('[%(asctime)s] %(levelname)s @ line %(lineno)d: %(message)s')
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(handler)

spark = SparkSession.builder.appName(AppName).getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
logger.info("Starting spark application")

data = spark.read.option("inferSchema", True).option("header", True).csv("./Data/FINALDATANOCLAIMCONVERTEDNOMINALSNOZIP.csv")
#data.printSchema()

#print(data.head())

(train_data, test_data) = data.randomSplit([0.9, 0.1])
train_data.cache()
test_data.cache() 

input_cols =  data.columns[:-1]
input_cols.pop(0)
vector_assembler = VectorAssembler(inputCols=input_cols,
                                    outputCol="featureVector")

assembled_train_data = vector_assembler.transform(train_data)

#assembled_train_data.select("featureVector").show(truncate = False)

classifier = DecisionTreeClassifier(seed = 12, labelCol="ReportedFraud", featuresCol="featureVector", predictionCol="prediction") 
#classifier = RandomForestClassifier(seed = 12, labelCol="ReportedFraud", featuresCol="featureVector", predictionCol="prediction") #worse performance


model = classifier.fit(assembled_train_data)
#print(model.toDebugString)

#print(pd.DataFrame(model.featureImportances.toArray(), index=input_cols, columns=['importance']).sort_values(by="importance", ascending=False))

evaluator = MulticlassClassificationEvaluator(labelCol="ReportedFraud", predictionCol="prediction")

predictions = model.transform(vector_assembler.transform(test_data))


evaluator.setMetricName("accuracy").evaluate(predictions)
evaluator.setMetricName("f1").evaluate(predictions)

confusion_matrix = predictions.groupBy("ReportedFraud").\
  pivot("prediction", range(0,2)).count().\
  na.fill(0.0).\
  orderBy("ReportedFraud")

confusion_matrix.show()

factor = 1
fraudFee=udf(lambda v,v2:float(v[1]*v2*0.86* factor),FloatType())
predictions.select("CustomerID", "ReportedFraud", "prediction", "probability", "PolicyAnnualPremium", fraudFee("probability", "PolicyAnnualPremium")).orderBy(fraudFee("probability", "PolicyAnnualPremium"), ascending=False).show(10, truncate = False)




