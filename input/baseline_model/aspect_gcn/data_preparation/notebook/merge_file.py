from pyspark.sql import SparkSession

DATA_ROOT_PATH = "/storage/nhatnt/ite_model_nhatnt/projects/input/site_data/"

retail_rocket_root_path = DATA_ROOT_PATH + 'retail_rocket/'
recobell_root_path = DATA_ROOT_PATH + "recobell/"
ml_1m_root_path = DATA_ROOT_PATH + "ml-1m/"

spark = SparkSession.builder.appName("create train, test data").getOrCreate()

old_df = spark.read.csv(retail_rocket_root_path + "sequence_train_data",
                        "user_id INT, last_item_ids STRING, last_item_ids_neg STRING, target_item_id INT, target_label INT",
                        quote="|")

old_df.coalesce(1).write.mode("overwrite").csv(retail_rocket_root_path + "new_sequence_train_data", quote="|",
                                               quoteAll=True)


'''
x = []

'''