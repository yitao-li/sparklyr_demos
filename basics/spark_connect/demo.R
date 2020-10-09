library(sparklyr)

sc <- spark_connect(master = "local", version = "3.0.0")
sdf <- sdf_len(sc, 10, repartition = 5)
print(sdf %>% sdf_num_partitions())
print(sdf)
