# Databricks notebook source
# MAGIC %md # Working with Ray
# MAGIC 
# MAGIC The Solutions have been tested to work on Databricks Runtime 11.0+</br>
# MAGIC The Ray Dashboard currently works on DBR 11.0 +</br>
# MAGIC Ray Autoscaling is currently not supported in Databricks.</br>
# MAGIC Current Ray supported version include  <b>2.2.0</b>  (earlier version include a bug which could not render the dashboard in Databricks)

# COMMAND ----------

# MAGIC %md ## Setting up Ray Cluster:
# MAGIC 
# MAGIC #### There are 2 methods to initialize ray on Databricks on a non-autoscaling cluster </br>
# MAGIC 
# MAGIC 1. Using the init_script.sh and adding it to cluster spin up 
# MAGIC 2. Using the ray_restart.sh notebook to start an adhoc ray cluster (Ray will not be initialized on nodes added later to the cluster)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Starting Ray Cluster with using init_script.sh
# MAGIC 
# MAGIC run the below command to initialize a Ray cluster for the current session only
# MAGIC ```
# MAGIC %run ./initialize_ray $version="2.2.0"
# MAGIC ```
# MAGIC 
# MAGIC Accepts a version arguement if not given defaults to 2.2.0 </br>

# COMMAND ----------

# MAGIC %run ./initialize_ray $version="2.2.0"

# COMMAND ----------

import ray
hst = spark.conf.get("spark.driver.host")
prt = 9339
ray.init(address=f"{hst}:{prt}", ignore_reinit_error=True)

# COMMAND ----------

# Get link to Dashboard 
from utils import get_dashboard_url
print(f"Link to ray Dashboard : {get_dashboard_url(spark,dbutils)}")

# COMMAND ----------

# MAGIC  %md ## Example workflow

# COMMAND ----------

import ray
import random
import time
import math
from fractions import Fraction


@ray.remote
def pi4_sample(sample_count):
    """pi4_sample runs sample_count experiments, and returns the 
    fraction of time it was inside the circle. 
    """
    in_count = 0
    for i in range(sample_count):
        x = random.random()
        y = random.random()
        if x*x + y*y <= 1:
            in_count += 1
    return Fraction(in_count, sample_count)

# COMMAND ----------

SAMPLE_COUNT = 1000 * 1000
start = time.time() 
future = pi4_sample.remote(sample_count = SAMPLE_COUNT)
pi4 = ray.get(future)
end = time.time()
dur = end - start
print(f'Running {SAMPLE_COUNT} tests took {dur} seconds')

# COMMAND ----------

FULL_SAMPLE_COUNT = 1000 * 1000 * 1000 # 100 billion samples! 
BATCHES = int(FULL_SAMPLE_COUNT / SAMPLE_COUNT)
print(f'Doing {BATCHES} batches')
results = []
for _ in range(BATCHES):
    results.append(pi4_sample.remote())
output = ray.get(results)

# COMMAND ----------


