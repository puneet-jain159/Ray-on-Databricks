# Databricks notebook source
# MAGIC %sh 
# MAGIC RAY_PORT=9339
# MAGIC ulimit -n 1000000 && ray stop --force &&  ray start  --head --min-worker-port=20000 --max-worker-port=25000 --temp-dir="/tmp/ray/job" --plasma-directory="/tmp/ray/job" --port=$RAY_PORT  --dashboard-port=8501 --dashboard-host="0.0.0.0" --include-dashboard=true --num-cpus=0

# COMMAND ----------

import time
time.sleep(25)

# COMMAND ----------

# MAGIC %scala
# MAGIC val hst = sc.getConf.get("spark.driver.host")
# MAGIC val prt = 9339
# MAGIC val REDIS_PASS = "d4t4bricks"
# MAGIC // val cmd_ray =  s"ray stop --force && ray start --min-worker-port=20000 --max-worker-port=25000 --temp-dir='/tmp/ray/job' --plasma-directory='/tmp/ray/job' --redis-password=${REDIS_PASS} port=${hst}:${prt} --include-dashboard=false"
# MAGIC val cmd_ray =  s"ulimit -n 1000000 && rm -rf /tmp/ray/job && ray stop --force && ray start --address='${hst}:${prt}' "
# MAGIC import scala.concurrent.duration._
# MAGIC import sys.process._
# MAGIC var res=sc.runOnEachExecutor({ () =>
# MAGIC   var cmd_Result=Seq("bash", "-c", cmd_ray).!!
# MAGIC     cmd_Result }, 500.seconds)

# COMMAND ----------

import time
time.sleep(15)

# COMMAND ----------

import ray
hst = spark.conf.get("spark.driver.host")
prt = 9339
ray.init(address=f"{hst}:{prt}", ignore_reinit_error=True)
