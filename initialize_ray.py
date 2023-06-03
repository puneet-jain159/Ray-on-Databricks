# Databricks notebook source
# %md # Installing RAY on a static cluster
# %md ### Get the version or default to 2.2.0

# COMMAND ----------

version = dbutils.widgets.get("version")
ray = f"ray[tune,default]=={version}"

# COMMAND ----------

# MAGIC %pip install $ray #other dependencies

# COMMAND ----------

# MAGIC %sh 
# MAGIC RAY_PORT=9339
# MAGIC ulimit -n 1000000 && ray stop --force &&  ray start  --head --min-worker-port=20000 --max-worker-port=25000 --temp-dir="/local_disk0/tmp/ray/job"  --port=$RAY_PORT  --dashboard-port=8501 --dashboard-host="0.0.0.0" --include-dashboard=true --num-cpus=0 --num-gpus=0 --system-config='{"object_spilling_config":"{\"type\":\"filesystem\",\"params\":{\"directory_path\":\"/local_disk0/tmp/spill\"}}"}'

# COMMAND ----------

import time
time.sleep(25)

# COMMAND ----------

# MAGIC %scala
# MAGIC val version = dbutils.widgets.get("version")
# MAGIC val hst = sc.getConf.get("spark.driver.host")
# MAGIC val prt = 9339
# MAGIC 
# MAGIC // val cmd_ray =  s"ray stop --force && ray start --min-worker-port=20000 --max-worker-port=25000 --temp-dir='/tmp/ray/job' --plasma-directory='/tmp/ray/job' --redis-password=${REDIS_PASS} port=${hst}:${prt} --include-dashboard=false"
# MAGIC // val cmd_ray =  s"/databricks/python/bin/pip install  ray[tune,default]==2.2.0 &&"
# MAGIC val cmd_ray =  s"/databricks/python/bin/pip install ray[tune,default]==${version} && ulimit -n 1000000 && rm -rf /tmp/ray/job && ray stop --force && ray start --address='${hst}:${prt}' "
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
from ray.runtime_env import RuntimeEnv
runtime_env = {
    "env_vars": {"GLOO_SOCKET_IFNAME":"eth0"}}
ray.init(address= f"{hst}:{prt}" ,
         runtime_env=RuntimeEnv(env_vars = runtime_env['env_vars']))

# COMMAND ----------

print("Ray initialized successfully !!")
