#RAY PORT
RAY_PORT=9339
REDIS_PASS="d4t4bricks"

# install ray
# Install additional ray libraries
/databricks/python/bin/pip install ray[tune]==2.2.0
/databricks/python/bin/pip install ray[default]==2.2.0
/databricks/python/bin/pip install ray[rllib]==2.2.0
/databricks/python/bin/pip install prophet==1.1.1
/databricks/python/bin/pip install "holidays==0.11.3.1" --force-reinstall

mkdir /tmp/ray/job

# If starting on the Spark driver node, initialize the Ray head node
# If starting on the Spark worker node, connect to the head Ray node
if [ ! -z $DB_IS_DRIVER ] && [ $DB_IS_DRIVER = TRUE ] ; then
  echo "Starting the head node"
  ulimit -n 1000000
  ray start  --head --min-worker-port=20000 --max-worker-port=25000 --temp-dir="/tmp/ray/job" --plasma-directory="/tmp/ray/job" --port=$RAY_PORT --dashboard-port=8501 --dashboard-host="0.0.0.0" --include-dashboard=true --num-cpus=0
else
  sleep 40
  ulimit -n 1000000
  echo "Starting the non-head node - connecting to $DB_DRIVER_IP:$RAY_PORT"
  ray start  --min-worker-port=20000 --max-worker-port=25000 --temp-dir="/tmp/ray/job" --plasma-directory="/tmp/ray/job" --address="$DB_DRIVER_IP:$RAY_PORT"  