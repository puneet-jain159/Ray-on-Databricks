#RAY PORT
RAY_PORT=9339

# install ray
# Install additional ray libraries
/databricks/python/bin/pip install ray[tune,default]==2.3.1

#Create the location if not exists
mkdir -p /local_disk0/tmp/ray/job

# If starting on the Spark driver node, initialize the Ray head node
# If starting on the Spark worker node, connect to the head Ray node
if [ ! -z $DB_IS_DRIVER ] && [ $DB_IS_DRIVER = TRUE ] ; then
  echo "Starting the head node"
  ulimit -n 1000000
  ray start  --head --min-worker-port=20000 --max-worker-port=25000 --temp-dir="/local_disk0/tmp/ray/job"  --port=$RAY_PORT --dashboard-port=8501 --dashboard-host="0.0.0.0" --include-dashboard=true --num-cpus=3
else
  sleep 40
  ulimit -n 1000000
  echo "Starting the non-head node - connecting to $DB_DRIVER_IP:$RAY_PORT"
  ray start  --min-worker-port=20000 --max-worker-port=25000 --temp-dir="/local_disk0/tmp/ray/job" --address="$DB_DRIVER_IP:$RAY_PORT"  
fi