# Ray on Databricks 
This is a Repository to get Ray working on Databrick.

The Solutions have been tested to work on Databricks Runtime 11.0+

The Ray Dashboard currently works on DBR 11.0 +

Ray Autoscaling is currently not supported in Databricks.

## Setting up Ray Cluster:

Ray uses the below ports to run : </br>

Port to connect to head node : **9339** </br>
Port to connect to dashboard : **8501** </br>


There are 2 methods to initialize ray on Databricks on a non-autoscaling cluster

1. Using the init_script.sh <br>
    Create an init_script.sh and attach to the cluster 
2. Using the **initialize_ray notebook** to start an adhoc ray cluster (Ray will not be initialized on nodes added later to the cluster) <br>
   run the below command in a notebook cell python ( Caution :interpreter will be restarted so run it at the start of the workflow) <br>
    ``` %run ./initialize_ray $version="2.2.0" ```


## Viewing the Ray Dashboard:

the Ray Dashboard  uses reverse proxy method and the url is generated using the below command
```
from utils import get_dashboard_url
print(f"Link to ray Dashboard : {get_dashboard_url(spark,dbutils)}")
```

## Examples:

- Deep Learning using pytorch </br>
     The notebook ```pytorch_distributed_example``` shows how to run a distributed training job using pytorch DDP and Ray 
