# Ray on Databricks 
This is a Repository to get Ray working on Databrick.

The Solutions have been tested to work on Databricks Runtime 11.0 to 11.3 LTS

The Ray Dashboard currently works on DBR 11.0 +

Ray Autoscaling is currently not supported in Databricks.

## Setting up Ray Cluster:

There are 2 methods to initialize ray on Databricks on a non-autoscaling cluster

1. Using the init_script.sh 
2. Using the ray_restart.sh notebook to start an adhoc ray cluster (Ray will not be initialized on nodes added later to the cluster)


## Viewing the Ray Dashboard:

the Ray Dashboard  uses reverse proxy method and the url is generated using the below command
```
%run ./dashboard_url
```
