import os

def get_dashboard_url(spark,dbutils):  
  base_url='https://' + spark.conf.get("spark.databricks.workspaceUrl")
  workspace_id=spark.conf.get("spark.databricks.clusterUsageTags.orgId")
  cluster_id=spark.conf.get("spark.databricks.clusterUsageTags.clusterId")
  dashboard_port='8501'

  pathname_prefix='/driver-proxy/o/' + workspace_id + '/' + cluster_id + '/' + dashboard_port+"/" 

  apitoken = dbutils.notebook().entry_point.getDbutils().notebook().getContext().apiToken().get()
  dashboard_url=base_url + pathname_prefix  # ?token=' + apitoken[0:10] + " " + apitoken[10:]

  return dashboard_url