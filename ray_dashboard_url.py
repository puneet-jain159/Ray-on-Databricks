# Databricks notebook source
import os
base_url='https://' + spark.conf.get("spark.databricks.workspaceUrl")
workspace_id=spark.conf.get("spark.databricks.clusterUsageTags.orgId")
cluster_id=spark.conf.get("spark.databricks.clusterUsageTags.clusterId")
dashboard_port='8501'

pathname_prefix='/driver-proxy/o/' + workspace_id + '/' + cluster_id + '/' + dashboard_port+"/" 
  
apitoken = dbutils.notebook().entry_point.getDbutils().notebook().getContext().apiToken().get()
dashboard_url=base_url + pathname_prefix  # ?token=' + apitoken[0:10] + " " + apitoken[10:]

#os.environ['STREAMLIT_REQUESTS_PATHNAME_PREFIX'] = pathname_prefix
#print(os.environ.get('STREAMLIT_REQUESTS_PATHNAME_PREFIX'))
print("Once the dashbord is running, it can be accessed at this link:\n\n" + dashboard_url)
