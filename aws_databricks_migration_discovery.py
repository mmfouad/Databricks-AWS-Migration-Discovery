# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # AWS Databricks Migration Discovery Notebook
# MAGIC
# MAGIC Purpose: inventory an existing **AWS Databricks** workspace and collect the Databricks usage/pricing evidence needed to size a move to **Azure Databricks**.
# MAGIC
# MAGIC This notebook separates the work into three outputs:
# MAGIC
# MAGIC - **API inventory**: clusters, SQL warehouses, instance pools, node types, jobs, task clusters, and API collection errors.
# MAGIC - **Billing and pricing**: `system.billing.usage`, `system.billing.list_prices`, current AWS/Azure price catalog snapshots, and Databricks list-price estimates.
# MAGIC - **Azure migration sizing**: compute and SQL warehouse sizing tables designed for an Azure migration workbook.
# MAGIC
# MAGIC Run options:
# MAGIC
# MAGIC - Inside the source AWS Databricks workspace.
# MAGIC - From VS Code using a Databricks-backed notebook/kernel.
# MAGIC - From VS Code or a local shell for REST API inventory only, with `DATABRICKS_HOST` and `DATABRICKS_TOKEN` set.
# MAGIC
# MAGIC Pricing note: Databricks list-price tables cover Databricks usage units such as DBUs. They do not replace AWS EC2/S3/EBS/network or Azure VM/ADLS/network pricing models.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

import configparser
import json
import os
import random
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

try:
    spark  # type: ignore[name-defined]
except NameError:
    try:
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()
    except Exception:
        spark = None

HAS_SPARK = spark is not None

if HAS_SPARK:
    try:
        from pyspark.sql.types import StringType, StructField, StructType
    except Exception:
        StringType = StructField = StructType = None
else:
    StringType = StructField = StructType = None


def env_bool(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        print(f"Ignoring invalid integer for {name}: {value!r}")
        return default


CONFIG = {
    "source_cloud": os.getenv("SOURCE_CLOUD", "AWS").upper(),
    "target_cloud": os.getenv("TARGET_CLOUD", "AZURE").upper(),
    "usage_lookback_days": env_int("USAGE_LOOKBACK_DAYS", 90),
    "api_timeout_seconds": env_int("DATABRICKS_API_TIMEOUT_SECONDS", 30),
    "api_max_retries": env_int("DATABRICKS_API_MAX_RETRIES", 5),
    "api_backoff_seconds": float(os.getenv("DATABRICKS_API_BACKOFF_SECONDS", "1.0")),
    "run_api_inventory": env_bool("RUN_API_INVENTORY", True),
    "run_billing_usage": env_bool("RUN_BILLING_USAGE", True),
    "output_base_path": os.getenv("OUTPUT_BASE_PATH", "dbfs:/tmp/aws_databricks_migration_discovery"),
    "local_output_dir": os.getenv("LOCAL_OUTPUT_DIR", "./outputs/aws_databricks_migration_discovery"),
}

start_date = (date.today() - timedelta(days=CONFIG["usage_lookback_days"])).isoformat()

print("Runtime summary")
print(f"- Spark available: {HAS_SPARK}")
print(f"- Source cloud: {CONFIG['source_cloud']}")
print(f"- Target cloud: {CONFIG['target_cloud']}")
print(f"- Usage lookback start date: {start_date}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Workspace credentials and display helpers
# MAGIC
# MAGIC Credential precedence:
# MAGIC
# MAGIC 1. `DATABRICKS_HOST` / `DATABRICKS_TOKEN`
# MAGIC 2. `DATABRICKS_WORKSPACE_URL` / `DATABRICKS_TOKEN`
# MAGIC 3. `~/.databrickscfg` using `DATABRICKS_CONFIG_PROFILE`, defaulting to `DEFAULT`
# MAGIC 4. Databricks notebook context, when running inside Databricks

# COMMAND ----------

def normalize_workspace_url(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    value = value.strip().rstrip("/")
    if not value:
        return None
    if not value.startswith(("http://", "https://")):
        value = f"https://{value}"
    return value


def load_databricks_cfg() -> Dict[str, str]:
    cfg_path = Path(os.getenv("DATABRICKS_CONFIG_FILE", "~/.databrickscfg")).expanduser()
    profile = os.getenv("DATABRICKS_CONFIG_PROFILE", "DEFAULT")
    if not cfg_path.exists():
        return {}

    parser = configparser.ConfigParser()
    parser.read(cfg_path)
    if not parser.has_section(profile):
        return {}

    section = parser[profile]
    return {
        "host": section.get("host", ""),
        "token": section.get("token", ""),
    }


def get_dbutils_if_available() -> Optional[Any]:
    if "dbutils" in globals():
        return globals()["dbutils"]
    if not HAS_SPARK:
        return None
    try:
        from pyspark.dbutils import DBUtils

        return DBUtils(spark)
    except Exception:
        return None


workspace_url = normalize_workspace_url(os.getenv("DATABRICKS_HOST") or os.getenv("DATABRICKS_WORKSPACE_URL"))
token = os.getenv("DATABRICKS_TOKEN")

cfg = load_databricks_cfg()
workspace_url = workspace_url or normalize_workspace_url(cfg.get("host"))
token = token or cfg.get("token")

dbutils_obj = get_dbutils_if_available()
if dbutils_obj is not None:
    try:
        ctx = dbutils_obj.notebook.entry_point.getDbutils().notebook().getContext()
        workspace_url = workspace_url or normalize_workspace_url(ctx.browserHostName().get())
        token = token or ctx.apiToken().get()
    except Exception as exc:
        print(f"Databricks notebook context was not available: {str(exc)[:300]}")

API_ENABLED = CONFIG["run_api_inventory"] and bool(workspace_url and token)

if workspace_url:
    print(f"Workspace URL: {workspace_url}")
else:
    print("Workspace URL not configured.")

if not API_ENABLED:
    print("REST API inventory is disabled or missing credentials. Set DATABRICKS_HOST and DATABRICKS_TOKEN to enable it.")


def value_is_null(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def normalize_cell_value(value: Any, force_string: bool = True) -> Optional[str]:
    if value_is_null(value):
        return None
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, default=str, sort_keys=True)
    if force_string:
        return str(value)
    return value


def prepare_pdf_for_spark(pdf: pd.DataFrame, force_string: bool = True) -> pd.DataFrame:
    if pdf is None or pdf.empty:
        return pd.DataFrame()
    cleaned = pdf.copy()
    cleaned.columns = [str(c) for c in cleaned.columns]
    for col in cleaned.columns:
        cleaned[col] = cleaned[col].map(lambda value: normalize_cell_value(value, force_string=force_string))
    return cleaned


def spark_df_from_pdf(pdf: pd.DataFrame, force_string: bool = True):
    if not HAS_SPARK or pdf is None or pdf.empty:
        return None

    cleaned = prepare_pdf_for_spark(pdf, force_string=force_string)
    if cleaned.empty:
        return None

    if force_string and StructType is not None:
        schema = StructType([StructField(str(col), StringType(), True) for col in cleaned.columns])
        rows = [tuple(row) for row in cleaned.itertuples(index=False, name=None)]
        return spark.createDataFrame(rows, schema)

    return spark.createDataFrame(cleaned)


def empty_spark_df(message: str = "No rows"):
    if not HAS_SPARK or StructType is None:
        return None
    return spark.createDataFrame([(message,)], ["message"])


def display_spark_df(df, message: str = "No rows") -> None:
    if df is None:
        if HAS_SPARK:
            df = empty_spark_df(message)
        else:
            print(message)
            return

    try:
        display(df)  # type: ignore[name-defined]
    except Exception:
        if hasattr(df, "show"):
            df.show(100, truncate=False)
        else:
            print(df)


def display_pdf(pdf: pd.DataFrame, message: str = "No rows", force_string: bool = True) -> None:
    if pdf is None or pdf.empty:
        display_spark_df(None, message)
        return
    sdf = spark_df_from_pdf(pdf, force_string=force_string)
    if sdf is not None:
        display_spark_df(sdf, message)
    else:
        try:
            display(pdf)  # type: ignore[name-defined]
        except Exception:
            print(pdf.to_string(index=False))


def write_pdf_outputs(pdf_by_name: Dict[str, pd.DataFrame], output_path: str, local_dir: str) -> None:
    Path(local_dir).mkdir(parents=True, exist_ok=True)

    for name, pdf in pdf_by_name.items():
        if pdf is None or pdf.empty:
            continue

        local_file = Path(local_dir) / f"{name}.csv"
        pdf.to_csv(local_file, index=False)

        if HAS_SPARK:
            sdf = spark_df_from_pdf(pdf, force_string=True)
            if sdf is not None:
                sdf.write.mode("overwrite").format("delta").save(f"{output_path}/{name}")

    print(f"Local CSV outputs written under: {Path(local_dir).resolve()}")
    if HAS_SPARK:
        print(f"Delta outputs written under: {output_path}")


def sql_quote(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. REST API helpers with retries and pagination

# COMMAND ----------

class DatabricksApiError(Exception):
    def __init__(self, message: str, path: str, status_code: Optional[int] = None, response_text: str = ""):
        super().__init__(message)
        self.path = path
        self.status_code = status_code
        self.response_text = response_text[:2000]


class DatabricksApiPermissionError(DatabricksApiError):
    pass


api_errors: List[Dict[str, Any]] = []
session = requests.Session()


def retry_delay_seconds(response: Optional[requests.Response], attempt: int) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return min(float(retry_after), 60.0)
            except ValueError:
                pass

    base = CONFIG["api_backoff_seconds"] * (2 ** max(attempt - 1, 0))
    jitter = random.uniform(0, CONFIG["api_backoff_seconds"])
    return min(base + jitter, 60.0)


def api_request(method: str, path: str, params: Optional[Dict[str, Any]] = None, json_body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not API_ENABLED:
        raise DatabricksApiError("REST API inventory is disabled or credentials are missing.", path)

    url = f"{workspace_url}{path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    transient_status_codes = {429, 500, 502, 503, 504}
    max_retries = max(CONFIG["api_max_retries"], 1)

    for attempt in range(1, max_retries + 1):
        response = None
        try:
            response = session.request(
                method=method.upper(),
                url=url,
                headers=headers,
                params=params,
                json=json_body,
                timeout=CONFIG["api_timeout_seconds"],
            )

            if response.status_code in transient_status_codes and attempt < max_retries:
                time.sleep(retry_delay_seconds(response, attempt))
                continue

            if response.status_code in {401, 403}:
                raise DatabricksApiPermissionError(
                    f"{method.upper()} {path} failed with HTTP {response.status_code}. Check workspace permissions and token scope.",
                    path=path,
                    status_code=response.status_code,
                    response_text=response.text,
                )

            if response.status_code >= 400:
                raise DatabricksApiError(
                    f"{method.upper()} {path} failed with HTTP {response.status_code}.",
                    path=path,
                    status_code=response.status_code,
                    response_text=response.text,
                )

            if not response.text:
                return {}
            return response.json()

        except (requests.ConnectionError, requests.Timeout) as exc:
            if attempt >= max_retries:
                raise DatabricksApiError(
                    f"{method.upper()} {path} failed after {max_retries} attempts: {exc}",
                    path=path,
                    response_text=str(exc),
                )
            time.sleep(retry_delay_seconds(response, attempt))

    raise DatabricksApiError(f"{method.upper()} {path} failed unexpectedly.", path=path)


def api_get(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return api_request("GET", path, params=params)


def record_api_error(source: str, path: str, exc: Exception) -> None:
    status_code = getattr(exc, "status_code", None)
    message = str(exc)
    action = "Review workspace permissions, token scope, endpoint availability, and admin settings."
    if isinstance(exc, DatabricksApiPermissionError) or status_code in {401, 403}:
        action = "Grant the caller permission to view this resource or use a workspace/admin token with the required scope."

    api_errors.append(
        {
            "source": source,
            "path": path,
            "status_code": status_code,
            "message": message[:2000],
            "response_text": getattr(exc, "response_text", "")[:2000],
            "recommended_action": action,
        }
    )
    print(f"Skipped {source}: {message[:300]}")


def paginated_get(
    source: str,
    path: str,
    response_key: str,
    params: Optional[Dict[str, Any]] = None,
    limit_param: Optional[str] = None,
    limit: Optional[int] = None,
    token_param: str = "page_token",
    token_response_key: str = "next_page_token",
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    page_token = None
    seen_tokens = set()

    while True:
        page_params = dict(params or {})
        if limit_param and limit:
            page_params[limit_param] = limit
        if page_token:
            page_params[token_param] = page_token

        response = api_get(path, params=page_params)
        batch = response.get(response_key, []) or []
        items.extend(batch)

        next_token = response.get(token_response_key)
        if not next_token:
            break
        if next_token in seen_tokens:
            raise DatabricksApiError(f"{source} pagination returned a repeated page token.", path=path)
        seen_tokens.add(next_token)
        page_token = next_token

    return items


def safe_paginated_get(source: str, path: str, response_key: str, **kwargs) -> List[Dict[str, Any]]:
    try:
        return paginated_get(source, path, response_key, **kwargs)
    except Exception as exc:
        record_api_error(source, path, exc)
        return []


def safe_get(source: str, path: str, response_key: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> Any:
    try:
        response = api_get(path, params=params)
        if response_key:
            return response.get(response_key, []) or []
        return response
    except Exception as exc:
        record_api_error(source, path, exc)
        return [] if response_key else {}

# COMMAND ----------

# MAGIC %md
# MAGIC # API Inventory

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Workspace compute inventory

# COMMAND ----------

clusters = safe_paginated_get(
    "clusters",
    "/api/2.0/clusters/list",
    "clusters",
) if API_ENABLED else []

warehouses = safe_paginated_get(
    "sql_warehouses",
    "/api/2.0/sql/warehouses",
    "warehouses",
    limit_param="max_results",
    limit=100,
) if API_ENABLED else []

instance_pools = safe_paginated_get(
    "instance_pools",
    "/api/2.0/instance-pools/list",
    "instance_pools",
) if API_ENABLED else []

node_types = safe_get(
    "node_types",
    "/api/2.0/clusters/list-node-types",
    "node_types",
) if API_ENABLED else []

clusters_df = pd.json_normalize(clusters)
warehouses_df = pd.json_normalize(warehouses)
pools_df = pd.json_normalize(instance_pools)
node_types_df = pd.json_normalize(node_types)

display_pdf(clusters_df, "No clusters returned by the clusters/list API")
display_pdf(warehouses_df, "No SQL warehouses found")
display_pdf(pools_df, "No instance pools found")
display_pdf(node_types_df, "No node types returned")

# COMMAND ----------

cluster_cols = [
    "cluster_id",
    "cluster_name",
    "state",
    "cluster_source",
    "spark_version",
    "node_type_id",
    "driver_node_type_id",
    "num_workers",
    "autoscale.min_workers",
    "autoscale.max_workers",
    "autotermination_minutes",
    "enable_elastic_disk",
    "runtime_engine",
    "policy_id",
    "creator_user_name",
]

warehouse_cols = [
    "id",
    "name",
    "cluster_size",
    "min_num_clusters",
    "max_num_clusters",
    "auto_stop_mins",
    "enable_photon",
    "warehouse_type",
    "spot_instance_policy",
    "state",
]

pool_cols = [
    "instance_pool_id",
    "instance_pool_name",
    "node_type_id",
    "min_idle_instances",
    "max_capacity",
    "idle_instance_autotermination_minutes",
    "enable_elastic_disk",
    "aws_attributes.availability",
    "aws_attributes.zone_id",
]

node_type_cols = [
    "node_type_id",
    "memory_mb",
    "num_cores",
    "description",
    "instance_type_id",
    "category",
    "is_deprecated",
    "support_ebs_volumes",
    "node_info.available_core_quota",
    "node_info.total_core_quota",
]


def select_existing(pdf: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    if pdf is None or pdf.empty:
        return pd.DataFrame()
    return pdf[[col for col in cols if col in pdf.columns]]


clusters_summary_pdf = select_existing(clusters_df, cluster_cols)
warehouses_summary_pdf = select_existing(warehouses_df, warehouse_cols)
pools_summary_pdf = select_existing(pools_df, pool_cols)
node_types_summary_pdf = select_existing(node_types_df, node_type_cols)

display_pdf(clusters_summary_pdf, "No clusters to summarize")
display_pdf(warehouses_summary_pdf, "No warehouses to summarize")
display_pdf(pools_summary_pdf, "No instance pools to summarize")
display_pdf(node_types_summary_pdf, "No node types to summarize")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Jobs inventory with paginated job details

# COMMAND ----------

job_summaries = safe_paginated_get(
    "jobs",
    "/api/2.1/jobs/list",
    "jobs",
    limit_param="limit",
    limit=100,
) if API_ENABLED else []


def merge_paginated_job_settings(base_job: Dict[str, Any], next_job: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base_job)
    merged_settings = dict(base_job.get("settings", {}) or {})
    next_settings = next_job.get("settings", {}) or {}

    for key, value in next_settings.items():
        if key not in {"tasks", "job_clusters", "parameters", "environments"}:
            merged_settings.setdefault(key, value)

    for array_key in ["tasks", "job_clusters", "parameters", "environments"]:
        existing = merged_settings.get(array_key) or []
        merged_settings[array_key] = existing + (next_settings.get(array_key) or [])

    merged["settings"] = merged_settings
    return merged


def get_job_detail(job_summary: Dict[str, Any]) -> Dict[str, Any]:
    job_id = job_summary.get("job_id")
    if not API_ENABLED or not job_id:
        return job_summary

    path = "/api/2.1/jobs/get"
    try:
        detail = api_get(path, params={"job_id": job_id})
        page_token = detail.get("next_page_token")
        seen_tokens = set()

        while page_token:
            if page_token in seen_tokens:
                raise DatabricksApiError(f"Job {job_id} pagination returned a repeated page token.", path=path)
            seen_tokens.add(page_token)
            next_detail = api_get(path, params={"job_id": job_id, "page_token": page_token})
            detail = merge_paginated_job_settings(detail, next_detail)
            page_token = next_detail.get("next_page_token")

        return detail
    except Exception as exc:
        record_api_error(f"job_detail:{job_id}", path, exc)
        return job_summary


jobs = [get_job_detail(job) for job in job_summaries]
jobs_df = pd.json_normalize(jobs)

display_pdf(jobs_df, "No jobs found")

# COMMAND ----------

job_cluster_rows: List[Dict[str, Any]] = []
job_task_rows: List[Dict[str, Any]] = []

for job in jobs:
    job_id = job.get("job_id")
    settings = job.get("settings", {}) or {}
    job_name = settings.get("name")

    for jc in settings.get("job_clusters", []) or []:
        cluster_key = jc.get("job_cluster_key")
        new_cluster = jc.get("new_cluster", {}) or {}
        autoscale = new_cluster.get("autoscale") or {}
        job_cluster_rows.append(
            {
                "job_id": job_id,
                "job_name": job_name,
                "task_key": None,
                "cluster_key": cluster_key,
                "cluster_scope": "job_cluster",
                "node_type_id": new_cluster.get("node_type_id"),
                "driver_node_type_id": new_cluster.get("driver_node_type_id"),
                "num_workers": new_cluster.get("num_workers"),
                "autoscale_min_workers": autoscale.get("min_workers"),
                "autoscale_max_workers": autoscale.get("max_workers"),
                "spark_version": new_cluster.get("spark_version"),
                "runtime_engine": new_cluster.get("runtime_engine"),
                "policy_id": new_cluster.get("policy_id"),
                "instance_pool_id": new_cluster.get("instance_pool_id"),
                "driver_instance_pool_id": new_cluster.get("driver_instance_pool_id"),
            }
        )

    for task in settings.get("tasks", []) or []:
        new_cluster = task.get("new_cluster") or {}
        autoscale = new_cluster.get("autoscale") or {}
        sql_task = task.get("sql_task") or {}
        task_warehouse_id = sql_task.get("warehouse_id")

        job_task_rows.append(
            {
                "job_id": job_id,
                "job_name": job_name,
                "task_key": task.get("task_key"),
                "existing_cluster_id": task.get("existing_cluster_id"),
                "job_cluster_key": task.get("job_cluster_key"),
                "sql_warehouse_id": task_warehouse_id,
                "has_new_cluster": bool(new_cluster),
                "node_type_id": new_cluster.get("node_type_id"),
                "driver_node_type_id": new_cluster.get("driver_node_type_id"),
                "num_workers": new_cluster.get("num_workers"),
                "autoscale_min_workers": autoscale.get("min_workers"),
                "autoscale_max_workers": autoscale.get("max_workers"),
                "spark_version": new_cluster.get("spark_version"),
                "runtime_engine": new_cluster.get("runtime_engine"),
                "policy_id": new_cluster.get("policy_id"),
                "instance_pool_id": new_cluster.get("instance_pool_id"),
                "driver_instance_pool_id": new_cluster.get("driver_instance_pool_id"),
            }
        )

        if new_cluster:
            job_cluster_rows.append(
                {
                    "job_id": job_id,
                    "job_name": job_name,
                    "task_key": task.get("task_key"),
                    "cluster_key": None,
                    "cluster_scope": "task_new_cluster",
                    "node_type_id": new_cluster.get("node_type_id"),
                    "driver_node_type_id": new_cluster.get("driver_node_type_id"),
                    "num_workers": new_cluster.get("num_workers"),
                    "autoscale_min_workers": autoscale.get("min_workers"),
                    "autoscale_max_workers": autoscale.get("max_workers"),
                    "spark_version": new_cluster.get("spark_version"),
                    "runtime_engine": new_cluster.get("runtime_engine"),
                    "policy_id": new_cluster.get("policy_id"),
                    "instance_pool_id": new_cluster.get("instance_pool_id"),
                    "driver_instance_pool_id": new_cluster.get("driver_instance_pool_id"),
                }
            )

job_clusters_pdf = pd.DataFrame(job_cluster_rows)
job_tasks_pdf = pd.DataFrame(job_task_rows)
api_errors_pdf = pd.DataFrame(api_errors)

display_pdf(job_clusters_pdf, "No job cluster definitions found")
display_pdf(job_tasks_pdf, "No job tasks found")
display_pdf(api_errors_pdf, "No REST API errors recorded")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Current notebook cluster executor details
# MAGIC
# MAGIC This only describes the cluster running this notebook. Use it as a sanity check, not as a full migration inventory.

# COMMAND ----------

executor_pdf = pd.DataFrame()
spark_conf_pdf = pd.DataFrame()

if HAS_SPARK:
    try:
        sc = spark.sparkContext
        executor_infos = sc._jsc.sc().statusTracker().getExecutorInfos()
        executor_rows = []
        for executor in executor_infos:
            executor_rows.append(
                {
                    "executor_id": executor.executorId(),
                    "host": executor.host(),
                    "total_cores": executor.totalCores(),
                    "max_memory_bytes": executor.maxMemory(),
                    "max_memory_gb": round(executor.maxMemory() / (1024 ** 3), 2),
                }
            )
        executor_pdf = pd.DataFrame(executor_rows)
        spark_conf_pdf = pd.DataFrame(spark.sparkContext.getConf().getAll(), columns=["key", "value"])
    except Exception as exc:
        print(f"Could not collect executor details: {str(exc)[:500]}")
else:
    print("Spark is not available, so executor details were skipped.")

display_pdf(executor_pdf, "No executor info returned")
display_pdf(spark_conf_pdf, "No Spark configuration returned")

# COMMAND ----------

# MAGIC %md
# MAGIC # Azure Migration Sizing Outputs From API Inventory

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Azure compute and SQL warehouse sizing tables
# MAGIC
# MAGIC These tables are intentionally mapping-friendly. Fill `azure_*_candidate` columns during assessment after choosing the Azure region, VM family, availability model, and performance target.

# COMMAND ----------

def int_or_none(value: Any) -> Optional[int]:
    if value_is_null(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def is_single_node_cluster(cluster: Dict[str, Any]) -> bool:
    spark_conf = cluster.get("spark_conf") or {}
    if spark_conf.get("spark.databricks.cluster.profile") == "singleNode":
        return True
    return int_or_none(cluster.get("num_workers")) == 0


def worker_bounds(config: Dict[str, Any]) -> Dict[str, Optional[int]]:
    autoscale = config.get("autoscale") or {}
    num_workers = int_or_none(config.get("num_workers"))
    min_workers = int_or_none(autoscale.get("min_workers"))
    max_workers = int_or_none(autoscale.get("max_workers"))

    if min_workers is None and num_workers is not None:
        min_workers = num_workers
    if max_workers is None and num_workers is not None:
        max_workers = num_workers

    return {
        "min_workers": min_workers,
        "max_workers": max_workers,
    }


def node_bounds(config: Dict[str, Any]) -> Dict[str, Optional[int]]:
    bounds = worker_bounds(config)
    if is_single_node_cluster(config):
        return {
            "min_nodes_including_driver": 1,
            "max_nodes_including_driver": 1,
        }

    min_workers = bounds["min_workers"]
    max_workers = bounds["max_workers"]
    return {
        "min_nodes_including_driver": min_workers + 1 if min_workers is not None else None,
        "max_nodes_including_driver": max_workers + 1 if max_workers is not None else None,
    }


def cluster_sizing_row(config: Dict[str, Any], source: str, item_id: Any, item_name: Any, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    bounds = worker_bounds(config)
    nodes = node_bounds(config)
    aws_attributes = config.get("aws_attributes") or {}
    row = {
        "inventory_source": source,
        "source_cloud": CONFIG["source_cloud"],
        "target_cloud": CONFIG["target_cloud"],
        "item_id": item_id,
        "item_name": item_name,
        "aws_worker_node_type_id": config.get("node_type_id"),
        "aws_driver_node_type_id": config.get("driver_node_type_id") or config.get("node_type_id"),
        "min_workers": bounds["min_workers"],
        "max_workers": bounds["max_workers"],
        "min_nodes_including_driver": nodes["min_nodes_including_driver"],
        "max_nodes_including_driver": nodes["max_nodes_including_driver"],
        "spark_version": config.get("spark_version"),
        "runtime_engine": config.get("runtime_engine"),
        "policy_id": config.get("policy_id"),
        "instance_pool_id": config.get("instance_pool_id"),
        "driver_instance_pool_id": config.get("driver_instance_pool_id"),
        "autotermination_minutes": config.get("autotermination_minutes"),
        "enable_elastic_disk": config.get("enable_elastic_disk"),
        "aws_availability": aws_attributes.get("availability"),
        "aws_zone_id": aws_attributes.get("zone_id"),
        "single_node": is_single_node_cluster(config),
        "azure_worker_vm_sku_candidate": None,
        "azure_driver_vm_sku_candidate": None,
        "azure_region_candidate": None,
        "azure_pricing_tier_candidate": None,
        "migration_notes": "Map AWS node type to Azure VM SKU, then validate DBU SKU and cloud infrastructure cost separately.",
    }
    if extra:
        row.update(extra)
    return row


cluster_sizing_rows = []
for cluster in clusters:
    cluster_sizing_rows.append(
        cluster_sizing_row(
            cluster,
            source="interactive_or_recent_cluster",
            item_id=cluster.get("cluster_id"),
            item_name=cluster.get("cluster_name"),
            extra={
                "cluster_state": cluster.get("state"),
                "cluster_source": cluster.get("cluster_source"),
                "creator_user_name": cluster.get("creator_user_name"),
            },
        )
    )

job_cluster_sizing_rows = []
for row in job_cluster_rows:
    config = {
        "node_type_id": row.get("node_type_id"),
        "driver_node_type_id": row.get("driver_node_type_id"),
        "num_workers": row.get("num_workers"),
        "autoscale": {
            "min_workers": row.get("autoscale_min_workers"),
            "max_workers": row.get("autoscale_max_workers"),
        },
        "spark_version": row.get("spark_version"),
        "runtime_engine": row.get("runtime_engine"),
        "policy_id": row.get("policy_id"),
        "instance_pool_id": row.get("instance_pool_id"),
        "driver_instance_pool_id": row.get("driver_instance_pool_id"),
    }
    job_cluster_sizing_rows.append(
        cluster_sizing_row(
            config,
            source=row.get("cluster_scope", "job_cluster"),
            item_id=row.get("job_id"),
            item_name=row.get("job_name"),
            extra={
                "task_key": row.get("task_key"),
                "cluster_key": row.get("cluster_key"),
            },
        )
    )

warehouse_sizing_rows = []
for wh in warehouses:
    warehouse_sizing_rows.append(
        {
            "inventory_source": "sql_warehouse",
            "source_cloud": CONFIG["source_cloud"],
            "target_cloud": CONFIG["target_cloud"],
            "warehouse_id": wh.get("id"),
            "warehouse_name": wh.get("name"),
            "warehouse_type": wh.get("warehouse_type"),
            "cluster_size": wh.get("cluster_size"),
            "min_num_clusters": wh.get("min_num_clusters"),
            "max_num_clusters": wh.get("max_num_clusters"),
            "auto_stop_mins": wh.get("auto_stop_mins"),
            "enable_photon": wh.get("enable_photon"),
            "spot_instance_policy": wh.get("spot_instance_policy"),
            "state": wh.get("state"),
            "azure_sql_warehouse_size_candidate": None,
            "azure_pricing_tier_candidate": None,
            "azure_region_candidate": None,
            "migration_notes": "Validate SQL warehouse SKU, Photon, serverless/pro classic choice, and concurrency separately in Azure.",
        }
    )

cluster_sizing_pdf = pd.DataFrame(cluster_sizing_rows)
job_cluster_sizing_pdf = pd.DataFrame(job_cluster_sizing_rows)
warehouse_sizing_pdf = pd.DataFrame(warehouse_sizing_rows)
combined_compute_sizing_pdf = pd.concat(
    [cluster_sizing_pdf, job_cluster_sizing_pdf],
    ignore_index=True,
) if cluster_sizing_rows or job_cluster_sizing_rows else pd.DataFrame()

display_pdf(combined_compute_sizing_pdf, "No compute sizing rows produced")
display_pdf(warehouse_sizing_pdf, "No SQL warehouse sizing rows produced")

# COMMAND ----------

# MAGIC %md
# MAGIC # Billing Usage And Databricks Pricing

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. SQL helpers for system tables

# COMMAND ----------

sql_errors: List[Dict[str, Any]] = []


def record_sql_error(name: str, exc: Exception, recommended_action: str) -> None:
    sql_errors.append(
        {
            "query_name": name,
            "message": str(exc)[:2000],
            "recommended_action": recommended_action,
        }
    )
    print(f"Skipped {name}: {str(exc)[:500]}")


def safe_sql(name: str, query: str, recommended_action: str):
    if not CONFIG["run_billing_usage"]:
        record_sql_error(name, Exception("Billing usage collection is disabled by RUN_BILLING_USAGE."), "Set RUN_BILLING_USAGE=true to enable it.")
        return None
    if not HAS_SPARK:
        record_sql_error(name, Exception("Spark is not available."), "Run from Databricks or VS Code with a Databricks-backed Spark session.")
        return None
    try:
        return spark.sql(query)
    except Exception as exc:
        record_sql_error(name, exc, recommended_action)
        return None


def describe_table_columns(table_name: str) -> List[str]:
    if not HAS_SPARK:
        return []
    try:
        rows = spark.sql(f"DESCRIBE TABLE {table_name}").collect()
        return [row["col_name"] for row in rows if row["col_name"] and not row["col_name"].startswith("#")]
    except Exception as exc:
        record_sql_error(
            f"describe_{table_name.replace('.', '_')}",
            exc,
            "Grant SELECT on the system table or ask an account admin to enable/access system tables.",
        )
        return []


usage_cols = set(describe_table_columns("system.billing.usage")) if CONFIG["run_billing_usage"] and HAS_SPARK else set()

usage_metadata_expr = "to_json(usage_metadata)" if "usage_metadata" in usage_cols else "CAST(NULL AS STRING)"
u_usage_metadata_expr = "to_json(u.usage_metadata)" if "usage_metadata" in usage_cols else "CAST(NULL AS STRING)"
product_features_expr = "to_json(product_features)" if "product_features" in usage_cols else "CAST(NULL AS STRING)"
identity_metadata_expr = "to_json(identity_metadata)" if "identity_metadata" in usage_cols else "CAST(NULL AS STRING)"
billing_origin_product_expr = "billing_origin_product" if "billing_origin_product" in usage_cols else "CAST(NULL AS STRING)"
u_billing_origin_product_expr = "u.billing_origin_product" if "billing_origin_product" in usage_cols else "CAST(NULL AS STRING)"
usage_type_expr = "usage_type" if "usage_type" in usage_cols else "CAST(NULL AS STRING)"
u_usage_type_expr = "u.usage_type" if "usage_type" in usage_cols else "CAST(NULL AS STRING)"

billing_permission_action = (
    "Grant SELECT on system.billing.usage and system.billing.list_prices, or ask a Databricks account admin "
    "to enable system tables and provide access."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Usage summaries from `system.billing.usage`

# COMMAND ----------

usage_detail = safe_sql(
    "billing_usage_detail",
    f"""
    SELECT
      date_trunc('month', usage_start_time) AS usage_month,
      workspace_id,
      sku_name,
      cloud,
      usage_unit,
      {usage_type_expr} AS usage_type,
      {billing_origin_product_expr} AS billing_origin_product,
      get_json_object({usage_metadata_expr}, '$.cluster_id') AS cluster_id,
      get_json_object({usage_metadata_expr}, '$.job_id') AS job_id,
      get_json_object({usage_metadata_expr}, '$.warehouse_id') AS warehouse_id,
      get_json_object({usage_metadata_expr}, '$.node_type') AS node_type,
      get_json_object({usage_metadata_expr}, '$.instance_pool_id') AS instance_pool_id,
      {product_features_expr} AS product_features_json,
      {identity_metadata_expr} AS identity_metadata_json,
      SUM(usage_quantity) AS total_usage_quantity,
      MIN(usage_start_time) AS first_seen,
      MAX(usage_end_time) AS last_seen
    FROM system.billing.usage
    WHERE usage_start_time >= DATE({sql_quote(start_date)})
    GROUP BY
      date_trunc('month', usage_start_time),
      workspace_id,
      sku_name,
      cloud,
      usage_unit,
      {usage_type_expr},
      {billing_origin_product_expr},
      get_json_object({usage_metadata_expr}, '$.cluster_id'),
      get_json_object({usage_metadata_expr}, '$.job_id'),
      get_json_object({usage_metadata_expr}, '$.warehouse_id'),
      get_json_object({usage_metadata_expr}, '$.node_type'),
      get_json_object({usage_metadata_expr}, '$.instance_pool_id'),
      {product_features_expr},
      {identity_metadata_expr}
    ORDER BY usage_month DESC, total_usage_quantity DESC
    """,
    billing_permission_action,
)
display_spark_df(usage_detail, "No billing usage detail available")

monthly_summary = safe_sql(
    "monthly_usage_summary",
    f"""
    SELECT
      date_trunc('month', usage_start_time) AS usage_month,
      sku_name,
      cloud,
      usage_unit,
      {usage_type_expr} AS usage_type,
      {billing_origin_product_expr} AS billing_origin_product,
      SUM(usage_quantity) AS total_usage_quantity
    FROM system.billing.usage
    WHERE usage_start_time >= DATE({sql_quote(start_date)})
    GROUP BY
      date_trunc('month', usage_start_time),
      sku_name,
      cloud,
      usage_unit,
      {usage_type_expr},
      {billing_origin_product_expr}
    ORDER BY usage_month DESC, total_usage_quantity DESC
    """,
    billing_permission_action,
)
display_spark_df(monthly_summary, "No monthly usage summary available")

cluster_usage = safe_sql(
    "cluster_usage_summary",
    f"""
    SELECT
      get_json_object({usage_metadata_expr}, '$.cluster_id') AS cluster_id,
      sku_name,
      cloud,
      usage_unit,
      {usage_type_expr} AS usage_type,
      {billing_origin_product_expr} AS billing_origin_product,
      SUM(usage_quantity) AS total_usage_quantity,
      MIN(usage_start_time) AS first_seen,
      MAX(usage_end_time) AS last_seen
    FROM system.billing.usage
    WHERE usage_start_time >= DATE({sql_quote(start_date)})
      AND get_json_object({usage_metadata_expr}, '$.cluster_id') IS NOT NULL
    GROUP BY
      get_json_object({usage_metadata_expr}, '$.cluster_id'),
      sku_name,
      cloud,
      usage_unit,
      {usage_type_expr},
      {billing_origin_product_expr}
    ORDER BY total_usage_quantity DESC
    """,
    billing_permission_action,
)
display_spark_df(cluster_usage, "No cluster usage summary available")

job_usage = safe_sql(
    "job_usage_summary",
    f"""
    SELECT
      get_json_object({usage_metadata_expr}, '$.job_id') AS job_id,
      sku_name,
      cloud,
      usage_unit,
      {usage_type_expr} AS usage_type,
      {billing_origin_product_expr} AS billing_origin_product,
      SUM(usage_quantity) AS total_usage_quantity,
      MIN(usage_start_time) AS first_seen,
      MAX(usage_end_time) AS last_seen
    FROM system.billing.usage
    WHERE usage_start_time >= DATE({sql_quote(start_date)})
      AND get_json_object({usage_metadata_expr}, '$.job_id') IS NOT NULL
    GROUP BY
      get_json_object({usage_metadata_expr}, '$.job_id'),
      sku_name,
      cloud,
      usage_unit,
      {usage_type_expr},
      {billing_origin_product_expr}
    ORDER BY total_usage_quantity DESC
    """,
    billing_permission_action,
)
display_spark_df(job_usage, "No job usage summary available")

warehouse_usage = safe_sql(
    "warehouse_usage_summary",
    f"""
    SELECT
      get_json_object({usage_metadata_expr}, '$.warehouse_id') AS warehouse_id,
      sku_name,
      cloud,
      usage_unit,
      {usage_type_expr} AS usage_type,
      {billing_origin_product_expr} AS billing_origin_product,
      SUM(usage_quantity) AS total_usage_quantity,
      MIN(usage_start_time) AS first_seen,
      MAX(usage_end_time) AS last_seen
    FROM system.billing.usage
    WHERE usage_start_time >= DATE({sql_quote(start_date)})
      AND get_json_object({usage_metadata_expr}, '$.warehouse_id') IS NOT NULL
    GROUP BY
      get_json_object({usage_metadata_expr}, '$.warehouse_id'),
      sku_name,
      cloud,
      usage_unit,
      {usage_type_expr},
      {billing_origin_product_expr}
    ORDER BY total_usage_quantity DESC
    """,
    billing_permission_action,
)
display_spark_df(warehouse_usage, "No SQL warehouse usage summary available")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Databricks pricing snapshots and AWS-to-Azure list-price estimate
# MAGIC
# MAGIC These tables use `system.billing.list_prices` to keep prices current instead of copying static public pricing into the notebook.

# COMMAND ----------

price_amount_expr = "try_cast(coalesce(get_json_object(to_json(pricing), '$.effective_list.default'), get_json_object(to_json(pricing), '$.default')) AS DOUBLE)"
source_cloud_sql = sql_quote(CONFIG["source_cloud"])
target_cloud_sql = sql_quote(CONFIG["target_cloud"])

price_catalog_history = safe_sql(
    "price_catalog_history_source_and_target",
    f"""
    SELECT
      sku_name,
      cloud,
      currency_code,
      usage_unit,
      price_start_time,
      price_end_time,
      {price_amount_expr} AS list_unit_price,
      to_json(pricing) AS pricing_json
    FROM system.billing.list_prices
    WHERE cloud IN ({source_cloud_sql}, {target_cloud_sql})
    ORDER BY cloud, sku_name, usage_unit, price_start_time DESC
    """,
    billing_permission_action,
)
display_spark_df(price_catalog_history, "No pricing history available")

current_price_catalog = safe_sql(
    "current_price_catalog_source_and_target",
    f"""
    SELECT
      sku_name,
      cloud,
      currency_code,
      usage_unit,
      price_start_time,
      price_end_time,
      {price_amount_expr} AS list_unit_price,
      to_json(pricing) AS pricing_json
    FROM system.billing.list_prices
    WHERE cloud IN ({source_cloud_sql}, {target_cloud_sql})
      AND price_end_time IS NULL
    ORDER BY cloud, sku_name, usage_unit
    """,
    billing_permission_action,
)
display_spark_df(current_price_catalog, "No current pricing catalog available")

aws_to_azure_dbu_estimate = safe_sql(
    "source_to_target_databricks_list_price_estimate",
    f"""
    WITH priced_usage AS (
      SELECT
        date_trunc('month', u.usage_start_time) AS usage_month,
        u.workspace_id,
        u.sku_name,
        u.cloud AS source_cloud,
        u.usage_unit,
        {u_usage_type_expr} AS usage_type,
        {u_billing_origin_product_expr} AS billing_origin_product,
        get_json_object({u_usage_metadata_expr}, '$.cluster_id') AS cluster_id,
        get_json_object({u_usage_metadata_expr}, '$.job_id') AS job_id,
        get_json_object({u_usage_metadata_expr}, '$.warehouse_id') AS warehouse_id,
        u.usage_quantity,
        try_cast(coalesce(get_json_object(to_json(source_price.pricing), '$.effective_list.default'), get_json_object(to_json(source_price.pricing), '$.default')) AS DOUBLE) AS source_list_unit_price,
        source_price.currency_code AS source_currency_code,
        try_cast(coalesce(get_json_object(to_json(target_price.pricing), '$.effective_list.default'), get_json_object(to_json(target_price.pricing), '$.default')) AS DOUBLE) AS target_list_unit_price,
        target_price.currency_code AS target_currency_code
      FROM system.billing.usage u
      LEFT JOIN system.billing.list_prices source_price
        ON source_price.cloud = u.cloud
       AND source_price.sku_name = u.sku_name
       AND source_price.usage_unit = u.usage_unit
       AND u.usage_start_time >= source_price.price_start_time
       AND (source_price.price_end_time IS NULL OR u.usage_start_time < source_price.price_end_time)
      LEFT JOIN system.billing.list_prices target_price
        ON target_price.cloud = {target_cloud_sql}
       AND target_price.sku_name = u.sku_name
       AND target_price.usage_unit = u.usage_unit
       AND u.usage_start_time >= target_price.price_start_time
       AND (target_price.price_end_time IS NULL OR u.usage_start_time < target_price.price_end_time)
      WHERE u.usage_start_time >= DATE({sql_quote(start_date)})
        AND u.cloud = {source_cloud_sql}
    )
    SELECT
      usage_month,
      workspace_id,
      sku_name,
      source_cloud,
      {target_cloud_sql} AS target_cloud,
      usage_unit,
      usage_type,
      billing_origin_product,
      cluster_id,
      job_id,
      warehouse_id,
      source_currency_code,
      target_currency_code,
      SUM(usage_quantity) AS total_usage_quantity,
      SUM(usage_quantity * source_list_unit_price) AS source_databricks_list_cost,
      SUM(usage_quantity * target_list_unit_price) AS target_databricks_list_cost,
      SUM(usage_quantity * target_list_unit_price) - SUM(usage_quantity * source_list_unit_price) AS estimated_databricks_list_cost_delta,
      SUM(CASE WHEN target_list_unit_price IS NULL THEN 1 ELSE 0 END) AS unpriced_usage_record_count
    FROM priced_usage
    GROUP BY
      usage_month,
      workspace_id,
      sku_name,
      source_cloud,
      usage_unit,
      usage_type,
      billing_origin_product,
      cluster_id,
      job_id,
      warehouse_id,
      source_currency_code,
      target_currency_code
    ORDER BY usage_month DESC, target_databricks_list_cost DESC
    """,
    billing_permission_action,
)
display_spark_df(aws_to_azure_dbu_estimate, "No AWS-to-Azure Databricks list-price estimate available")

sql_errors_pdf = pd.DataFrame(sql_errors)
display_pdf(sql_errors_pdf, "No billing SQL errors recorded")

# COMMAND ----------

# MAGIC %md
# MAGIC # Output Persistence

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Save API inventory, sizing outputs, and billing outputs separately

# COMMAND ----------

output_base_path = CONFIG["output_base_path"].rstrip("/")
api_output_path = f"{output_base_path}/api_inventory"
sizing_output_path = f"{output_base_path}/azure_migration_sizing"
billing_output_path = f"{output_base_path}/billing_usage_and_pricing"

local_output_dir = Path(CONFIG["local_output_dir"])
api_local_dir = str(local_output_dir / "api_inventory")
sizing_local_dir = str(local_output_dir / "azure_migration_sizing")
billing_local_dir = str(local_output_dir / "billing_usage_and_pricing")

api_outputs = {
    "clusters_raw": clusters_df,
    "clusters_summary": clusters_summary_pdf,
    "sql_warehouses_raw": warehouses_df,
    "sql_warehouses_summary": warehouses_summary_pdf,
    "instance_pools_raw": pools_df,
    "instance_pools_summary": pools_summary_pdf,
    "node_types_raw": node_types_df,
    "node_types_summary": node_types_summary_pdf,
    "jobs_raw": jobs_df,
    "job_clusters": job_clusters_pdf,
    "job_tasks": job_tasks_pdf,
    "api_errors": api_errors_pdf,
    "notebook_executors": executor_pdf,
    "notebook_spark_conf": spark_conf_pdf,
}

sizing_outputs = {
    "azure_compute_sizing_from_api": combined_compute_sizing_pdf,
    "azure_interactive_cluster_sizing_from_api": cluster_sizing_pdf,
    "azure_job_cluster_sizing_from_api": job_cluster_sizing_pdf,
    "azure_sql_warehouse_sizing_from_api": warehouse_sizing_pdf,
    "azure_node_type_reference_from_api": node_types_summary_pdf,
}

write_pdf_outputs(api_outputs, api_output_path, api_local_dir)
write_pdf_outputs(sizing_outputs, sizing_output_path, sizing_local_dir)

spark_outputs = {
    "billing_usage_detail": usage_detail,
    "monthly_usage_summary": monthly_summary,
    "cluster_usage_summary": cluster_usage,
    "job_usage_summary": job_usage,
    "warehouse_usage_summary": warehouse_usage,
    "price_catalog_history_source_and_target": price_catalog_history,
    "current_price_catalog_source_and_target": current_price_catalog,
    "source_to_target_databricks_list_price_estimate": aws_to_azure_dbu_estimate,
}

if HAS_SPARK:
    for name, df in spark_outputs.items():
        if df is not None:
            df.write.mode("overwrite").format("delta").save(f"{billing_output_path}/{name}")
            df.coalesce(1).write.mode("overwrite").option("header", "true").csv(f"{billing_output_path}_csv/{name}")

    if not sql_errors_pdf.empty:
        sdf = spark_df_from_pdf(sql_errors_pdf, force_string=True)
        if sdf is not None:
            sdf.write.mode("overwrite").format("delta").save(f"{billing_output_path}/billing_sql_errors")

    print(f"Billing Delta outputs written under: {billing_output_path}")
    print(f"Billing CSV outputs written under: {billing_output_path}_csv")
else:
    print("Spark is not available, so billing Delta/CSV outputs were skipped.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Azure migration interpretation checklist
# MAGIC
# MAGIC Use the outputs to build the Azure migration sizing model:
# MAGIC
# MAGIC - Map AWS node types from `azure_node_type_reference_from_api` to Azure VM families and region availability.
# MAGIC - Use `azure_compute_sizing_from_api` for all-purpose, interactive, and job-cluster node counts.
# MAGIC - Use `azure_sql_warehouse_sizing_from_api` for SQL warehouse size, min/max cluster count, Photon, and stop policy.
# MAGIC - Use `source_to_target_databricks_list_price_estimate` for Databricks DBU/list-price comparison by SKU.
# MAGIC - Add Azure VM, disk, ADLS Gen2, network, Private Link, monitoring, and backup costs separately.
# MAGIC - Reconcile API inventory with billing usage because cluster APIs only show active or recently terminated clusters, while billing usage is historical.
# MAGIC - Treat rows with `unpriced_usage_record_count > 0` as manual review items. SKU names can differ across clouds or service generations.
