# Databricks notebook source
# MAGIC %md
# MAGIC # AWS Databricks &rarr; Azure Databricks Migration Sizing &amp; Pricing
# MAGIC
# MAGIC **What this notebook does.** It looks at the Databricks workloads running in *this* AWS workspace, recommends
# MAGIC the equivalent **Azure Databricks cluster size**, and prices it using the **public Azure Retail Prices API**.
# MAGIC
# MAGIC **How to run it (3 steps).**
# MAGIC
# MAGIC 1. Attach this notebook to any all-purpose cluster (DBR 12.2 LTS or later).
# MAGIC 2. Press **Run all**. A **widget bar** appears at the top of the notebook.
# MAGIC 3. Pick your **Azure region** in the widget bar. The default is `uaenorth` (UAE North). Everything re-runs
# MAGIC    from the widgets, so you never have to edit code.
# MAGIC
# MAGIC **What you get.**
# MAGIC
# MAGIC | Section | Output |
# MAGIC | --- | --- |
# MAGIC | Quick estimator | A single cluster sized and priced from the widgets. Works with zero permissions. |
# MAGIC | Workspace sizing | Every discovered AWS cluster, job cluster and SQL warehouse mapped to an Azure VM SKU. |
# MAGIC | Azure VM pricing | Hourly and monthly Azure VM cost per cluster, in pay-as-you-go, spot, savings plan and reserved terms. |
# MAGIC | Databricks DBU pricing | AWS DBU spend repriced onto Azure Databricks list prices. |
# MAGIC | Executive summary | One roll-up table for the business case. |
# MAGIC | Notes &amp; assumptions | Every assumption written out in plain English. |
# MAGIC
# MAGIC **Access and security.**
# MAGIC
# MAGIC - No Azure credentials, subscription or login are needed. The Azure Retail Prices API is public and anonymous.
# MAGIC - No secrets are stored in this notebook. The Databricks token comes from the notebook context at run time.
# MAGIC - Nothing is sent to Azure. The notebook only *reads* a public price list; your workload details never leave
# MAGIC   the workspace.
# MAGIC - Every step degrades gracefully. Missing permissions or blocked outbound internet reduce detail but never
# MAGIC   abort the run.
# MAGIC
# MAGIC > **Disclaimer.** All figures are **estimates based on public list prices**. They exclude your EA / MCA / CSP
# MAGIC > discounts, Azure Hybrid Benefit, managed disks, storage, networking and support. Validate against
# MAGIC > <https://azure.microsoft.com/pricing/details/databricks/> and your Microsoft agreement before committing.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup
# MAGIC
# MAGIC Only `pandas` and `requests` are used, and both ship with the Databricks runtime, so there is nothing to
# MAGIC install. No Azure SDK is required. If you are on a stripped-down image, run `%pip install pandas requests`
# MAGIC in a cell above this one and restart Python.

# COMMAND ----------

import configparser
import difflib
import json
import math
import os
import random
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import requests

NOTEBOOK_VERSION = "2.0.0"
REFERENCE_DATA_AS_OF = "2026-07"

# Standard billing month used for every monthly figure in this notebook.
DEFAULT_HOURS_PER_MONTH = 730.0
# Hours in a year, used to amortise reserved-instance prices back to an hourly rate.
HOURS_PER_YEAR = 8760.0
# Placeholder for a missing value. Deliberately not "n/a", which pandas reads back as NaN.
NOT_AVAILABLE = "not available"

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


def get_dbutils_if_available() -> Optional[Any]:
    """Return the ``dbutils`` handle when running in Databricks, otherwise ``None``."""
    if "dbutils" in globals():
        return globals()["dbutils"]
    if not HAS_SPARK:
        return None
    try:
        from pyspark.dbutils import DBUtils

        return DBUtils(spark)
    except Exception:
        return None


dbutils_obj = get_dbutils_if_available()
RUNNING_IN_DATABRICKS = dbutils_obj is not None


def env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read an environment variable, treating blank strings as unset."""
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def env_int(name: str, default: int) -> int:
    """Read an integer environment variable, falling back to ``default`` when invalid."""
    value = env_str(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        print(f"[warn] Ignoring invalid integer for {name}: {value!r}. Using {default}.")
        return default


def env_float(name: str, default: float) -> float:
    """Read a float environment variable, falling back to ``default`` when invalid."""
    value = env_str(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        print(f"[warn] Ignoring invalid number for {name}: {value!r}. Using {default}.")
        return default


print(f"AWS -> Azure Databricks Migration Sizing & Pricing  v{NOTEBOOK_VERSION}")
print(f"- Spark available:       {HAS_SPARK}")
print(f"- Running in Databricks: {RUNNING_IN_DATABRICKS}")
print(f"- pandas {pd.__version__} / requests {requests.__version__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Customer settings
# MAGIC
# MAGIC **You do not need to edit this cell.** Running it builds the widget bar at the top of the notebook, and the
# MAGIC widgets override everything below. Editing the constants is only needed when the notebook is run outside
# MAGIC Databricks (for example from VS Code or as a scheduled job).
# MAGIC
# MAGIC Precedence, highest first: **widget value** &rarr; **environment variable** &rarr; **constant in this cell**.

# COMMAND ----------

# =============================================================================================
#  CUSTOMER SETTINGS
#  Defaults are already sensible. Use the widget bar at the top of the notebook to change them.
# =============================================================================================

# ---------------------------------------------------------------------------------------------
#  A. TARGET AZURE REGION
# ---------------------------------------------------------------------------------------------
#  Valid Azure region values accepted by this notebook are retrieved from the Azure Retail Prices
#  API at run time (see the next section, which prints the live list). Use the EXACT values below,
#  which are the Azure `armRegionName` form: all lowercase, no spaces, no hyphens.
#
#    uaenorth        <- default
#    eastus
#    westeurope
#    northeurope
#
#  Do NOT use friendly names such as "UAE North", "East US" or "uae-north" as input values.
#  (They are auto-corrected when the intent is obvious, but the canonical value is the one below.)
#
#  ALLOWED VALUES - 69 regions, verified against https://prices.azure.com in 2026-07.
#  This list is the offline fallback; the live API list takes precedence when reachable.
#
#  Middle East & Africa      Europe                    Americas                  Asia Pacific
#  ------------------------  ------------------------  ------------------------  ------------------------
#  uaenorth                  austriaeast               brazilsouth               australiacentral
#  uaecentral                belgiumcentral            brazilsoutheast           australiacentral2
#  qatarcentral              denmarkeast               canadacentral             australiaeast
#  israelcentral             francecentral             canadaeast                australiasoutheast
#  israelnorthwest           francesouth               centralus                 centralindia
#  southafricanorth          germanynorth              chilecentral              eastasia
#  southafricawest           germanywestcentral        eastus                    indiasouthcentral
#                            italynorth                eastus2                   indonesiacentral
#                            northeurope               mexicocentral             japaneast
#                            norwayeast                northcentralus            japanwest
#                            norwaywest                southcentralus            jioindiacentral
#                            polandcentral             westcentralus             jioindiawest
#                            spaincentral              westus                    koreacentral
#                            swedencentral             westus2                   koreasouth
#                            swedensouth               westus3                   malaysiawest
#                            switzerlandnorth                                    newzealandnorth
#                            switzerlandwest                                     southeastasia
#                            uksouth                                             southindia
#                            ukwest                                              westindia
#                            westeurope
#
#  Sovereign clouds and edge zones - only pick these if you are deliberately targeting them:
#    usgovarizona   usgovtexas   usgovvirginia
#    attatlanta1    attdallas1   attdetroit1   attnewyork1   sgxsingapore1
#
#  Azure Databricks regional availability is a separate check:
#    https://learn.microsoft.com/azure/databricks/resources/supported-regions
# ---------------------------------------------------------------------------------------------
AZURE_REGION = "uaenorth"

#  Optional side-by-side comparison regions. Comma separated, same spelling rules as above.
#  Example: "uaecentral,westeurope,northeurope". Leave empty to skip the comparison.
AZURE_COMPARISON_REGIONS = ""

# ---------------------------------------------------------------------------------------------
#  B. PRICING OPTIONS
# ---------------------------------------------------------------------------------------------
#  AZURE_PRICING_MODEL - which Azure VM rate headlines the cost summary. All models are always
#  collected; this only chooses the headline column. Allowed values (exact spelling):
#    payg             pay-as-you-go Linux on demand              (default, most conservative)
#    spot             Azure Spot Linux
#    savings_plan_1y  1-year Azure savings plan for compute
#    savings_plan_3y  3-year Azure savings plan for compute
#    reserved_1y      1-year reserved instance, amortised hourly
#    reserved_3y      3-year reserved instance, amortised hourly
AZURE_PRICING_MODEL = "payg"

#  AZURE_CURRENCY - ISO currency for Azure retail prices. Allowed values (exact spelling):
#    USD  AUD  BRL  CAD  CHF  CNY  DKK  EUR  GBP  INR  JPY  KRW  NOK  NZD  RUB  SEK  TWD
AZURE_CURRENCY = "USD"

#  Hours per month used for every monthly figure. 730 = 365 * 24 / 12, the Azure standard.
HOURS_PER_MONTH = 730.0

#  Databricks workload tier for the Azure DBU price lookup. Allowed values: premium, standard
AZURE_DATABRICKS_TIER = "premium"

# ---------------------------------------------------------------------------------------------
#  C. QUICK ESTIMATOR - a single "what if" cluster, priced without needing any permissions
# ---------------------------------------------------------------------------------------------
#  WORKLOAD_SIZE - a t-shirt size that fills in vCPU and memory per worker node.
#    small   8 vCPU / 32 GB per node      light ETL, dev and test
#    medium  16 vCPU / 64 GB per node     standard production ETL           (default)
#    large   32 vCPU / 128 GB per node    heavy ETL, large joins
#    xlarge  64 vCPU / 256 GB per node    very large batch and ML training
#    custom  use QUICK_CUSTOM_VCPU_PER_NODE and QUICK_CUSTOM_MEMORY_GB_PER_NODE below
WORKLOAD_SIZE = "medium"
QUICK_NUM_WORKERS = 4                    # worker nodes, excluding the driver
QUICK_CUSTOM_VCPU_PER_NODE = 16          # used only when WORKLOAD_SIZE = "custom"
QUICK_CUSTOM_MEMORY_GB_PER_NODE = 64     # used only when WORKLOAD_SIZE = "custom"

#  AZURE_VM_FAMILY_PREFERENCE - constrain the recommended Azure VM family. Allowed values:
#    auto                pick the family from the workload's memory-per-core ratio  (default)
#    general_purpose     Dsv3 / Ddsv4 / Ddsv5 / Dadsv5     ~4 GB per vCPU
#    memory_optimized    Esv3 / Edsv4 / Edsv5 / Eadsv5     ~8 GB per vCPU
#    compute_optimized   Fsv2                              ~2 GB per vCPU
#    storage_optimized   Lsv2 / Lsv3                       large local NVMe
#    gpu                 NCasT4_v3 / NCads_A100_v4 / NDasr_v4
AZURE_VM_FAMILY_PREFERENCE = "auto"

#  Force a specific Azure VM SKU and skip the recommendation engine, for example
#  "Standard_E8ds_v5". Leave empty to let the notebook choose.
AZURE_VM_SKU_OVERRIDE = ""

# ---------------------------------------------------------------------------------------------
#  D. SIZING RULES applied to discovered AWS clusters
# ---------------------------------------------------------------------------------------------
#  AZURE_SIZING_STRATEGY - how an AWS instance type is translated into an Azure VM SKU.
#    like_for_like   the Azure VM must have at least the AWS vCPU and memory   (default, safest)
#    cost_optimized  allow up to 20% less memory per node when that lands on a materially cheaper
#                    Azure SKU. Every relaxed row is flagged so you can review it.
AZURE_SIZING_STRATEGY = "like_for_like"

#  Prefer Azure VM families with local NVMe temp storage (the "d" sizes such as D8ds_v5).
#  Databricks shuffle and disk caching benefit from local NVMe, so keep this True unless you have
#  standardised on remote-disk-only VM families.
AZURE_PREFER_LOCAL_SSD = True

#  Billable hours per month per cluster, used only when real node-hours cannot be read from
#  system.compute.node_timeline. 730 = always on, 160 = a typical business-hours cluster.
ASSUMED_MONTHLY_HOURS_INTERACTIVE = 160.0
ASSUMED_MONTHLY_HOURS_JOB = 60.0
ASSUMED_MONTHLY_HOURS_WAREHOUSE = 160.0

#  Azure VM SKU used to model one Databricks SQL warehouse node (classic and pro warehouses).
#  Serverless SQL warehouses have no customer-visible VM cost and are reported separately.
AZURE_SQL_WAREHOUSE_NODE_VM = "Standard_E8ds_v5"

# ---------------------------------------------------------------------------------------------
#  E. COLLECTION OPTIONS
# ---------------------------------------------------------------------------------------------
USAGE_LOOKBACK_DAYS = 90          # billing and node-hour window, in days
RUN_API_INVENTORY = True          # REST inventory of clusters, jobs, warehouses, pools
RUN_BILLING_USAGE = True          # system.billing.* and system.compute.* queries
RUN_AZURE_PRICING = True          # live call to https://prices.azure.com (needs outbound HTTPS)

#  Where results are written. Delta goes to OUTPUT_BASE_PATH, CSV to LOCAL_OUTPUT_DIR.
OUTPUT_BASE_PATH = "dbfs:/tmp/aws_databricks_migration_discovery"
LOCAL_OUTPUT_DIR = "./outputs/aws_databricks_migration_discovery"

# =============================================================================================
#  END OF CUSTOMER SETTINGS - nothing below this line needs to be edited
# =============================================================================================

# Offline fallback list of Azure regions, exactly as the Azure Retail Prices API reports them.
# Key = armRegionName (what the API and this notebook accept), value = display label only.
AZURE_REGIONS_FALLBACK: Dict[str, str] = {
    "attatlanta1": "ATT Atlanta 1",
    "attdallas1": "ATT Dallas 1",
    "attdetroit1": "ATT Detroit 1",
    "attnewyork1": "ATT New York 1",
    "australiacentral": "AU Central",
    "australiacentral2": "AU Central 2",
    "australiaeast": "AU East",
    "australiasoutheast": "AU Southeast",
    "austriaeast": "AT East",
    "belgiumcentral": "BE Central",
    "brazilsouth": "BR South",
    "brazilsoutheast": "BR Southeast",
    "canadacentral": "CA Central",
    "canadaeast": "CA East",
    "centralindia": "IN Central",
    "centralus": "US Central",
    "chilecentral": "CL Central",
    "denmarkeast": "DK East",
    "eastasia": "AP East",
    "eastus": "US East",
    "eastus2": "US East 2",
    "francecentral": "FR Central",
    "francesouth": "FR South",
    "germanynorth": "DE North",
    "germanywestcentral": "DE West Central",
    "indiasouthcentral": "IN South Central",
    "indonesiacentral": "ID Central",
    "israelcentral": "IL Central",
    "israelnorthwest": "IL Northwest",
    "italynorth": "IT North",
    "japaneast": "JA East",
    "japanwest": "JA West",
    "jioindiacentral": "IN Central Jio",
    "jioindiawest": "IN West Jio",
    "koreacentral": "KR Central",
    "koreasouth": "KR South",
    "malaysiawest": "MY West",
    "mexicocentral": "MX Central",
    "newzealandnorth": "NZ North",
    "northcentralus": "US North Central",
    "northeurope": "EU North",
    "norwayeast": "NO East",
    "norwaywest": "NO West",
    "polandcentral": "PL Central",
    "qatarcentral": "QA Central",
    "sgxsingapore1": "SGX Singapore 1",
    "southafricanorth": "ZA North",
    "southafricawest": "ZA West",
    "southcentralus": "US South Central",
    "southeastasia": "AP Southeast",
    "southindia": "IN South",
    "spaincentral": "ES Central",
    "swedencentral": "SE Central",
    "swedensouth": "SE South",
    "switzerlandnorth": "CH North",
    "switzerlandwest": "CH West",
    "uaecentral": "AE Central",
    "uaenorth": "AE North",
    "uksouth": "UK South",
    "ukwest": "UK West",
    "usgovarizona": "US Gov AZ",
    "usgovtexas": "US Gov TX",
    "usgovvirginia": "US Gov Virginia",
    "westcentralus": "US West Central",
    "westeurope": "EU West",
    "westindia": "IN West",
    "westus": "US West",
    "westus2": "US West 2",
    "westus3": "US West 3",
}

AZURE_PRICING_MODELS = ["payg", "spot", "savings_plan_1y", "savings_plan_3y", "reserved_1y", "reserved_3y"]
AZURE_CURRENCIES = [
    "USD", "AUD", "BRL", "CAD", "CHF", "CNY", "DKK", "EUR", "GBP",
    "INR", "JPY", "KRW", "NOK", "NZD", "RUB", "SEK", "TWD",
]
AZURE_SIZING_STRATEGIES = ["like_for_like", "cost_optimized"]
AZURE_VM_FAMILY_PREFERENCES = [
    "auto", "general_purpose", "memory_optimized", "compute_optimized", "storage_optimized", "gpu",
]
WORKLOAD_SIZES = ["small", "medium", "large", "xlarge", "custom"]

# t-shirt sizes for the quick estimator: vCPU and memory per worker node.
WORKLOAD_SIZE_PRESETS: Dict[str, Dict[str, float]] = {
    "small": {"vcpu_per_node": 8, "memory_gb_per_node": 32},
    "medium": {"vcpu_per_node": 16, "memory_gb_per_node": 64},
    "large": {"vcpu_per_node": 32, "memory_gb_per_node": 128},
    "xlarge": {"vcpu_per_node": 64, "memory_gb_per_node": 256},
}

# (widget name, label shown in the widget bar, default, choices - an empty list means free text)
WIDGET_SPECS: List[Tuple[str, str, str, List[str]]] = [
    ("azure_region", "A1 Azure region", AZURE_REGION, sorted(AZURE_REGIONS_FALLBACK)),
    ("azure_currency", "A2 Currency", AZURE_CURRENCY, AZURE_CURRENCIES),
    ("azure_pricing_model", "A3 VM pricing model", AZURE_PRICING_MODEL, AZURE_PRICING_MODELS),
    ("hours_per_month", "A4 Hours per month", str(int(HOURS_PER_MONTH)), []),
    ("workload_size", "B1 Workload size", WORKLOAD_SIZE, WORKLOAD_SIZES),
    ("quick_num_workers", "B2 Worker count", str(QUICK_NUM_WORKERS), []),
    ("quick_custom_vcpu_per_node", "B3 Custom vCPU per node", str(QUICK_CUSTOM_VCPU_PER_NODE), []),
    ("quick_custom_memory_gb_per_node", "B4 Custom GB per node", str(QUICK_CUSTOM_MEMORY_GB_PER_NODE), []),
    ("azure_vm_family_preference", "B5 VM family preference", AZURE_VM_FAMILY_PREFERENCE, AZURE_VM_FAMILY_PREFERENCES),
    ("azure_vm_sku_override", "B6 Force a VM SKU (optional)", AZURE_VM_SKU_OVERRIDE, []),
    ("azure_sizing_strategy", "C1 Sizing strategy", AZURE_SIZING_STRATEGY, AZURE_SIZING_STRATEGIES),
    ("azure_prefer_local_ssd", "C2 Prefer local NVMe VMs", str(AZURE_PREFER_LOCAL_SSD).lower(), ["true", "false"]),
    ("usage_lookback_days", "C3 Usage lookback days", str(USAGE_LOOKBACK_DAYS), []),
    ("azure_comparison_regions", "C4 Compare extra regions", AZURE_COMPARISON_REGIONS, []),
    ("run_api_inventory", "D1 Collect REST inventory", str(RUN_API_INVENTORY).lower(), ["true", "false"]),
    ("run_billing_usage", "D2 Query system tables", str(RUN_BILLING_USAGE).lower(), ["true", "false"]),
    ("run_azure_pricing", "D3 Fetch live Azure prices", str(RUN_AZURE_PRICING).lower(), ["true", "false"]),
]

# Environment variable that seeds each widget, so env vars still work for headless job runs.
WIDGET_ENV_VARS: Dict[str, str] = {
    "azure_region": "AZURE_REGION",
    "azure_currency": "AZURE_CURRENCY",
    "azure_pricing_model": "AZURE_PRICING_MODEL",
    "hours_per_month": "HOURS_PER_MONTH",
    "workload_size": "WORKLOAD_SIZE",
    "quick_num_workers": "QUICK_NUM_WORKERS",
    "quick_custom_vcpu_per_node": "QUICK_CUSTOM_VCPU_PER_NODE",
    "quick_custom_memory_gb_per_node": "QUICK_CUSTOM_MEMORY_GB_PER_NODE",
    "azure_vm_family_preference": "AZURE_VM_FAMILY_PREFERENCE",
    "azure_vm_sku_override": "AZURE_VM_SKU_OVERRIDE",
    "azure_sizing_strategy": "AZURE_SIZING_STRATEGY",
    "azure_prefer_local_ssd": "AZURE_PREFER_LOCAL_SSD",
    "usage_lookback_days": "USAGE_LOOKBACK_DAYS",
    "azure_comparison_regions": "AZURE_COMPARISON_REGIONS",
    "run_api_inventory": "RUN_API_INVENTORY",
    "run_billing_usage": "RUN_BILLING_USAGE",
    "run_azure_pricing": "RUN_AZURE_PRICING",
}

WIDGETS_AVAILABLE = False

if dbutils_obj is not None:
    try:
        for widget_name, label, default_value, choices in WIDGET_SPECS:
            seeded_default = env_str(WIDGET_ENV_VARS[widget_name], default_value) or default_value
            if choices:
                options = list(choices)
                if seeded_default not in options:
                    options = sorted(set(options) | {seeded_default})
                dbutils_obj.widgets.dropdown(widget_name, seeded_default, options, label)
            else:
                dbutils_obj.widgets.text(widget_name, seeded_default, label)
        WIDGETS_AVAILABLE = True
        print("Widget bar is ready at the top of the notebook.")
        print("Pick your Azure region in 'A1 Azure region'. Everything else already has a sensible default.")
    except Exception as exc:
        print(f"[info] Widgets unavailable, using constants and environment variables instead: {str(exc)[:200]}")
else:
    print("Not running in Databricks. The constants in this cell (or environment variables) are used.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Allowed Azure region values
# MAGIC
# MAGIC The list of valid regions is pulled **live from the Azure Retail Prices API** so it never goes stale. If the
# MAGIC workspace has no outbound internet access, the notebook falls back to the verified list embedded in the
# MAGIC settings cell above and says so.
# MAGIC
# MAGIC The values printed below are exactly what the region validator accepts and exactly what is sent to the
# MAGIC pricing API.

# COMMAND ----------

AZURE_PRICES_ENDPOINT = "https://prices.azure.com/api/retail/prices"
AZURE_PRICES_API_VERSION = "2023-01-01-preview"  # needed for the savingsPlan field

# Probe SKUs used to discover the region list. They are ordinary general-purpose sizes with very
# broad availability, so between them they cover every region that sells virtual machines.
_REGION_PROBE_SKUS = ["Standard_D4s_v3", "Standard_D4ds_v5", "Standard_E8ds_v4"]

azure_price_session = requests.Session()
azure_price_session.headers.update({"User-Agent": f"aws-databricks-migration-discovery/{NOTEBOOK_VERSION}"})

# Populated by the pricing section. Declared here so region discovery can report on connectivity.
AZURE_PRICING_DIAGNOSTICS: Dict[str, Any] = {
    "endpoint": AZURE_PRICES_ENDPOINT,
    "reachable": None,
    "last_error": None,
    "requests_made": 0,
}


def _http_get_json(url: str, params: Optional[Dict[str, Any]], timeout: int, max_retries: int) -> Dict[str, Any]:
    """GET a JSON document with exponential backoff on throttling and transient errors."""
    transient = {408, 429, 500, 502, 503, 504}
    last_error: Optional[Exception] = None

    for attempt in range(1, max(max_retries, 1) + 1):
        try:
            response = azure_price_session.get(url, params=params, timeout=timeout)
            AZURE_PRICING_DIAGNOSTICS["requests_made"] += 1

            if response.status_code in transient and attempt < max_retries:
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else min(2 ** attempt, 30)
                time.sleep(delay + random.uniform(0, 0.5))
                continue

            response.raise_for_status()
            return response.json()
        except Exception as exc:  # network error, bad status, or unparseable body
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(min(2 ** attempt, 30) + random.uniform(0, 0.5))

    raise RuntimeError(f"Azure Retail Prices request failed after {max_retries} attempt(s): {last_error}")


def fetch_azure_price_items(
    odata_filter: str,
    currency_code: str = "USD",
    timeout: int = 30,
    max_retries: int = 3,
    max_pages: int = 40,
) -> List[Dict[str, Any]]:
    """Return every price item matching an OData filter, following the API's paging links.

    The Azure Retail Prices API is public and anonymous: no subscription, login or key is needed.
    """
    params: Dict[str, Any] = {
        "api-version": AZURE_PRICES_API_VERSION,
        "currencyCode": currency_code,
        "$filter": odata_filter,
    }
    url: Optional[str] = AZURE_PRICES_ENDPOINT
    items: List[Dict[str, Any]] = []

    for _ in range(max_pages):
        if not url:
            break
        payload = _http_get_json(url, params, timeout, max_retries)
        items.extend(payload.get("Items", []) or [])
        url = payload.get("NextPageLink")
        params = None  # NextPageLink already carries the query string

    return items


def discover_azure_regions(timeout: int = 30, max_retries: int = 2) -> Tuple[Dict[str, str], str, Optional[str]]:
    """Discover valid ``armRegionName`` values from the Azure Retail Prices API.

    Returns ``(regions, source, error)`` where ``source`` is ``azure_retail_prices_api``,
    ``azure_retail_prices_api_partial`` or ``builtin_fallback_list``.
    """
    discovered: Dict[str, str] = {}
    error: Optional[str] = None

    for sku in _REGION_PROBE_SKUS:
        try:
            items = fetch_azure_price_items(
                f"serviceName eq 'Virtual Machines' and armSkuName eq '{sku}' and priceType eq 'Consumption'",
                timeout=timeout,
                max_retries=max_retries,
                max_pages=5,
            )
        except Exception as exc:
            error = str(exc)[:400]
            continue
        for item in items:
            arm_region = (item.get("armRegionName") or "").strip()
            if arm_region:
                discovered.setdefault(arm_region, item.get("location") or arm_region)

    if not discovered:
        AZURE_PRICING_DIAGNOSTICS["reachable"] = False
        AZURE_PRICING_DIAGNOSTICS["last_error"] = error
        return dict(AZURE_REGIONS_FALLBACK), "builtin_fallback_list", error

    AZURE_PRICING_DIAGNOSTICS["reachable"] = True
    # Union with the fallback so a partial probe never removes a region the customer needs.
    merged = dict(AZURE_REGIONS_FALLBACK)
    merged.update(discovered)
    source = "azure_retail_prices_api" if error is None else "azure_retail_prices_api_partial"
    return merged, source, error


_region_discovery_enabled = str(
    env_str("RUN_AZURE_PRICING", str(RUN_AZURE_PRICING))
).strip().lower() in {"1", "true", "yes", "y", "on"}

if _region_discovery_enabled:
    AZURE_REGIONS, AZURE_REGION_SOURCE, AZURE_REGION_DISCOVERY_ERROR = discover_azure_regions()
else:
    AZURE_REGIONS, AZURE_REGION_SOURCE, AZURE_REGION_DISCOVERY_ERROR = (
        dict(AZURE_REGIONS_FALLBACK),
        "builtin_fallback_list",
        "Live pricing lookups are switched off (RUN_AZURE_PRICING=false).",
    )

if AZURE_REGION_SOURCE.startswith("azure_retail_prices_api"):
    print(f"Region list source: live Azure Retail Prices API ({len(AZURE_REGIONS)} regions).")
    if AZURE_REGION_DISCOVERY_ERROR:
        print(f"[warn] One probe failed, so the built-in list was merged in: {AZURE_REGION_DISCOVERY_ERROR}")
else:
    print(f"Region list source: built-in fallback list ({len(AZURE_REGIONS)} regions, verified {REFERENCE_DATA_AS_OF}).")
    if AZURE_REGION_DISCOVERY_ERROR:
        print(f"[warn] Could not reach {AZURE_PRICES_ENDPOINT}: {AZURE_REGION_DISCOVERY_ERROR}")
        print("       This usually means the workspace has no outbound HTTPS access. Region validation still works.")

print()
print("Valid Azure region values accepted by this notebook (use the exact spelling shown):")
print()
_sorted_regions = sorted(AZURE_REGIONS)
for _row_start in range(0, len(_sorted_regions), 4):
    print("  " + "".join(f"{name:<24}" for name in _sorted_regions[_row_start:_row_start + 4]).rstrip())
print()
print('Do not use friendly names such as "UAE North" or "East US" as input values.')

azure_regions_pdf = pd.DataFrame(
    [
        {"arm_region_name": arm, "display_label": label, "is_default": arm == "uaenorth", "source": AZURE_REGION_SOURCE}
        for arm, label in sorted(AZURE_REGIONS.items())
    ]
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Validate the settings
# MAGIC
# MAGIC Input is corrected when the intent is obvious (`UAE North` &rarr; `uaenorth`) and rejected with a clear
# MAGIC message plus suggestions when it is not. The validator uses exactly the region list printed above.

# COMMAND ----------

class SettingsError(ValueError):
    """Raised when a customer setting cannot be interpreted. The message tells you how to fix it."""


def widget_value(name: str) -> Optional[str]:
    """Read a widget, returning ``None`` when widgets are unavailable or the value is blank."""
    if not WIDGETS_AVAILABLE or dbutils_obj is None:
        return None
    try:
        value = dbutils_obj.widgets.get(name)
    except Exception:
        return None
    value = (value or "").strip()
    return value or None


def resolve_setting(widget_name: str, constant_value: Any) -> str:
    """Resolve one setting: widget wins, then the environment variable, then the constant."""
    from_widget = widget_value(widget_name)
    if from_widget is not None:
        return from_widget
    from_env = env_str(WIDGET_ENV_VARS.get(widget_name, ""))
    if from_env is not None:
        return from_env
    return str(constant_value)


def as_bool(value: Any, default: bool) -> bool:
    """Interpret 'true'/'false'/'yes'/'1' style values as booleans."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _slug(value: str) -> str:
    """Lowercase and strip everything that is not a letter or digit."""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def normalize_region(value: Optional[str]) -> str:
    """Validate a region and return the canonical ``armRegionName``.

    Accepts 'uaenorth', 'UAE North', 'uae-north' and 'AE North'. Raises :class:`SettingsError`
    with the allowed values when the input cannot be matched.
    """
    raw = (value or "").strip()
    if not raw:
        raise SettingsError(
            "No Azure region was provided. Set the 'A1 Azure region' widget to a value such as 'uaenorth'."
        )

    candidate = _slug(raw)
    if candidate in AZURE_REGIONS:
        return candidate

    # Allow the friendly display label as input, but always return the armRegionName.
    label_lookup = {_slug(label): arm for arm, label in AZURE_REGIONS.items()}
    if candidate in label_lookup:
        return label_lookup[candidate]

    suggestions = difflib.get_close_matches(candidate, list(AZURE_REGIONS), n=5, cutoff=0.5)
    lines = [
        f"'{raw}' is not a valid Azure region name.",
        "",
        "Use the exact armRegionName spelling, for example: uaenorth, eastus, westeurope, northeurope.",
    ]
    if suggestions:
        lines += ["", f"Did you mean: {', '.join(suggestions)}?"]
    lines += [
        "",
        f"All {len(AZURE_REGIONS)} allowed values:",
        "  " + ", ".join(sorted(AZURE_REGIONS)),
    ]
    raise SettingsError("\n".join(lines))


def normalize_choice(value: Optional[str], allowed: Sequence[str], setting_name: str, default: str) -> str:
    """Validate a value against a fixed list of allowed values, with a helpful error message."""
    raw = (value or "").strip()
    if not raw:
        return default
    lookup = {option.lower(): option for option in allowed}
    if raw.lower() in lookup:
        return lookup[raw.lower()]
    suggestions = difflib.get_close_matches(raw.lower(), list(lookup), n=3, cutoff=0.5)
    hint = f" Did you mean: {', '.join(lookup[s] for s in suggestions)}?" if suggestions else ""
    raise SettingsError(f"'{raw}' is not a valid {setting_name}.{hint} Allowed values: {', '.join(allowed)}.")


def parse_region_list(value: Optional[str]) -> List[str]:
    """Parse a comma or space separated list of regions into canonical armRegionName values."""
    if not value:
        return []
    return [normalize_region(part) for part in re.split(r"[,;\s]+", value) if part]


def parse_number(
    value: Any,
    setting_name: str,
    default: float,
    minimum: float,
    maximum: float,
    as_int: bool = False,
) -> float:
    """Parse a numeric setting and range-check it, with a clear error when it is out of bounds."""
    raw = str(value).strip() if value is not None else ""
    if not raw:
        return default
    try:
        parsed = float(raw)
    except (TypeError, ValueError):
        raise SettingsError(f"{setting_name} must be a number, got {value!r}.")
    if math.isnan(parsed) or not minimum <= parsed <= maximum:
        raise SettingsError(f"{setting_name} must be between {minimum:g} and {maximum:g}, got {parsed:g}.")
    return int(parsed) if as_int else parsed


def normalize_vm_sku(value: Optional[str]) -> Optional[str]:
    """Normalise a VM SKU to the ``Standard_*`` armSkuName form, or return ``None`` when blank."""
    raw = (value or "").strip().replace(" ", "_")
    if not raw:
        return None
    if not raw.lower().startswith("standard_"):
        raw = f"Standard_{raw}"
    return "Standard_" + raw.split("_", 1)[1]


azure_region = normalize_region(resolve_setting("azure_region", AZURE_REGION))
azure_currency = normalize_choice(
    resolve_setting("azure_currency", AZURE_CURRENCY), AZURE_CURRENCIES, "currency", "USD"
)
azure_pricing_model = normalize_choice(
    resolve_setting("azure_pricing_model", AZURE_PRICING_MODEL), AZURE_PRICING_MODELS, "pricing model", "payg"
)
azure_sizing_strategy = normalize_choice(
    resolve_setting("azure_sizing_strategy", AZURE_SIZING_STRATEGY),
    AZURE_SIZING_STRATEGIES,
    "sizing strategy",
    "like_for_like",
)
azure_vm_family_preference = normalize_choice(
    resolve_setting("azure_vm_family_preference", AZURE_VM_FAMILY_PREFERENCE),
    AZURE_VM_FAMILY_PREFERENCES,
    "VM family preference",
    "auto",
)
workload_size = normalize_choice(
    resolve_setting("workload_size", WORKLOAD_SIZE), WORKLOAD_SIZES, "workload size", "medium"
)
azure_databricks_tier = normalize_choice(
    env_str("AZURE_DATABRICKS_TIER", AZURE_DATABRICKS_TIER), ["premium", "standard"], "Databricks tier", "premium"
)
comparison_regions = [
    region
    for region in parse_region_list(resolve_setting("azure_comparison_regions", AZURE_COMPARISON_REGIONS))
    if region != azure_region
]

CONFIG: Dict[str, Any] = {
    "notebook_version": NOTEBOOK_VERSION,
    "reference_data_as_of": REFERENCE_DATA_AS_OF,
    "run_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "source_cloud": (env_str("SOURCE_CLOUD", "AWS") or "AWS").upper(),
    "target_cloud": (env_str("TARGET_CLOUD", "AZURE") or "AZURE").upper(),
    "azure_region": azure_region,
    "azure_region_label": AZURE_REGIONS.get(azure_region, azure_region),
    "azure_region_source": AZURE_REGION_SOURCE,
    "azure_comparison_regions": comparison_regions,
    "azure_currency": azure_currency,
    "azure_pricing_model": azure_pricing_model,
    "azure_databricks_tier": azure_databricks_tier,
    "azure_sizing_strategy": azure_sizing_strategy,
    "azure_vm_family_preference": azure_vm_family_preference,
    "azure_vm_sku_override": normalize_vm_sku(resolve_setting("azure_vm_sku_override", AZURE_VM_SKU_OVERRIDE)),
    "azure_prefer_local_ssd": as_bool(resolve_setting("azure_prefer_local_ssd", AZURE_PREFER_LOCAL_SSD), AZURE_PREFER_LOCAL_SSD),
    "azure_sql_warehouse_node_vm": normalize_vm_sku(env_str("AZURE_SQL_WAREHOUSE_NODE_VM", AZURE_SQL_WAREHOUSE_NODE_VM)),
    "hours_per_month": parse_number(resolve_setting("hours_per_month", HOURS_PER_MONTH), "hours per month", DEFAULT_HOURS_PER_MONTH, 1, 744),
    "workload_size": workload_size,
    "quick_num_workers": int(parse_number(resolve_setting("quick_num_workers", QUICK_NUM_WORKERS), "worker count", QUICK_NUM_WORKERS, 0, 5000, as_int=True)),
    "quick_custom_vcpu_per_node": int(parse_number(resolve_setting("quick_custom_vcpu_per_node", QUICK_CUSTOM_VCPU_PER_NODE), "custom vCPU per node", QUICK_CUSTOM_VCPU_PER_NODE, 1, 512, as_int=True)),
    "quick_custom_memory_gb_per_node": parse_number(resolve_setting("quick_custom_memory_gb_per_node", QUICK_CUSTOM_MEMORY_GB_PER_NODE), "custom GB per node", QUICK_CUSTOM_MEMORY_GB_PER_NODE, 1, 6000),
    "assumed_monthly_hours_interactive": env_float("ASSUMED_MONTHLY_HOURS_INTERACTIVE", ASSUMED_MONTHLY_HOURS_INTERACTIVE),
    "assumed_monthly_hours_job": env_float("ASSUMED_MONTHLY_HOURS_JOB", ASSUMED_MONTHLY_HOURS_JOB),
    "assumed_monthly_hours_warehouse": env_float("ASSUMED_MONTHLY_HOURS_WAREHOUSE", ASSUMED_MONTHLY_HOURS_WAREHOUSE),
    "usage_lookback_days": int(parse_number(resolve_setting("usage_lookback_days", USAGE_LOOKBACK_DAYS), "usage lookback days", USAGE_LOOKBACK_DAYS, 1, 3650, as_int=True)),
    "run_api_inventory": as_bool(resolve_setting("run_api_inventory", RUN_API_INVENTORY), RUN_API_INVENTORY),
    "run_billing_usage": as_bool(resolve_setting("run_billing_usage", RUN_BILLING_USAGE), RUN_BILLING_USAGE),
    "run_azure_pricing": as_bool(resolve_setting("run_azure_pricing", RUN_AZURE_PRICING), RUN_AZURE_PRICING),
    "api_timeout_seconds": env_int("DATABRICKS_API_TIMEOUT_SECONDS", 30),
    "api_max_retries": env_int("DATABRICKS_API_MAX_RETRIES", 5),
    "api_backoff_seconds": env_float("DATABRICKS_API_BACKOFF_SECONDS", 1.0),
    "azure_price_timeout_seconds": env_int("AZURE_PRICE_TIMEOUT_SECONDS", 30),
    "azure_price_max_retries": env_int("AZURE_PRICE_MAX_RETRIES", 3),
    "output_base_path": (env_str("OUTPUT_BASE_PATH", OUTPUT_BASE_PATH) or OUTPUT_BASE_PATH).rstrip("/"),
    "local_output_dir": env_str("LOCAL_OUTPUT_DIR", LOCAL_OUTPUT_DIR) or LOCAL_OUTPUT_DIR,
}

start_date = (date.today() - timedelta(days=CONFIG["usage_lookback_days"])).isoformat()
CONFIG["usage_start_date"] = start_date

print("Settings accepted")
print(f"  Azure region          : {CONFIG['azure_region']}  ({CONFIG['azure_region_label']})")
if CONFIG["azure_comparison_regions"]:
    print(f"  Comparison regions    : {', '.join(CONFIG['azure_comparison_regions'])}")
print(f"  Currency / model      : {CONFIG['azure_currency']} / {CONFIG['azure_pricing_model']}")
print(f"  Hours per month       : {CONFIG['hours_per_month']:g}")
print(f"  Quick estimate        : {CONFIG['workload_size']} x {CONFIG['quick_num_workers']} workers")
print(f"  VM family preference  : {CONFIG['azure_vm_family_preference']}" + (f"  (forced SKU: {CONFIG['azure_vm_sku_override']})" if CONFIG["azure_vm_sku_override"] else ""))
print(f"  Sizing strategy       : {CONFIG['azure_sizing_strategy']} (prefer local NVMe: {CONFIG['azure_prefer_local_ssd']})")
print(f"  Databricks tier       : {CONFIG['azure_databricks_tier']}")
print(f"  Usage window          : last {CONFIG['usage_lookback_days']} days, from {start_date}")
print(f"  Steps enabled         : rest_inventory={CONFIG['run_api_inventory']}  system_tables={CONFIG['run_billing_usage']}  azure_pricing={CONFIG['run_azure_pricing']}")
print(f"  Delta output          : {CONFIG['output_base_path']}")
print(f"  CSV output            : {CONFIG['local_output_dir']}")

run_settings_pdf = pd.DataFrame(
    [{"setting": key, "value": json.dumps(value) if isinstance(value, (list, dict)) else value} for key, value in CONFIG.items()]
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Workspace credentials and shared helpers
# MAGIC
# MAGIC Inside Databricks there is **nothing to configure** &mdash; the notebook picks up the workspace URL and a
# MAGIC scoped token from the notebook context. No token is ever printed or written to the outputs.
# MAGIC
# MAGIC Credential precedence:
# MAGIC
# MAGIC 1. `DATABRICKS_HOST` / `DATABRICKS_TOKEN`
# MAGIC 2. `DATABRICKS_WORKSPACE_URL` / `DATABRICKS_TOKEN`
# MAGIC 3. `~/.databrickscfg`, using `DATABRICKS_CONFIG_PROFILE` (default `DEFAULT`)
# MAGIC 4. The Databricks notebook context

# COMMAND ----------

def normalize_workspace_url(value: Optional[str]) -> Optional[str]:
    """Normalise a workspace host into a scheme-qualified URL with no trailing slash."""
    if not value:
        return None
    value = value.strip().rstrip("/")
    if not value:
        return None
    if not value.startswith(("http://", "https://")):
        value = f"https://{value}"
    return value


def load_databricks_cfg() -> Dict[str, str]:
    """Read host and token from ``~/.databrickscfg`` for local runs. Returns {} when unavailable."""
    cfg_path = Path(env_str("DATABRICKS_CONFIG_FILE", "~/.databrickscfg") or "~/.databrickscfg").expanduser()
    profile = env_str("DATABRICKS_CONFIG_PROFILE", "DEFAULT") or "DEFAULT"
    if not cfg_path.exists():
        return {}

    parser = configparser.ConfigParser()
    try:
        parser.read(cfg_path)
    except Exception as exc:
        print(f"[warn] Could not read {cfg_path}: {str(exc)[:200]}")
        return {}
    if not parser.has_section(profile):
        return {}

    section = parser[profile]
    return {"host": section.get("host", ""), "token": section.get("token", "")}


workspace_url = normalize_workspace_url(env_str("DATABRICKS_HOST") or env_str("DATABRICKS_WORKSPACE_URL"))
token = env_str("DATABRICKS_TOKEN")

cfg = load_databricks_cfg()
workspace_url = workspace_url or normalize_workspace_url(cfg.get("host"))
token = token or cfg.get("token")

if dbutils_obj is not None:
    try:
        ctx = dbutils_obj.notebook.entry_point.getDbutils().notebook().getContext()
        workspace_url = workspace_url or normalize_workspace_url(ctx.browserHostName().get())
        token = token or ctx.apiToken().get()
    except Exception as exc:
        print(f"[info] Databricks notebook context was not available: {str(exc)[:300]}")

API_ENABLED = CONFIG["run_api_inventory"] and bool(workspace_url and token)

# Only the host is printed. The token is never printed, logged or written to any output table.
print(f"Workspace URL: {workspace_url}" if workspace_url else "Workspace URL: not configured")
print(f"Workspace token: {'resolved' if token else 'missing'}")
if not API_ENABLED:
    print(
        "[warn] REST inventory is off or credentials are missing. Run the notebook inside the workspace, "
        "or set DATABRICKS_HOST and DATABRICKS_TOKEN. The quick estimator still works without them."
    )


def value_is_null(value: Any) -> bool:
    """True for ``None`` and pandas/numpy NaN values."""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def first_non_null(*values: Any) -> Any:
    """Return the first argument that is not null."""
    for value in values:
        if not value_is_null(value):
            return value
    return None


def to_float(value: Any) -> Optional[float]:
    """Best-effort float conversion that never raises."""
    if value_is_null(value):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(parsed) else parsed


def to_int(value: Any) -> Optional[int]:
    """Best-effort int conversion that never raises."""
    parsed = to_float(value)
    return None if parsed is None else int(parsed)


def round_or_none(value: Any, digits: int = 4) -> Optional[float]:
    """Round a value, returning ``None`` for anything non-numeric."""
    parsed = to_float(value)
    return None if parsed is None else round(parsed, digits)


def money(value: Any, currency: Optional[str] = None, digits: int = 2) -> str:
    """Format a number for display, or ``NOT_AVAILABLE`` when it is missing."""
    parsed = to_float(value)
    if parsed is None:
        return NOT_AVAILABLE
    return f"{parsed:,.{digits}f} {currency or CONFIG['azure_currency']}"


def normalize_cell_value(value: Any, force_string: bool = True) -> Optional[str]:
    """Flatten dicts/lists to JSON so any value can be written to a string Spark column."""
    if value_is_null(value):
        return None
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, default=str, sort_keys=True)
    if force_string:
        return str(value)
    return value


def prepare_pdf_for_spark(pdf: pd.DataFrame, force_string: bool = True) -> pd.DataFrame:
    """Return a copy of a pandas frame whose values are all safe for ``createDataFrame``."""
    if pdf is None or pdf.empty:
        return pd.DataFrame()
    cleaned = pdf.copy()
    cleaned.columns = [str(c) for c in cleaned.columns]
    for col in cleaned.columns:
        cleaned[col] = cleaned[col].map(lambda value: normalize_cell_value(value, force_string=force_string))
    return cleaned


def spark_df_from_pdf(pdf: pd.DataFrame, force_string: bool = True):
    """Convert a pandas frame to a Spark frame, or ``None`` when Spark is unavailable/empty."""
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
    """A one-row Spark frame carrying an explanatory message, used in place of an empty table."""
    if not HAS_SPARK or StructType is None:
        return None
    return spark.createDataFrame([(message,)], ["message"])


def display_spark_df(df, message: str = "No rows") -> None:
    """Render a Spark frame with ``display()`` in Databricks, or ``show()`` elsewhere."""
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
    """Render a pandas frame in the nicest way available for the current environment."""
    if pdf is None or pdf.empty:
        display_spark_df(None, message)
        return
    sdf = spark_df_from_pdf(pdf, force_string=force_string)
    if sdf is not None:
        display_spark_df(sdf, message)
        return
    try:
        display(pdf)  # type: ignore[name-defined]
    except Exception:
        print(pdf.to_string(index=False))


def select_existing(pdf: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Project the columns that actually exist, so a missing API field never breaks the run."""
    if pdf is None or pdf.empty:
        return pd.DataFrame()
    return pdf[[col for col in cols if col in pdf.columns]]


def write_pdf_outputs(pdf_by_name: Dict[str, pd.DataFrame], output_path: str, local_dir: str) -> None:
    """Write each non-empty frame as CSV locally and as Delta under ``output_path``."""
    written_local: List[str] = []
    try:
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        local_writable = True
    except Exception as exc:
        print(f"[warn] Could not create local output directory {local_dir}: {str(exc)[:200]}")
        local_writable = False

    for name, pdf in pdf_by_name.items():
        if pdf is None or pdf.empty:
            continue

        if local_writable:
            try:
                pdf.to_csv(Path(local_dir) / f"{name}.csv", index=False)
                written_local.append(name)
            except Exception as exc:
                print(f"[warn] Could not write {name}.csv locally: {str(exc)[:200]}")

        if HAS_SPARK:
            sdf = spark_df_from_pdf(pdf, force_string=True)
            if sdf is not None:
                try:
                    sdf.write.mode("overwrite").format("delta").save(f"{output_path}/{name}")
                except Exception as exc:
                    print(f"[warn] Could not write Delta output {name}: {str(exc)[:200]}")

    if written_local:
        print(f"  CSV  : {Path(local_dir).resolve()}  ({len(written_local)} tables)")
    if HAS_SPARK:
        print(f"  Delta: {output_path}")


def sql_quote(value: str) -> str:
    """Quote a literal for safe inclusion in a Spark SQL statement."""
    return "'" + str(value).replace("'", "''") + "'"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Databricks REST API client
# MAGIC
# MAGIC Retries on HTTP 429 and 5xx with exponential backoff and jitter, follows pagination tokens, and records
# MAGIC permission problems in an `api_errors` table rather than failing the run.

# COMMAND ----------

class DatabricksApiError(Exception):
    """A Databricks REST call failed. Carries the path, status code and truncated response."""

    def __init__(self, message: str, path: str, status_code: Optional[int] = None, response_text: str = ""):
        super().__init__(message)
        self.path = path
        self.status_code = status_code
        self.response_text = (response_text or "")[:2000]


class DatabricksApiPermissionError(DatabricksApiError):
    """The caller is authenticated but not authorised for this resource (HTTP 401/403)."""


api_errors: List[Dict[str, Any]] = []
session = requests.Session()


def retry_delay_seconds(response: Optional[requests.Response], attempt: int, base_backoff: float) -> float:
    """Honour a ``Retry-After`` header when present, otherwise exponential backoff with jitter."""
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return min(float(retry_after), 60.0)
            except ValueError:
                pass

    base = base_backoff * (2 ** max(attempt - 1, 0))
    return min(base + random.uniform(0, base_backoff), 60.0)


def api_request(
    method: str,
    path: str,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Call a Databricks REST endpoint and return the decoded JSON body."""
    if not API_ENABLED:
        raise DatabricksApiError("REST API inventory is disabled or credentials are missing.", path)

    url = f"{workspace_url}{path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": f"aws-databricks-migration-discovery/{NOTEBOOK_VERSION}",
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
                time.sleep(retry_delay_seconds(response, attempt, CONFIG["api_backoff_seconds"]))
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

            return response.json() if response.text else {}

        except (requests.ConnectionError, requests.Timeout) as exc:
            if attempt >= max_retries:
                raise DatabricksApiError(
                    f"{method.upper()} {path} failed after {max_retries} attempts: {exc}",
                    path=path,
                    response_text=str(exc),
                )
            time.sleep(retry_delay_seconds(response, attempt, CONFIG["api_backoff_seconds"]))

    raise DatabricksApiError(f"{method.upper()} {path} failed unexpectedly.", path=path)


def api_get(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """GET a Databricks REST endpoint."""
    return api_request("GET", path, params=params)


def record_api_error(source: str, path: str, exc: Exception) -> None:
    """Record a collection failure with a recommended fix instead of raising."""
    status_code = getattr(exc, "status_code", None)
    action = "Review workspace permissions, token scope, endpoint availability, and admin settings."
    if isinstance(exc, DatabricksApiPermissionError) or status_code in {401, 403}:
        action = "Grant the caller permission to view this resource, or use a workspace admin token."

    api_errors.append(
        {
            "source": source,
            "path": path,
            "status_code": status_code,
            "message": str(exc)[:2000],
            "response_text": getattr(exc, "response_text", "")[:2000],
            "recommended_action": action,
        }
    )
    print(f"[skip] {source}: {str(exc)[:300]}")


def paginated_get(
    source: str,
    path: str,
    response_key: str,
    params: Optional[Dict[str, Any]] = None,
    limit_param: Optional[str] = None,
    limit: Optional[int] = None,
    token_param: str = "page_token",
    token_response_key: str = "next_page_token",
    max_pages: int = 1000,
) -> List[Dict[str, Any]]:
    """Collect every page of a list endpoint, guarding against repeated or endless page tokens."""
    items: List[Dict[str, Any]] = []
    page_token = None
    seen_tokens = set()

    for _ in range(max_pages):
        page_params = dict(params or {})
        if limit_param and limit:
            page_params[limit_param] = limit
        if page_token:
            page_params[token_param] = page_token

        response = api_get(path, params=page_params)
        items.extend(response.get(response_key, []) or [])

        next_token = response.get(token_response_key)
        if not next_token:
            return items
        if next_token in seen_tokens:
            raise DatabricksApiError(f"{source} pagination returned a repeated page token.", path=path)
        seen_tokens.add(next_token)
        page_token = next_token

    raise DatabricksApiError(f"{source} exceeded {max_pages} pages, stopping to avoid an endless loop.", path=path)


def safe_paginated_get(source: str, path: str, response_key: str, **kwargs) -> List[Dict[str, Any]]:
    """:func:`paginated_get` that records failures and returns ``[]`` instead of raising."""
    if not API_ENABLED:
        return []
    try:
        return paginated_get(source, path, response_key, **kwargs)
    except Exception as exc:
        record_api_error(source, path, exc)
        return []


def safe_get(source: str, path: str, response_key: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> Any:
    """:func:`api_get` that records failures and returns an empty value instead of raising."""
    if not API_ENABLED:
        return [] if response_key else {}
    try:
        response = api_get(path, params=params)
        return response.get(response_key, []) or [] if response_key else response
    except Exception as exc:
        record_api_error(source, path, exc)
        return [] if response_key else {}

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 1 &mdash; Discover the AWS workload

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Compute inventory: clusters, SQL warehouses, instance pools, node types
# MAGIC
# MAGIC `list-node-types` is the important one for sizing: it returns the **actual vCPU and memory** of every AWS
# MAGIC instance type this workspace can launch, so the Azure recommendation is built on measured specs rather than
# MAGIC a hardcoded lookup table.

# COMMAND ----------

clusters = safe_paginated_get("clusters", "/api/2.0/clusters/list", "clusters")
warehouses = safe_paginated_get("sql_warehouses", "/api/2.0/sql/warehouses", "warehouses", limit_param="max_results", limit=100)
instance_pools = safe_paginated_get("instance_pools", "/api/2.0/instance-pools/list", "instance_pools")
node_types = safe_get("node_types", "/api/2.0/clusters/list-node-types", "node_types")

clusters_df = pd.json_normalize(clusters)
warehouses_df = pd.json_normalize(warehouses)
pools_df = pd.json_normalize(instance_pools)
node_types_df = pd.json_normalize(node_types)

print(
    f"Collected: {len(clusters)} clusters, {len(warehouses)} SQL warehouses, "
    f"{len(instance_pools)} instance pools, {len(node_types)} node types."
)

display_pdf(clusters_df, "No clusters returned by the clusters/list API")
display_pdf(warehouses_df, "No SQL warehouses found")
display_pdf(pools_df, "No instance pools found")
display_pdf(node_types_df, "No node types returned")

# COMMAND ----------

CLUSTER_COLS = [
    "cluster_id", "cluster_name", "state", "cluster_source", "spark_version", "node_type_id",
    "driver_node_type_id", "num_workers", "autoscale.min_workers", "autoscale.max_workers",
    "autotermination_minutes", "enable_elastic_disk", "runtime_engine", "policy_id", "creator_user_name",
]
WAREHOUSE_COLS = [
    "id", "name", "cluster_size", "min_num_clusters", "max_num_clusters", "auto_stop_mins",
    "enable_photon", "warehouse_type", "spot_instance_policy", "state",
]
POOL_COLS = [
    "instance_pool_id", "instance_pool_name", "node_type_id", "min_idle_instances", "max_capacity",
    "idle_instance_autotermination_minutes", "enable_elastic_disk", "aws_attributes.availability",
    "aws_attributes.zone_id",
]
NODE_TYPE_COLS = [
    "node_type_id", "memory_mb", "num_cores", "description", "instance_type_id", "category",
    "is_deprecated", "support_ebs_volumes", "num_gpus",
    # The Databricks API nests the disk attributes under node_instance_type.
    "node_instance_type.instance_type_id", "node_instance_type.local_disk_size_gb",
    "node_instance_type.local_disks", "node_instance_type.local_nvme_disks",
    "node_info.available_core_quota", "node_info.total_core_quota",
]

clusters_summary_pdf = select_existing(clusters_df, CLUSTER_COLS)
warehouses_summary_pdf = select_existing(warehouses_df, WAREHOUSE_COLS)
pools_summary_pdf = select_existing(pools_df, POOL_COLS)
node_types_summary_pdf = select_existing(node_types_df, NODE_TYPE_COLS)

display_pdf(clusters_summary_pdf, "No clusters to summarize")
display_pdf(warehouses_summary_pdf, "No warehouses to summarize")
display_pdf(pools_summary_pdf, "No instance pools to summarize")
display_pdf(node_types_summary_pdf, "No node types to summarize")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Jobs inventory, including job clusters and task clusters

# COMMAND ----------

job_summaries = safe_paginated_get("jobs", "/api/2.1/jobs/list", "jobs", limit_param="limit", limit=100)

ARRAY_SETTING_KEYS = ["tasks", "job_clusters", "parameters", "environments"]


def merge_paginated_job_settings(base_job: Dict[str, Any], next_job: Dict[str, Any]) -> Dict[str, Any]:
    """Merge a follow-up ``jobs/get`` page into the base job, concatenating the array fields."""
    merged = dict(base_job)
    merged_settings = dict(base_job.get("settings", {}) or {})
    next_settings = next_job.get("settings", {}) or {}

    for key, value in next_settings.items():
        if key not in ARRAY_SETTING_KEYS:
            merged_settings.setdefault(key, value)

    for array_key in ARRAY_SETTING_KEYS:
        merged_settings[array_key] = (merged_settings.get(array_key) or []) + (next_settings.get(array_key) or [])

    merged["settings"] = merged_settings
    return merged


def get_job_detail(job_summary: Dict[str, Any], max_pages: int = 200) -> Dict[str, Any]:
    """Fetch full job settings, following pagination for jobs with many tasks."""
    job_id = job_summary.get("job_id")
    if not API_ENABLED or not job_id:
        return job_summary

    path = "/api/2.1/jobs/get"
    try:
        detail = api_get(path, params={"job_id": job_id})
        page_token = detail.get("next_page_token")
        seen_tokens = set()

        for _ in range(max_pages):
            if not page_token:
                break
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

print(f"Collected {len(jobs)} jobs.")
display_pdf(jobs_df, "No jobs found")

# COMMAND ----------

def cluster_config_row(job_id: Any, job_name: Any, task_key: Any, cluster_key: Any, scope: str, new_cluster: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a ``new_cluster`` block from a job definition into one tabular row."""
    autoscale = new_cluster.get("autoscale") or {}
    return {
        "job_id": job_id,
        "job_name": job_name,
        "task_key": task_key,
        "cluster_key": cluster_key,
        "cluster_scope": scope,
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


job_cluster_rows: List[Dict[str, Any]] = []
job_task_rows: List[Dict[str, Any]] = []

for job in jobs:
    job_id = job.get("job_id")
    settings = job.get("settings", {}) or {}
    job_name = settings.get("name")

    for job_cluster in settings.get("job_clusters", []) or []:
        job_cluster_rows.append(
            cluster_config_row(
                job_id, job_name, None, job_cluster.get("job_cluster_key"), "job_cluster",
                job_cluster.get("new_cluster", {}) or {},
            )
        )

    for task in settings.get("tasks", []) or []:
        new_cluster = task.get("new_cluster") or {}
        autoscale = new_cluster.get("autoscale") or {}
        sql_task = task.get("sql_task") or {}

        job_task_rows.append(
            {
                "job_id": job_id,
                "job_name": job_name,
                "task_key": task.get("task_key"),
                "existing_cluster_id": task.get("existing_cluster_id"),
                "job_cluster_key": task.get("job_cluster_key"),
                "sql_warehouse_id": sql_task.get("warehouse_id"),
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
                cluster_config_row(job_id, job_name, task.get("task_key"), None, "task_new_cluster", new_cluster)
            )

job_clusters_pdf = pd.DataFrame(job_cluster_rows)
job_tasks_pdf = pd.DataFrame(job_task_rows)
api_errors_pdf = pd.DataFrame(api_errors)

print(f"Found {len(job_cluster_rows)} job cluster definitions across {len(job_task_rows)} tasks.")
display_pdf(job_clusters_pdf, "No job cluster definitions found")
display_pdf(job_tasks_pdf, "No job tasks found")
display_pdf(api_errors_pdf, "No REST API errors recorded")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Current notebook cluster executor details
# MAGIC
# MAGIC A sanity check on the cluster running this notebook, not a migration input.

# COMMAND ----------

executor_pdf = pd.DataFrame()
spark_conf_pdf = pd.DataFrame()

if HAS_SPARK:
    try:
        status_tracker = spark.sparkContext._jsc.sc().statusTracker()
        executor_pdf = pd.DataFrame(
            [
                {
                    "executor_id": executor.executorId(),
                    "host": executor.host(),
                    "total_cores": executor.totalCores(),
                    "max_memory_bytes": executor.maxMemory(),
                    "max_memory_gb": round(executor.maxMemory() / (1024 ** 3), 2),
                }
                for executor in status_tracker.getExecutorInfos()
            ]
        )
        spark_conf_pdf = pd.DataFrame(spark.sparkContext.getConf().getAll(), columns=["key", "value"])
    except Exception as exc:
        print(f"[warn] Could not collect executor details: {str(exc)[:500]}")
else:
    print("Spark is not available, so executor details were skipped.")

display_pdf(executor_pdf, "No executor info returned")
display_pdf(spark_conf_pdf, "No Spark configuration returned")

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 2 &mdash; Size the Azure equivalent

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Hardware reference: AWS instance specs and the Azure VM catalog
# MAGIC
# MAGIC **How AWS specs are resolved**, best source first:
# MAGIC
# MAGIC 1. **This workspace's own `list-node-types` API** &mdash; the real vCPU, memory, GPU and local-disk figures
# MAGIC    Databricks reports for every instance type it can launch here. This is the authoritative source.
# MAGIC 2. **Derived from the instance name** &mdash; AWS size names are perfectly regular
# MAGIC    (`large` = 2 vCPU, `xlarge` = 4, `N` + `xlarge` = 4&times;N), and each family has a fixed memory-per-vCPU
# MAGIC    ratio. Used for node types seen in billing history that the API no longer lists.
# MAGIC 3. **Unknown** &mdash; flagged in the output, never silently guessed.
# MAGIC
# MAGIC Every sizing row carries `aws_spec_source` so you can see which of the three was used.

# COMMAND ----------

# AWS size token -> vCPU. AWS naming is regular: "Nxlarge" is always N * 4 vCPU.
AWS_SIZE_VCPU: Dict[str, int] = {
    "nano": 1, "micro": 1, "small": 1, "medium": 1, "large": 2, "xlarge": 4,
    **{f"{n}xlarge": n * 4 for n in (2, 3, 4, 6, 8, 9, 10, 12, 16, 18, 24, 32, 48, 56, 112)},
}

# AWS family prefix -> (memory GiB per vCPU, workload category, ships local NVMe).
# The "d" and "i" families have local NVMe, which is what Databricks uses for shuffle and disk cache.
AWS_FAMILY_SPECS: Dict[str, Tuple[float, str, bool]] = {
    "t2": (2.0, "general_purpose", False),
    "t3": (4.0, "general_purpose", False),
    "t3a": (4.0, "general_purpose", False),
    "m4": (4.0, "general_purpose", False),
    "m5": (4.0, "general_purpose", False),
    "m5a": (4.0, "general_purpose", False),
    "m5ad": (4.0, "general_purpose", True),
    "m5d": (4.0, "general_purpose", True),
    "m5dn": (4.0, "general_purpose", True),
    "m5n": (4.0, "general_purpose", False),
    "m6a": (4.0, "general_purpose", False),
    "m6g": (4.0, "general_purpose", False),
    "m6gd": (4.0, "general_purpose", True),
    "m6i": (4.0, "general_purpose", False),
    "m6id": (4.0, "general_purpose", True),
    "m7a": (4.0, "general_purpose", False),
    "m7g": (4.0, "general_purpose", False),
    "m7gd": (4.0, "general_purpose", True),
    "m7i": (4.0, "general_purpose", False),
    "m7id": (4.0, "general_purpose", True),
    "c4": (2.0, "compute_optimized", False),
    "c5": (2.0, "compute_optimized", False),
    "c5a": (2.0, "compute_optimized", False),
    "c5ad": (2.0, "compute_optimized", True),
    "c5d": (2.0, "compute_optimized", True),
    "c5n": (2.625, "compute_optimized", False),
    "c6a": (2.0, "compute_optimized", False),
    "c6g": (2.0, "compute_optimized", False),
    "c6gd": (2.0, "compute_optimized", True),
    "c6i": (2.0, "compute_optimized", False),
    "c6id": (2.0, "compute_optimized", True),
    "c7a": (2.0, "compute_optimized", False),
    "c7g": (2.0, "compute_optimized", False),
    "c7gd": (2.0, "compute_optimized", True),
    "c7i": (2.0, "compute_optimized", False),
    "r4": (7.625, "memory_optimized", False),
    "r5": (8.0, "memory_optimized", False),
    "r5a": (8.0, "memory_optimized", False),
    "r5ad": (8.0, "memory_optimized", True),
    "r5d": (8.0, "memory_optimized", True),
    "r5dn": (8.0, "memory_optimized", True),
    "r5n": (8.0, "memory_optimized", False),
    "r6a": (8.0, "memory_optimized", False),
    "r6g": (8.0, "memory_optimized", False),
    "r6gd": (8.0, "memory_optimized", True),
    "r6i": (8.0, "memory_optimized", False),
    "r6id": (8.0, "memory_optimized", True),
    "r7a": (8.0, "memory_optimized", False),
    "r7g": (8.0, "memory_optimized", False),
    "r7gd": (8.0, "memory_optimized", True),
    "r7i": (8.0, "memory_optimized", False),
    "r7iz": (8.0, "memory_optimized", False),
    "x1": (15.25, "memory_optimized", True),
    "x1e": (30.5, "memory_optimized", True),
    "x2gd": (16.0, "memory_optimized", True),
    "x2idn": (16.0, "memory_optimized", True),
    "x2iedn": (32.0, "memory_optimized", True),
    "z1d": (8.0, "memory_optimized", True),
    "i2": (30.5, "storage_optimized", True),
    "i3": (7.625, "storage_optimized", True),
    "i3en": (8.0, "storage_optimized", True),
    "i4g": (8.0, "storage_optimized", True),
    "i4i": (8.0, "storage_optimized", True),
    "im4gn": (4.0, "storage_optimized", True),
    "is4gen": (6.0, "storage_optimized", True),
    "d2": (7.625, "storage_optimized", True),
    "d3": (8.0, "storage_optimized", True),
    "d3en": (6.0, "storage_optimized", True),
    "h1": (8.0, "storage_optimized", True),
    "g4dn": (4.0, "gpu", True),
    "g5": (4.0, "gpu", True),
    "g5g": (2.0, "gpu", True),
    "g6": (4.0, "gpu", True),
    "p2": (12.2, "gpu", False),
    "p3": (7.625, "gpu", False),
    "p3dn": (10.66, "gpu", True),
    "p4d": (12.0, "gpu", True),
    "p5": (16.0, "gpu", True),
}

# GPUs per instance, for the AWS families where the count is not simply derivable.
AWS_GPU_COUNTS: Dict[str, int] = {
    "g4dn.12xlarge": 4, "g4dn.metal": 8, "g5.12xlarge": 4, "g5.24xlarge": 4, "g5.48xlarge": 8,
    "g6.12xlarge": 4, "g6.24xlarge": 4, "g6.48xlarge": 8, "p2.8xlarge": 8, "p2.16xlarge": 16,
    "p3.8xlarge": 4, "p3.16xlarge": 8, "p3dn.24xlarge": 8, "p4d.24xlarge": 8, "p5.48xlarge": 8,
}


def parse_aws_instance_type(instance_type: Optional[str]) -> Optional[Dict[str, Any]]:
    """Derive vCPU, memory and category from an AWS instance name such as ``r5d.4xlarge``.

    Returns ``None`` when the name cannot be parsed. AWS naming is regular enough that this is a
    reliable fallback for node types the workspace API no longer returns.
    """
    if not instance_type:
        return None
    name = str(instance_type).strip().lower()
    # Databricks node_type_ids look like "r5d.4xlarge"; strip any trailing Databricks suffix.
    match = re.match(r"^([a-z][a-z0-9\-]*?)\.([a-z0-9]+)", name)
    if not match:
        return None

    family, size = match.group(1), match.group(2)
    if family not in AWS_FAMILY_SPECS or size not in AWS_SIZE_VCPU:
        return None

    memory_per_vcpu, category, has_local_nvme = AWS_FAMILY_SPECS[family]
    vcpu = AWS_SIZE_VCPU[size]
    gpus = AWS_GPU_COUNTS.get(f"{family}.{size}", 1 if category == "gpu" else 0)

    return {
        "vcpu": vcpu,
        "memory_gb": round(vcpu * memory_per_vcpu, 1),
        "category": category,
        "has_local_nvme": has_local_nvme,
        "num_gpus": gpus,
        "aws_family": family,
        "aws_size": size,
    }


def build_aws_node_type_index(node_type_items: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Index the live ``list-node-types`` response by ``node_type_id``.

    This is the authoritative source of AWS vCPU and memory for this workspace.
    """
    index: Dict[str, Dict[str, Any]] = {}
    for item in node_type_items or []:
        node_type_id = item.get("node_type_id")
        if not node_type_id:
            continue

        # Disk attributes are nested under node_instance_type in the Databricks API response,
        # but tolerate a flat shape too in case the response format changes.
        instance_info = item.get("node_instance_type") or {}
        instance_type_id = instance_info.get("instance_type_id") or item.get("instance_type_id") or node_type_id
        local_disk_gb = to_float(instance_info.get("local_disk_size_gb")) or to_float(item.get("local_disk_size_gb")) or 0.0
        local_nvme_disks = to_int(instance_info.get("local_nvme_disks")) or to_int(item.get("local_nvme_disks")) or 0
        local_disks = to_int(instance_info.get("local_disks")) or to_int(item.get("local_disks")) or 0

        vcpu = to_int(item.get("num_cores"))
        memory_mb = to_float(item.get("memory_mb"))
        derived = parse_aws_instance_type(instance_type_id) or {}

        index[str(node_type_id)] = {
            "vcpu": vcpu if vcpu else derived.get("vcpu"),
            "memory_gb": round(memory_mb / 1024.0, 1) if memory_mb else derived.get("memory_gb"),
            "category": derived.get("category", "general_purpose"),
            "has_local_nvme": bool(local_nvme_disks or local_disks or derived.get("has_local_nvme")),
            "local_disk_gb": local_disk_gb,
            "num_gpus": to_int(item.get("num_gpus")) or derived.get("num_gpus") or 0,
            "instance_type_id": instance_type_id,
            "description": item.get("description"),
            "is_deprecated": bool(item.get("is_deprecated")),
            "spec_source": "workspace_node_types_api",
        }
    return index


AWS_NODE_TYPE_INDEX = build_aws_node_type_index(node_types)


def resolve_aws_node_spec(node_type_id: Optional[str]) -> Dict[str, Any]:
    """Resolve an AWS node type to a spec dict, recording which source was used."""
    if not node_type_id:
        return {
            "vcpu": None, "memory_gb": None, "category": None, "has_local_nvme": None,
            "num_gpus": None, "instance_type_id": None, "local_disk_gb": None,
            "spec_source": "not_specified",
        }

    key = str(node_type_id)
    if key in AWS_NODE_TYPE_INDEX:
        return dict(AWS_NODE_TYPE_INDEX[key])

    derived = parse_aws_instance_type(key)
    if derived:
        derived = dict(derived)
        derived.update({"instance_type_id": key, "spec_source": "derived_from_instance_name", "local_disk_gb": None})
        return derived

    return {
        "vcpu": None, "memory_gb": None, "category": None, "has_local_nvme": None,
        "num_gpus": None, "instance_type_id": key, "local_disk_gb": None,
        "spec_source": "unknown_node_type",
    }


print(f"AWS node type specs available from the workspace API: {len(AWS_NODE_TYPE_INDEX)}")
print(f"AWS families understood by the name-derivation fallback: {len(AWS_FAMILY_SPECS)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Azure VM catalog
# MAGIC
# MAGIC The Azure VM sizes supported by Azure Databricks, with vCPU, memory and local NVMe temp storage.
# MAGIC `armSkuName` values (`Standard_E8ds_v5`) are exactly what the Retail Prices API expects.

# COMMAND ----------

def _azure_family(
    sku_template: str,
    family: str,
    category: str,
    generation_rank: int,
    has_local_ssd: bool,
    specs: Sequence[Tuple[int, float, int, int]],
    gpu_model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Expand one Azure VM family into catalog rows.

    ``specs`` entries are ``(vcpu, memory_gb, local_ssd_gb, num_gpus)`` and ``sku_template`` is
    formatted with the vCPU count, for example ``"Standard_E{}ds_v5"``.
    """
    return [
        {
            "azure_vm_sku": sku_template.format(vcpu),
            "azure_vm_family": family,
            "category": category,
            "generation_rank": generation_rank,
            "vcpu": vcpu,
            "memory_gb": float(memory_gb),
            "memory_per_vcpu": round(memory_gb / vcpu, 2),
            "local_ssd_gb": local_ssd_gb,
            "has_local_ssd": has_local_ssd and local_ssd_gb > 0,
            "num_gpus": gpus,
            "gpu_model": gpu_model,
        }
        for vcpu, memory_gb, local_ssd_gb, gpus in specs
    ]


AZURE_VM_CATALOG: List[Dict[str, Any]] = (
    # General purpose, ~4 GiB per vCPU
    _azure_family("Standard_D{}s_v3", "Dsv3", "general_purpose", 1, False,
                  [(4, 16, 0, 0), (8, 32, 0, 0), (16, 64, 0, 0), (32, 128, 0, 0), (64, 256, 0, 0)])
    + _azure_family("Standard_D{}ds_v4", "Ddsv4", "general_purpose", 2, True,
                    [(4, 16, 150, 0), (8, 32, 300, 0), (16, 64, 600, 0), (32, 128, 1200, 0),
                     (48, 192, 1800, 0), (64, 256, 2400, 0)])
    + _azure_family("Standard_D{}ads_v5", "Dadsv5", "general_purpose", 3, True,
                    [(4, 16, 150, 0), (8, 32, 300, 0), (16, 64, 600, 0), (32, 128, 1200, 0),
                     (48, 192, 1800, 0), (64, 256, 2400, 0), (96, 384, 3600, 0)])
    + _azure_family("Standard_D{}ds_v5", "Ddsv5", "general_purpose", 4, True,
                    [(4, 16, 150, 0), (8, 32, 300, 0), (16, 64, 600, 0), (32, 128, 1200, 0),
                     (48, 192, 1800, 0), (64, 256, 2400, 0), (96, 384, 3600, 0)])
    # Memory optimized, ~8 GiB per vCPU
    + _azure_family("Standard_E{}s_v3", "Esv3", "memory_optimized", 1, False,
                    [(4, 32, 0, 0), (8, 64, 0, 0), (16, 128, 0, 0), (20, 160, 0, 0), (32, 256, 0, 0), (64, 432, 0, 0)])
    + _azure_family("Standard_E{}ds_v4", "Edsv4", "memory_optimized", 2, True,
                    [(4, 32, 150, 0), (8, 64, 300, 0), (16, 128, 600, 0), (20, 160, 750, 0),
                     (32, 256, 1200, 0), (48, 384, 1800, 0), (64, 504, 2400, 0)])
    + _azure_family("Standard_E{}ads_v5", "Eadsv5", "memory_optimized", 3, True,
                    [(4, 32, 150, 0), (8, 64, 300, 0), (16, 128, 600, 0), (20, 160, 750, 0),
                     (32, 256, 1200, 0), (48, 384, 1800, 0), (64, 512, 2400, 0), (96, 672, 3600, 0)])
    + _azure_family("Standard_E{}ds_v5", "Edsv5", "memory_optimized", 4, True,
                    [(4, 32, 150, 0), (8, 64, 300, 0), (16, 128, 600, 0), (20, 160, 750, 0),
                     (32, 256, 1200, 0), (48, 384, 1800, 0), (64, 512, 2400, 0), (96, 672, 3600, 0)])
    # Compute optimized, ~2 GiB per vCPU
    + _azure_family("Standard_F{}s_v2", "Fsv2", "compute_optimized", 2, True,
                    [(4, 8, 32, 0), (8, 16, 64, 0), (16, 32, 128, 0), (32, 64, 256, 0),
                     (48, 96, 384, 0), (64, 128, 512, 0), (72, 144, 576, 0)])
    # Storage optimized, very large local NVMe
    + _azure_family("Standard_L{}s_v2", "Lsv2", "storage_optimized", 1, True,
                    [(8, 64, 1920, 0), (16, 128, 3840, 0), (32, 256, 7680, 0),
                     (48, 384, 11520, 0), (64, 512, 15360, 0), (80, 640, 19200, 0)])
    + _azure_family("Standard_L{}s_v3", "Lsv3", "storage_optimized", 2, True,
                    [(8, 64, 1920, 0), (16, 128, 3840, 0), (32, 256, 7680, 0),
                     (48, 384, 11520, 0), (64, 512, 15360, 0), (80, 640, 19200, 0)])
    # GPU
    + _azure_family("Standard_NC{}as_T4_v3", "NCasT4v3", "gpu", 1, True,
                    [(4, 28, 176, 1), (8, 56, 352, 1), (16, 110, 352, 1), (64, 440, 2880, 4)], gpu_model="NVIDIA T4")
    + _azure_family("Standard_NC{}ads_A100_v4", "NCadsA100v4", "gpu", 2, True,
                    [(24, 220, 1123, 1), (48, 440, 2246, 2), (96, 880, 4492, 4)], gpu_model="NVIDIA A100 80GB")
    + _azure_family("Standard_ND{}asr_v4", "NDasrv4", "gpu", 3, True,
                    [(96, 900, 6000, 8)], gpu_model="NVIDIA A100 40GB")
)

AZURE_VM_INDEX: Dict[str, Dict[str, Any]] = {row["azure_vm_sku"]: row for row in AZURE_VM_CATALOG}
azure_vm_catalog_pdf = pd.DataFrame(AZURE_VM_CATALOG)

print(f"Azure VM catalog: {len(AZURE_VM_CATALOG)} SKUs across {azure_vm_catalog_pdf['azure_vm_family'].nunique()} families.")
display_pdf(azure_vm_catalog_pdf, "Azure VM catalog is empty")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Sizing engine: pick the Azure VM SKU
# MAGIC
# MAGIC The rule in plain English:
# MAGIC
# MAGIC 1. **Pick the family.** GPU nodes go to GPU VMs. Otherwise the workload's *memory per vCPU* decides:
# MAGIC    &ge;&nbsp;7&nbsp;GB &rarr; memory optimized (E), &le;&nbsp;2.5&nbsp;GB &rarr; compute optimized (F), anything
# MAGIC    in between &rarr; general purpose (D). AWS storage-optimized nodes (i3, i3en, i4i) go to the L series so the
# MAGIC    local NVMe is preserved. You can override all of this with the **VM family preference** widget.
# MAGIC 2. **Pick the size.** The *smallest* VM in that family with **at least** the same vCPU and memory as the AWS
# MAGIC    node. `like_for_like` never under-provisions; `cost_optimized` allows up to 20% less memory and flags it.
# MAGIC 3. **Keep the node count.** Driver and worker counts carry over unchanged, so the comparison is like for like.
# MAGIC 4. **Report the delta.** Every row shows the vCPU and memory difference so you can see where Azure gives you
# MAGIC    more or less headroom.

# COMMAND ----------

# Memory-per-vCPU thresholds that steer the family choice.
MEMORY_OPTIMIZED_THRESHOLD_GB_PER_VCPU = 7.0
# Ratio thresholds that decide the Azure VM family. c5n sits at ~2.67 GB/vCPU and is genuinely
# compute-optimized, so the threshold sits just above it.
COMPUTE_OPTIMIZED_THRESHOLD_GB_PER_VCPU = 2.75
# cost_optimized may land on a VM with this fraction of the requested memory.
COST_OPTIMIZED_MEMORY_TOLERANCE = 0.8
# Local NVMe of at least this size means the workload is genuinely storage-optimized.
STORAGE_OPTIMIZED_LOCAL_DISK_GB = 1000
# Flag a recommendation for review once it exceeds the AWS node by this much on vCPU or memory.
OVERPROVISION_FACTOR = 2.0
# ...but only when the absolute excess is material, so small nodes hitting the Azure minimum stay quiet.
OVERPROVISION_MIN_EXCESS_VCPU = 8
OVERPROVISION_MIN_EXCESS_GB = 32
# The smallest VM size Azure Databricks supports for a worker.
SMALLEST_AZURE_WORKER_VCPU = 4


def choose_candidate_categories(
    memory_per_vcpu: Optional[float],
    aws_category: Optional[str],
    num_gpus: Optional[int],
    local_disk_gb: Optional[float],
    family_preference: str,
) -> List[str]:
    """Return the Azure VM categories to search, in priority order."""
    if num_gpus and num_gpus > 0:
        return ["gpu"]
    if family_preference != "auto":
        return [family_preference]

    if aws_category == "storage_optimized" and (local_disk_gb is None or local_disk_gb >= STORAGE_OPTIMIZED_LOCAL_DISK_GB):
        # AWS storage-optimized families exist for their large local NVMe, so keep that on Azure.
        # An unknown disk size still counts: the family name alone is a reliable signal.
        return ["storage_optimized", "memory_optimized", "general_purpose"]
    if memory_per_vcpu is None:
        return ["general_purpose", "memory_optimized", "compute_optimized"]
    if memory_per_vcpu >= MEMORY_OPTIMIZED_THRESHOLD_GB_PER_VCPU:
        return ["memory_optimized", "storage_optimized", "general_purpose"]
    if memory_per_vcpu <= COMPUTE_OPTIMIZED_THRESHOLD_GB_PER_VCPU:
        return ["compute_optimized", "general_purpose", "memory_optimized"]
    return ["general_purpose", "memory_optimized", "compute_optimized"]


def _candidate_sort_key(candidate: Dict[str, Any]) -> Tuple[float, float, int]:
    """Smallest adequate VM first; newer generation wins an exact tie."""
    return (candidate["vcpu"], candidate["memory_gb"], -candidate["generation_rank"])


def recommend_azure_vm(
    required_vcpu: Optional[float],
    required_memory_gb: Optional[float],
    aws_category: Optional[str] = None,
    num_gpus: Optional[int] = 0,
    local_disk_gb: Optional[float] = None,
    strategy: Optional[str] = None,
    prefer_local_ssd: Optional[bool] = None,
    family_preference: Optional[str] = None,
    forced_sku: Optional[str] = None,
) -> Dict[str, Any]:
    """Recommend the Azure VM SKU that best matches one AWS node.

    Returns the chosen SKU with its specs, the rule that produced it, a confidence level, and the
    vCPU/memory delta against the AWS node. Never raises: an unmatched node returns
    ``azure_vm_sku=None`` with an explanatory note.
    """
    strategy = strategy or CONFIG["azure_sizing_strategy"]
    prefer_local_ssd = CONFIG["azure_prefer_local_ssd"] if prefer_local_ssd is None else prefer_local_ssd
    family_preference = family_preference or CONFIG["azure_vm_family_preference"]

    def result(sku: Optional[str], rule: str, confidence: str, note: str) -> Dict[str, Any]:
        spec = AZURE_VM_INDEX.get(sku or "", {})
        vcpu = spec.get("vcpu")
        memory_gb = spec.get("memory_gb")
        return {
            "azure_vm_sku": sku,
            "azure_vm_family": spec.get("azure_vm_family"),
            "azure_vm_category": spec.get("category"),
            "azure_vcpu_per_node": vcpu,
            "azure_memory_gb_per_node": memory_gb,
            "azure_local_ssd_gb_per_node": spec.get("local_ssd_gb"),
            "azure_gpus_per_node": spec.get("num_gpus"),
            "vcpu_delta_per_node": round(vcpu - required_vcpu, 2) if vcpu is not None and required_vcpu else None,
            "memory_gb_delta_per_node": round(memory_gb - required_memory_gb, 2) if memory_gb is not None and required_memory_gb else None,
            "mapping_rule": rule,
            "mapping_confidence": confidence,
            "mapping_note": note,
        }

    if forced_sku:
        if forced_sku in AZURE_VM_INDEX:
            return result(forced_sku, "forced_by_user", "user_specified", "VM SKU was set explicitly in the settings.")
        return {
            **result(None, "forced_by_user_unknown_sku", "unverified", ""),
            "azure_vm_sku": forced_sku,
            "mapping_note": (
                f"'{forced_sku}' is not in the built-in Azure VM catalog, so vCPU and memory are unknown. "
                "Pricing is still attempted; check the SKU name if no price is returned."
            ),
        }

    if not required_vcpu or not required_memory_gb:
        return result(None, "insufficient_input", "none", "AWS vCPU or memory was unknown, so no Azure SKU could be chosen.")

    memory_per_vcpu = required_memory_gb / required_vcpu
    categories = choose_candidate_categories(memory_per_vcpu, aws_category, num_gpus, local_disk_gb, family_preference)
    memory_floor = required_memory_gb * (COST_OPTIMIZED_MEMORY_TOLERANCE if strategy == "cost_optimized" else 1.0)

    for category_rank, category in enumerate(categories):
        pool = [vm for vm in AZURE_VM_CATALOG if vm["category"] == category]
        if num_gpus and num_gpus > 0:
            pool = [vm for vm in pool if (vm["num_gpus"] or 0) >= num_gpus]

        # Try the NVMe-equipped sizes first when local disk matters, then fall back to the whole family.
        pools = [[vm for vm in pool if vm["has_local_ssd"]], pool] if prefer_local_ssd else [pool]

        for pool_rank, current_pool in enumerate(pools):
            fits = [
                vm for vm in current_pool
                if vm["vcpu"] >= required_vcpu and vm["memory_gb"] >= memory_floor
            ]
            if not fits:
                continue

            best = sorted(fits, key=_candidate_sort_key)[0]
            exact_category = category_rank == 0
            relaxed_memory = best["memory_gb"] < required_memory_gb

            if relaxed_memory:
                rule, confidence = "cost_optimized_memory_relaxed", "review"
                note = (
                    f"cost_optimized allowed {best['memory_gb']:g} GB against {required_memory_gb:g} GB requested "
                    f"({best['memory_gb'] / required_memory_gb:.0%} of source memory). Confirm the workload fits."
                )
            elif exact_category and pool_rank == 0:
                rule, confidence = "capacity_match", "high"
                note = f"Smallest {best['azure_vm_family']} VM meeting {required_vcpu:g} vCPU and {required_memory_gb:g} GB."
            else:
                rule, confidence = "capacity_match_fallback_family", "medium"
                note = (
                    f"No fit in the preferred category, so {best['azure_vm_family']} ({category}) was used. "
                    "Review the family choice against the workload profile."
                )

            # Azure's fixed memory-per-vCPU ratios sometimes force a much larger VM than the workload
            # needs. Say so, because it is the difference between a like-for-like move and overspending.
            # Both a large ratio and a large absolute excess are required, so tiny nodes that simply hit
            # Azure Databricks' minimum supported size are not flagged as a problem.
            excess_vcpu = best["vcpu"] - required_vcpu
            excess_memory = best["memory_gb"] - required_memory_gb
            overprovisioned = (
                best["vcpu"] >= required_vcpu * OVERPROVISION_FACTOR and excess_vcpu >= OVERPROVISION_MIN_EXCESS_VCPU
            ) or (
                best["memory_gb"] >= required_memory_gb * OVERPROVISION_FACTOR and excess_memory >= OVERPROVISION_MIN_EXCESS_GB
            )
            if overprovisioned:
                confidence = "review"
                note += (
                    f" Note: this is {best['vcpu'] / required_vcpu:.1f}x the vCPU and "
                    f"{best['memory_gb'] / required_memory_gb:.1f}x the memory of the AWS node, because no "
                    "Azure size matches the source ratio more closely. Consider more, smaller nodes instead."
                )
            elif excess_vcpu > 0 and required_vcpu < SMALLEST_AZURE_WORKER_VCPU:
                note += (
                    f" The AWS node is smaller than {SMALLEST_AZURE_WORKER_VCPU} vCPU, which is the smallest "
                    "practical Azure Databricks worker, so the Azure node is slightly larger."
                )

            # GPU generations differ between clouds, so a count match is never the whole story.
            if num_gpus and num_gpus > 0:
                confidence = "review"
                note += (
                    f" GPU workload: {best['azure_vm_sku']} provides {best['num_gpus']} x {best.get('gpu_model') or 'GPU'} "
                    "against the AWS accelerator. Benchmark before committing, as GPU model, memory and "
                    "interconnect differ between clouds."
                )

            return result(best["azure_vm_sku"], rule, confidence, note)

    largest = sorted(
        [vm for vm in AZURE_VM_CATALOG if vm["category"] == categories[0]] or AZURE_VM_CATALOG,
        key=lambda vm: (vm["vcpu"], vm["memory_gb"]),
    )[-1]
    return result(
        largest["azure_vm_sku"],
        "largest_available_in_family",
        "low",
        (
            f"No single Azure VM matches {required_vcpu:g} vCPU / {required_memory_gb:g} GB, so the largest "
            f"{largest['azure_vm_family']} size is shown. Split the workload across more, smaller nodes, or "
            "use a specialised Azure VM family such as the M series."
        ),
    )


def size_cluster(
    worker_node_type: Optional[str],
    driver_node_type: Optional[str],
    min_workers: Optional[int],
    max_workers: Optional[int],
    single_node: bool = False,
    forced_sku: Optional[str] = None,
) -> Dict[str, Any]:
    """Size one whole cluster: driver plus workers, on AWS and on Azure.

    Returns a flat dict combining the AWS spec, the Azure recommendation and the total capacity of
    the cluster at its minimum and maximum autoscale bounds.
    """
    driver_node_type = driver_node_type or worker_node_type
    worker_spec = resolve_aws_node_spec(worker_node_type)
    driver_spec = resolve_aws_node_spec(driver_node_type)

    worker_azure = recommend_azure_vm(
        worker_spec.get("vcpu"), worker_spec.get("memory_gb"), worker_spec.get("category"),
        worker_spec.get("num_gpus"), worker_spec.get("local_disk_gb"), forced_sku=forced_sku,
    )
    driver_azure = recommend_azure_vm(
        driver_spec.get("vcpu"), driver_spec.get("memory_gb"), driver_spec.get("category"),
        driver_spec.get("num_gpus"), driver_spec.get("local_disk_gb"), forced_sku=forced_sku,
    )

    if single_node:
        min_workers = max_workers = 0

    min_workers = 0 if min_workers is None else max(int(min_workers), 0)
    max_workers = min_workers if max_workers is None else max(int(max_workers), min_workers)

    min_nodes = min_workers + 1
    max_nodes = max_workers + 1

    def total(per_node: Optional[float], workers: int) -> Optional[float]:
        """Total capacity = driver + workers, using the worker figure for every worker node."""
        if per_node is None:
            return None
        driver_value = per_node if single_node else per_node
        return round(driver_value + per_node * workers, 1)

    return {
        "aws_worker_node_type": worker_node_type,
        "aws_driver_node_type": driver_node_type,
        "aws_worker_vcpu": worker_spec.get("vcpu"),
        "aws_worker_memory_gb": worker_spec.get("memory_gb"),
        "aws_worker_gpus": worker_spec.get("num_gpus"),
        "aws_worker_category": worker_spec.get("category"),
        "aws_worker_local_disk_gb": worker_spec.get("local_disk_gb"),
        "aws_spec_source": worker_spec.get("spec_source"),
        "min_workers": min_workers,
        "max_workers": max_workers,
        "min_nodes_including_driver": min_nodes,
        "max_nodes_including_driver": max_nodes,
        "single_node": single_node,
        "azure_worker_vm_sku": worker_azure["azure_vm_sku"],
        "azure_driver_vm_sku": driver_azure["azure_vm_sku"],
        "azure_vm_family": worker_azure["azure_vm_family"],
        "azure_vcpu_per_worker": worker_azure["azure_vcpu_per_node"],
        "azure_memory_gb_per_worker": worker_azure["azure_memory_gb_per_node"],
        "azure_local_ssd_gb_per_worker": worker_azure["azure_local_ssd_gb_per_node"],
        "vcpu_delta_per_node": worker_azure["vcpu_delta_per_node"],
        "memory_gb_delta_per_node": worker_azure["memory_gb_delta_per_node"],
        "aws_total_vcpu_at_max": total(worker_spec.get("vcpu"), max_workers),
        "aws_total_memory_gb_at_max": total(worker_spec.get("memory_gb"), max_workers),
        "azure_total_vcpu_at_min": total(worker_azure["azure_vcpu_per_node"], min_workers),
        "azure_total_vcpu_at_max": total(worker_azure["azure_vcpu_per_node"], max_workers),
        "azure_total_memory_gb_at_min": total(worker_azure["azure_memory_gb_per_node"], min_workers),
        "azure_total_memory_gb_at_max": total(worker_azure["azure_memory_gb_per_node"], max_workers),
        "mapping_rule": worker_azure["mapping_rule"],
        "mapping_confidence": worker_azure["mapping_confidence"],
        "mapping_note": worker_azure["mapping_note"],
    }


# Show the mapping rules at work on a representative set of AWS node types.
_mapping_demo_types = [
    node_type for node_type in ["m5d.2xlarge", "m5d.4xlarge", "r5d.4xlarge", "r6id.8xlarge",
                                "c5d.4xlarge", "i3.4xlarge", "i3en.6xlarge", "g4dn.4xlarge"]
]
mapping_examples_pdf = pd.DataFrame(
    [
        {
            "aws_node_type": node_type,
            "aws_vcpu": spec.get("vcpu"),
            "aws_memory_gb": spec.get("memory_gb"),
            "aws_spec_source": spec.get("spec_source"),
            "azure_vm_sku": rec["azure_vm_sku"],
            "azure_vcpu": rec["azure_vcpu_per_node"],
            "azure_memory_gb": rec["azure_memory_gb_per_node"],
            "mapping_rule": rec["mapping_rule"],
            "mapping_confidence": rec["mapping_confidence"],
        }
        for node_type, spec, rec in (
            (
                node_type,
                resolve_aws_node_spec(node_type),
                recommend_azure_vm(
                    resolve_aws_node_spec(node_type).get("vcpu"),
                    resolve_aws_node_spec(node_type).get("memory_gb"),
                    resolve_aws_node_spec(node_type).get("category"),
                    resolve_aws_node_spec(node_type).get("num_gpus"),
                    resolve_aws_node_spec(node_type).get("local_disk_gb"),
                ),
            )
            for node_type in _mapping_demo_types
        )
    ]
)

print("Worked examples of the AWS -> Azure mapping rules:")
display_pdf(mapping_examples_pdf, "No mapping examples produced")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Azure VM pricing lookup
# MAGIC
# MAGIC Prices come from the **public Azure Retail Prices API** at `https://prices.azure.com`. It needs no Azure
# MAGIC subscription, login, key or SDK &mdash; just outbound HTTPS.
# MAGIC
# MAGIC For every VM SKU the notebook collects, in your chosen currency:
# MAGIC
# MAGIC | Column | Meaning |
# MAGIC | --- | --- |
# MAGIC | `payg_hourly` | Pay-as-you-go Linux, on demand. |
# MAGIC | `spot_hourly` | Azure Spot Linux. Can be evicted, so use it for fault-tolerant workers only. |
# MAGIC | `savings_plan_1y_hourly` / `savings_plan_3y_hourly` | Azure savings plan for compute. |
# MAGIC | `reserved_1y_hourly` / `reserved_3y_hourly` | Reserved instance, amortised to an hourly rate. |
# MAGIC
# MAGIC Linux rates only &mdash; Windows and Dev/Test meters are filtered out, because Databricks runs Linux.
# MAGIC
# MAGIC If a SKU or region has no price, the notebook records a **warning with a suggested alternative** instead of
# MAGIC failing, and the sizing tables still render.

# COMMAND ----------

# The Retail Prices API rejects filters with too many OR clauses, so SKUs are requested in batches.
AZURE_PRICE_SKU_BATCH_SIZE = 12

RESERVATION_TERM_YEARS = {"1 Year": 1, "3 Years": 3, "5 Years": 5}
SAVINGS_PLAN_TERM_COLUMNS = {"1 Year": "savings_plan_1y_hourly", "3 Years": "savings_plan_3y_hourly"}

PRICING_MODEL_COLUMNS = {
    "payg": "payg_hourly",
    "spot": "spot_hourly",
    "savings_plan_1y": "savings_plan_1y_hourly",
    "savings_plan_3y": "savings_plan_3y_hourly",
    "reserved_1y": "reserved_1y_hourly",
    "reserved_3y": "reserved_3y_hourly",
}

PRICING_MODEL_LABELS = {
    "payg": "pay-as-you-go (Linux, on demand)",
    "spot": "Azure Spot (Linux, evictable)",
    "savings_plan_1y": "1-year Azure savings plan for compute",
    "savings_plan_3y": "3-year Azure savings plan for compute",
    "reserved_1y": "1-year reserved instance (amortised hourly)",
    "reserved_3y": "3-year reserved instance (amortised hourly)",
}


def _is_windows_meter(item: Dict[str, Any]) -> bool:
    """Databricks runs Linux, so Windows-licensed meters must be excluded."""
    return "windows" in str(item.get("productName", "")).lower()


def _is_spot_meter(item: Dict[str, Any]) -> bool:
    return "spot" in str(item.get("meterName", "")).lower()


def _is_low_priority_meter(item: Dict[str, Any]) -> bool:
    """Legacy Batch 'Low Priority' meters are not Azure Spot and must not be mixed in."""
    return "low priority" in str(item.get("meterName", "")).lower()


def parse_azure_vm_prices(items: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Turn raw Retail Prices rows into one record per ``(armSkuName, armRegionName)``."""
    prices: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for item in items:
        sku = item.get("armSkuName")
        region = item.get("armRegionName")
        if not sku or not region:
            continue
        # Dev/Test rates need an eligible subscription, so they are never used for a migration estimate.
        if item.get("type") == "DevTestConsumption" or _is_windows_meter(item) or _is_low_priority_meter(item):
            continue

        record = prices.setdefault(
            (sku, region),
            {
                "azure_vm_sku": sku,
                "azure_region": region,
                "azure_region_label": item.get("location"),
                "currency": item.get("currencyCode"),
                "payg_hourly": None,
                "spot_hourly": None,
                "savings_plan_1y_hourly": None,
                "savings_plan_3y_hourly": None,
                "reserved_1y_hourly": None,
                "reserved_3y_hourly": None,
                "meter_name": None,
                "product_name": None,
                "price_source": "azure_retail_prices_api",
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
        )

        unit_price = to_float(item.get("unitPrice"))
        if unit_price is None:
            continue

        if item.get("type") == "Reservation":
            years = RESERVATION_TERM_YEARS.get(str(item.get("reservationTerm")))
            if years:
                # The API returns the total price for the whole term, so amortise it back to an hour.
                record[f"reserved_{years}y_hourly"] = round(unit_price / (HOURS_PER_YEAR * years), 6)
            continue

        if item.get("type") != "Consumption":
            continue

        if _is_spot_meter(item):
            record["spot_hourly"] = round(unit_price, 6)
            continue

        record["payg_hourly"] = round(unit_price, 6)
        record["meter_name"] = item.get("meterName")
        record["product_name"] = item.get("productName")
        for plan in item.get("savingsPlan") or []:
            column = SAVINGS_PLAN_TERM_COLUMNS.get(str(plan.get("term")))
            plan_price = to_float(plan.get("unitPrice") if plan.get("unitPrice") is not None else plan.get("retailPrice"))
            if column and plan_price is not None:
                record[column] = round(plan_price, 6)

    return prices


class AzureVmPriceBook:
    """Fetches and caches Azure VM prices for a set of SKUs and regions.

    Results are cached in memory for the life of the notebook run, so re-running a cell does not
    re-query the API. Every failure is captured in :attr:`warnings` rather than raised.
    """

    def __init__(self, currency: str = "USD", enabled: bool = True, timeout: int = 30, max_retries: int = 3):
        self.currency = currency
        self.enabled = enabled
        self.timeout = timeout
        self.max_retries = max_retries
        self.prices: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.requested: Dict[str, set] = {}
        self.warnings: List[Dict[str, Any]] = []
        self.api_reachable: Optional[bool] = None

    def _warn(self, sku: Optional[str], region: str, issue: str, recommended_action: str) -> None:
        self.warnings.append(
            {
                "azure_vm_sku": sku,
                "azure_region": region,
                "issue": issue[:1000],
                "recommended_action": recommended_action[:1000],
            }
        )
        print(f"[warn] {issue}")

    def fetch(self, skus: Iterable[str], region: str) -> None:
        """Fetch prices for ``skus`` in ``region``, skipping anything already cached."""
        region_cache = self.requested.setdefault(region, set())
        wanted = sorted({sku for sku in skus if sku and sku not in region_cache})
        if not wanted:
            return

        if not self.enabled:
            self._warn(
                None, region,
                f"Live Azure pricing is switched off, so {len(wanted)} VM SKU(s) in {region} have no price.",
                "Set the 'D3 Fetch live Azure prices' widget to true and re-run.",
            )
            region_cache.update(wanted)
            return

        for start in range(0, len(wanted), AZURE_PRICE_SKU_BATCH_SIZE):
            batch = wanted[start:start + AZURE_PRICE_SKU_BATCH_SIZE]
            sku_filter = " or ".join(f"armSkuName eq '{sku}'" for sku in batch)
            odata_filter = (
                f"serviceName eq 'Virtual Machines' and armRegionName eq '{region}' and ({sku_filter})"
            )
            try:
                items = fetch_azure_price_items(
                    odata_filter, currency_code=self.currency, timeout=self.timeout, max_retries=self.max_retries
                )
                self.api_reachable = True
            except Exception as exc:
                self.api_reachable = False
                AZURE_PRICING_DIAGNOSTICS["reachable"] = False
                AZURE_PRICING_DIAGNOSTICS["last_error"] = str(exc)[:400]
                self._warn(
                    None, region,
                    f"Could not reach the Azure Retail Prices API for {len(batch)} SKU(s) in {region}: {str(exc)[:250]}",
                    "Check outbound HTTPS access to prices.azure.com from this workspace, then re-run. "
                    "Sizing output is unaffected; only the cost columns are blank.",
                )
                region_cache.update(batch)
                continue

            self.prices.update(parse_azure_vm_prices(items))
            region_cache.update(batch)

            for sku in batch:
                if (sku, region) not in self.prices:
                    self._warn(
                        sku, region,
                        f"No Azure price found for {sku} in {region}.",
                        self.suggest_alternative(sku, region),
                    )

    def suggest_alternative(self, sku: str, region: str) -> str:
        """Suggest a next step when a SKU has no price in the chosen region."""
        spec = AZURE_VM_INDEX.get(sku)
        if not spec:
            return (
                f"'{sku}' is not a recognised Azure VM size. Check the spelling against the Azure VM catalog "
                "table in section 10, for example 'Standard_E8ds_v5'."
            )

        siblings = sorted(
            [
                vm["azure_vm_sku"]
                for vm in AZURE_VM_CATALOG
                if vm["category"] == spec["category"]
                and vm["azure_vm_family"] != spec["azure_vm_family"]
                and vm["vcpu"] >= spec["vcpu"]
                and vm["memory_gb"] >= spec["memory_gb"]
            ]
        )
        alternatives = ", ".join(siblings[:3]) if siblings else "a different VM family"
        return (
            f"{sku} is probably not offered in {region}. Try an equivalent size from another family "
            f"({alternatives}), or price a different region such as uaecentral or westeurope using the "
            "'C4 Compare extra regions' widget. Check region availability at "
            "https://azure.microsoft.com/global-infrastructure/services/"
        )

    def get(self, sku: Optional[str], region: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Return the cached price record for a SKU in a region, or ``None``."""
        if not sku:
            return None
        return self.prices.get((sku, region or CONFIG["azure_region"]))

    def hourly(self, sku: Optional[str], model: Optional[str] = None, region: Optional[str] = None) -> Optional[float]:
        """Return the hourly rate for one VM under the requested pricing model."""
        record = self.get(sku, region)
        if not record:
            return None
        return to_float(record.get(PRICING_MODEL_COLUMNS[model or CONFIG["azure_pricing_model"]]))

    def to_frame(self) -> pd.DataFrame:
        """All cached prices as a pandas frame."""
        return pd.DataFrame(sorted(self.prices.values(), key=lambda row: (row["azure_region"], row["azure_vm_sku"])))

    def warnings_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.warnings)


price_book = AzureVmPriceBook(
    currency=CONFIG["azure_currency"],
    enabled=CONFIG["run_azure_pricing"],
    timeout=CONFIG["azure_price_timeout_seconds"],
    max_retries=CONFIG["azure_price_max_retries"],
)

HEADLINE_PRICE_COLUMN = PRICING_MODEL_COLUMNS[CONFIG["azure_pricing_model"]]


def price_cluster_nodes(
    worker_sku: Optional[str],
    driver_sku: Optional[str],
    worker_count: Optional[int],
    hours_per_month: Optional[float] = None,
    pricing_model: Optional[str] = None,
    region: Optional[str] = None,
    prefix: str = "",
) -> Dict[str, Any]:
    """Cost one cluster: driver + workers, hourly and monthly.

    ``hours_per_month`` defaults to the configured billing month (730 hours = always on).
    Returns ``None`` cost values when a price is unavailable, together with a status column.
    """
    pricing_model = pricing_model or CONFIG["azure_pricing_model"]
    region = region or CONFIG["azure_region"]
    hours = CONFIG["hours_per_month"] if hours_per_month is None else hours_per_month
    worker_count = 0 if worker_count is None else max(int(worker_count), 0)

    driver_rate = price_book.hourly(driver_sku, pricing_model, region)
    worker_rate = price_book.hourly(worker_sku, pricing_model, region)

    if driver_rate is None and worker_rate is None:
        status = "no_price_available"
    elif driver_rate is None or (worker_count > 0 and worker_rate is None):
        status = "partial_price_available"
    else:
        status = "priced"

    cluster_hourly = None
    if driver_rate is not None and (worker_count == 0 or worker_rate is not None):
        cluster_hourly = driver_rate + (worker_rate or 0.0) * worker_count

    return {
        f"{prefix}azure_pricing_model": pricing_model,
        f"{prefix}azure_price_region": region,
        f"{prefix}azure_currency": CONFIG["azure_currency"],
        f"{prefix}azure_driver_vm_hourly": round_or_none(driver_rate, 6),
        f"{prefix}azure_worker_vm_hourly": round_or_none(worker_rate, 6),
        f"{prefix}azure_cluster_vm_hourly": round_or_none(cluster_hourly, 4),
        f"{prefix}billable_hours_per_month": round_or_none(hours, 2),
        f"{prefix}azure_cluster_vm_monthly": round_or_none(cluster_hourly * hours, 2) if cluster_hourly is not None else None,
        f"{prefix}azure_price_status": status,
    }


print(f"Azure VM pricing: {'enabled' if CONFIG['run_azure_pricing'] else 'DISABLED by the D3 widget'}")
print(f"  Endpoint       : {AZURE_PRICES_ENDPOINT} (public, no credentials required)")
print(f"  Region         : {CONFIG['azure_region']} ({CONFIG['azure_region_label']})")
print(f"  Currency       : {CONFIG['azure_currency']}")
print(f"  Headline model : {CONFIG['azure_pricing_model']} - {PRICING_MODEL_LABELS[CONFIG['azure_pricing_model']]}")
print(f"  Monthly basis  : {CONFIG['hours_per_month']:g} hours per month")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Quick estimator
# MAGIC
# MAGIC A single cluster, sized and priced straight from the widget bar. This works with **no permissions at all**,
# MAGIC so it is the fastest way to answer "what would this cost on Azure?" in a meeting.
# MAGIC
# MAGIC Change **B1 Workload size**, **B2 Worker count** or **B5 VM family preference** and re-run this cell.

# COMMAND ----------

def resolve_quick_workload() -> Dict[str, Any]:
    """Turn the quick-estimator widgets into a per-node vCPU and memory requirement."""
    size = CONFIG["workload_size"]
    if size == "custom":
        return {
            "workload_size": "custom",
            "vcpu_per_node": float(CONFIG["quick_custom_vcpu_per_node"]),
            "memory_gb_per_node": float(CONFIG["quick_custom_memory_gb_per_node"]),
            "input_source": "custom widgets B3 and B4",
        }
    preset = WORKLOAD_SIZE_PRESETS[size]
    return {
        "workload_size": size,
        "vcpu_per_node": float(preset["vcpu_per_node"]),
        "memory_gb_per_node": float(preset["memory_gb_per_node"]),
        "input_source": f"'{size}' preset",
    }


quick_workload = resolve_quick_workload()
quick_worker_count = CONFIG["quick_num_workers"]

quick_recommendation = recommend_azure_vm(
    required_vcpu=quick_workload["vcpu_per_node"],
    required_memory_gb=quick_workload["memory_gb_per_node"],
    forced_sku=CONFIG["azure_vm_sku_override"],
)

quick_worker_sku = quick_recommendation["azure_vm_sku"]
quick_driver_sku = quick_worker_sku  # Databricks defaults the driver to the worker size.

price_book.fetch([sku for sku in {quick_worker_sku, quick_driver_sku} if sku], CONFIG["azure_region"])

quick_costs = price_cluster_nodes(quick_worker_sku, quick_driver_sku, quick_worker_count)
quick_total_nodes = quick_worker_count + 1
quick_vcpu_total = (quick_recommendation["azure_vcpu_per_node"] or 0) * quick_total_nodes or None
quick_memory_total = (quick_recommendation["azure_memory_gb_per_node"] or 0) * quick_total_nodes or None

quick_estimate_pdf = pd.DataFrame(
    [
        {"section": "Input", "item": "Workload size", "value": f"{quick_workload['workload_size']} ({quick_workload['input_source']})"},
        {"section": "Input", "item": "Requested vCPU per node", "value": f"{quick_workload['vcpu_per_node']:g}"},
        {"section": "Input", "item": "Requested memory per node", "value": f"{quick_workload['memory_gb_per_node']:g} GB"},
        {"section": "Input", "item": "Worker nodes", "value": f"{quick_worker_count}"},
        {"section": "Input", "item": "Azure region", "value": f"{CONFIG['azure_region']} ({CONFIG['azure_region_label']})"},
        {"section": "Assumption", "item": "Sizing strategy", "value": CONFIG["azure_sizing_strategy"]},
        {"section": "Assumption", "item": "VM family preference", "value": CONFIG["azure_vm_family_preference"]},
        {"section": "Assumption", "item": "Driver VM size", "value": "same as the worker VM size"},
        {"section": "Assumption", "item": "Hours per month", "value": f"{CONFIG['hours_per_month']:g} (always-on basis is 730)"},
        {"section": "Assumption", "item": "Pricing model", "value": PRICING_MODEL_LABELS[CONFIG["azure_pricing_model"]]},
        {"section": "Recommendation", "item": "Azure VM SKU (worker)", "value": quick_worker_sku or "none found"},
        {"section": "Recommendation", "item": "Azure VM SKU (driver)", "value": quick_driver_sku or "none found"},
        {"section": "Recommendation", "item": "Azure VM family", "value": quick_recommendation["azure_vm_family"] or NOT_AVAILABLE},
        {"section": "Recommendation", "item": "Worker count", "value": f"{quick_worker_count}"},
        {"section": "Recommendation", "item": "Total nodes including driver", "value": f"{quick_total_nodes}"},
        {"section": "Recommendation", "item": "vCPU per node", "value": f"{quick_recommendation['azure_vcpu_per_node'] or 'n/a'}"},
        {"section": "Recommendation", "item": "Memory per node", "value": f"{quick_recommendation['azure_memory_gb_per_node'] or 'n/a'} GB"},
        {"section": "Recommendation", "item": "Local NVMe per node", "value": f"{quick_recommendation['azure_local_ssd_gb_per_node'] or 0} GB"},
        {"section": "Recommendation", "item": "Total cluster vCPU", "value": f"{quick_vcpu_total or 'n/a'}"},
        {"section": "Recommendation", "item": "Total cluster memory", "value": f"{quick_memory_total or 'n/a'} GB"},
        {"section": "Recommendation", "item": "Why this SKU", "value": quick_recommendation["mapping_note"]},
        {"section": "Cost", "item": "VM cost per node per hour", "value": money(quick_costs["azure_worker_vm_hourly"], digits=4)},
        {"section": "Cost", "item": "Cluster VM cost per hour", "value": money(quick_costs["azure_cluster_vm_hourly"], digits=4)},
        {"section": "Cost", "item": f"Cluster VM cost per month ({CONFIG['hours_per_month']:g} h)", "value": money(quick_costs["azure_cluster_vm_monthly"])},
        {"section": "Cost", "item": "Price status", "value": quick_costs["azure_price_status"]},
    ]
)

print("=" * 78)
print(f"QUICK ESTIMATE - {quick_workload['workload_size']} workload, {quick_worker_count} workers + 1 driver")
print("=" * 78)
print(f"  Azure region        : {CONFIG['azure_region']}  ({CONFIG['azure_region_label']})")
print(f"  Recommended VM SKU  : {quick_worker_sku or 'none found'}")
print(f"  Cluster capacity    : {quick_total_nodes} nodes, {quick_vcpu_total or 'n/a'} vCPU, {quick_memory_total or 'n/a'} GB RAM")
print(f"  VM cost per hour    : {money(quick_costs['azure_cluster_vm_hourly'], digits=4)}")
print(f"  VM cost per month   : {money(quick_costs['azure_cluster_vm_monthly'])}   ({CONFIG['hours_per_month']:g} hours)")
print(f"  Pricing model       : {PRICING_MODEL_LABELS[CONFIG['azure_pricing_model']]}")
if quick_costs["azure_price_status"] != "priced":
    print("  [warn] No live price was returned. See the pricing warnings table below for the suggested fix.")
print("  Note: Azure VM cost only. Databricks DBU cost is estimated separately in section 17.")
print("=" * 78)

display_pdf(quick_estimate_pdf, "Quick estimate could not be produced")

# COMMAND ----------

# MAGIC %md
# MAGIC ### All pricing options for the recommended SKU
# MAGIC
# MAGIC The same cluster under every Azure commercial model, so you can see the trade-off between commitment and
# MAGIC price. Spot is the cheapest but is evictable, so it suits fault-tolerant workers rather than drivers.

# COMMAND ----------

def build_pricing_options(worker_sku: Optional[str], driver_sku: Optional[str], worker_count: int, region: Optional[str] = None) -> pd.DataFrame:
    """Cost one cluster under every supported Azure pricing model."""
    rows = []
    payg_monthly = None
    for model in AZURE_PRICING_MODELS:
        costs = price_cluster_nodes(worker_sku, driver_sku, worker_count, pricing_model=model, region=region)
        monthly = costs["azure_cluster_vm_monthly"]
        if model == "payg":
            payg_monthly = monthly
        rows.append(
            {
                "pricing_model": model,
                "description": PRICING_MODEL_LABELS[model],
                "vm_hourly_per_node": costs["azure_worker_vm_hourly"],
                "cluster_hourly": costs["azure_cluster_vm_hourly"],
                "cluster_monthly": monthly,
                "saving_vs_payg_pct": (
                    round((1 - monthly / payg_monthly) * 100, 1)
                    if monthly is not None and payg_monthly not in (None, 0)
                    else None
                ),
                "currency": CONFIG["azure_currency"],
                "status": costs["azure_price_status"],
            }
        )
    return pd.DataFrame(rows)


quick_pricing_options_pdf = build_pricing_options(quick_worker_sku, quick_driver_sku, quick_worker_count)
display_pdf(quick_pricing_options_pdf, "No pricing options available")

pricing_warnings_pdf = price_book.warnings_frame()
if not pricing_warnings_pdf.empty:
    print("Pricing warnings - each row includes a suggested next step:")
display_pdf(pricing_warnings_pdf, "No pricing warnings. Every requested SKU returned a price.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Optional: compare other Azure regions
# MAGIC
# MAGIC Set the **C4 Compare extra regions** widget to something like `uaecentral,westeurope,northeurope` to price
# MAGIC the same cluster elsewhere. Leave it empty to skip.

# COMMAND ----------

region_comparison_pdf = pd.DataFrame()

if CONFIG["azure_comparison_regions"] and quick_worker_sku:
    comparison_rows = []
    for region in [CONFIG["azure_region"]] + CONFIG["azure_comparison_regions"]:
        price_book.fetch([quick_worker_sku, quick_driver_sku], region)
        costs = price_cluster_nodes(quick_worker_sku, quick_driver_sku, quick_worker_count, region=region)
        comparison_rows.append(
            {
                "azure_region": region,
                "region_label": AZURE_REGIONS.get(region, region),
                "is_selected_region": region == CONFIG["azure_region"],
                "azure_vm_sku": quick_worker_sku,
                "vm_hourly_per_node": costs["azure_worker_vm_hourly"],
                "cluster_hourly": costs["azure_cluster_vm_hourly"],
                "cluster_monthly": costs["azure_cluster_vm_monthly"],
                "currency": CONFIG["azure_currency"],
                "pricing_model": CONFIG["azure_pricing_model"],
                "status": costs["azure_price_status"],
            }
        )
    region_comparison_pdf = pd.DataFrame(comparison_rows)
    display_pdf(region_comparison_pdf, "No region comparison produced")
else:
    print("No comparison regions selected. Set the 'C4 Compare extra regions' widget to compare, for example: uaecentral,westeurope")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Actual workload telemetry
# MAGIC
# MAGIC Cluster definitions tell you how big a cluster *can* be. `system.compute.node_timeline` tells you how many
# MAGIC node-hours were **actually consumed**, which is what drives the bill. When this table is readable the cost
# MAGIC model uses measured node-hours; otherwise it falls back to the assumed monthly hours in the settings cell
# MAGIC and says so in the `hours_source` column.
# MAGIC
# MAGIC Average CPU and memory utilisation are also collected. They do not change the recommendation, but a cluster
# MAGIC sitting at 10% CPU is an obvious right-sizing candidate and is flagged as such.

# COMMAND ----------

sql_errors: List[Dict[str, Any]] = []

BILLING_PERMISSION_ACTION = (
    "Grant SELECT on the system tables (system.billing.usage, system.billing.list_prices, "
    "system.compute.clusters, system.compute.node_timeline), or ask a Databricks account admin to "
    "enable system tables and provide access."
)


def record_sql_error(name: str, exc: Exception, recommended_action: str) -> None:
    """Record a failed system-table query with the fix, instead of raising."""
    sql_errors.append({"query_name": name, "message": str(exc)[:2000], "recommended_action": recommended_action})
    print(f"[skip] {name}: {str(exc)[:400]}")


# Report an intentional or environmental skip once, rather than once per query.
SYSTEM_TABLES_SKIP_REASON: Optional[str] = None
if not CONFIG["run_billing_usage"]:
    SYSTEM_TABLES_SKIP_REASON = "the 'D2 Query system tables' widget is set to false"
elif not HAS_SPARK:
    SYSTEM_TABLES_SKIP_REASON = "Spark is not available outside a Databricks cluster"

if SYSTEM_TABLES_SKIP_REASON:
    print(
        f"[info] System tables were not queried because {SYSTEM_TABLES_SKIP_REASON}. "
        "Cluster sizing still runs; monthly cost falls back to the assumed hours in the settings cell."
    )


def safe_sql(name: str, query: str, recommended_action: str = BILLING_PERMISSION_ACTION):
    """Run a Spark SQL query, returning ``None`` and recording the reason when it cannot run."""
    if not CONFIG["run_billing_usage"] or not HAS_SPARK:
        return None  # Already reported once by SYSTEM_TABLES_SKIP_REASON, so do not repeat it per query.
    try:
        return spark.sql(query)
    except Exception as exc:
        record_sql_error(name, exc, recommended_action)
        return None


def spark_to_pdf(df, limit: int = 100000) -> pd.DataFrame:
    """Collect a Spark frame into pandas, returning an empty frame on any failure."""
    if df is None:
        return pd.DataFrame()
    try:
        return df.limit(limit).toPandas()
    except Exception as exc:
        print(f"[warn] Could not collect results to pandas: {str(exc)[:300]}")
        return pd.DataFrame()


def table_columns(table_name: str) -> List[str]:
    """Return a table's column names, or ``[]`` when it cannot be described."""
    if not HAS_SPARK or not CONFIG["run_billing_usage"]:
        return []
    try:
        rows = spark.sql(f"DESCRIBE TABLE {table_name}").collect()
        return [row["col_name"] for row in rows if row["col_name"] and not row["col_name"].startswith("#")]
    except Exception as exc:
        record_sql_error(f"describe_{table_name.replace('.', '_')}", exc, BILLING_PERMISSION_ACTION)
        return []


DAYS_PER_MONTH = 30.44  # average calendar month, used to scale the lookback window to a month
lookback_days = float(CONFIG["usage_lookback_days"])

node_timeline_cols = set(table_columns("system.compute.node_timeline"))
compute_clusters_cols = set(table_columns("system.compute.clusters"))

node_hours_pdf = pd.DataFrame()

if node_timeline_cols:
    driver_expr = "driver" if "driver" in node_timeline_cols else "CAST(NULL AS BOOLEAN)"
    cpu_expr = (
        "cpu_user_percent + cpu_system_percent"
        if {"cpu_user_percent", "cpu_system_percent"} <= node_timeline_cols
        else "CAST(NULL AS DOUBLE)"
    )
    mem_expr = "mem_used_percent" if "mem_used_percent" in node_timeline_cols else "CAST(NULL AS DOUBLE)"

    node_hours_df = safe_sql(
        "cluster_node_hours",
        f"""
        SELECT
          cluster_id,
          COUNT(*) / 60.0                                                     AS total_node_hours,
          SUM(CASE WHEN {driver_expr} THEN 1 ELSE 0 END) / 60.0               AS driver_node_hours,
          SUM(CASE WHEN {driver_expr} THEN 0 ELSE 1 END) / 60.0               AS worker_node_hours,
          COUNT(DISTINCT instance_id)                                         AS distinct_instances,
          COUNT(DISTINCT date_trunc('HOUR', start_time))                      AS active_hours,
          AVG({cpu_expr})                                                     AS avg_cpu_utilization,
          percentile_approx({cpu_expr}, 0.95)                                 AS p95_cpu_utilization,
          AVG({mem_expr})                                                     AS avg_memory_utilization,
          percentile_approx({mem_expr}, 0.95)                                 AS p95_memory_utilization,
          MIN(start_time)                                                     AS first_seen,
          MAX(end_time)                                                       AS last_seen
        FROM system.compute.node_timeline
        WHERE start_time >= DATE({sql_quote(start_date)})
        GROUP BY cluster_id
        ORDER BY total_node_hours DESC
        """,
    )
    node_hours_pdf = spark_to_pdf(node_hours_df)

    if not node_hours_pdf.empty:
        # The system table reports utilisation as a fraction in some runtimes and as a percentage in
        # others. Normalise to a 0-100 percentage so the output is unambiguous.
        for column in ["avg_cpu_utilization", "p95_cpu_utilization", "avg_memory_utilization", "p95_memory_utilization"]:
            if column in node_hours_pdf.columns:
                values = pd.to_numeric(node_hours_pdf[column], errors="coerce")
                if values.notna().any() and values.max() <= 1.5:
                    values = values * 100.0
                node_hours_pdf[column.replace("utilization", "utilization_percent")] = values.round(1)
                node_hours_pdf.drop(columns=[column], inplace=True)

        for column in ["total_node_hours", "driver_node_hours", "worker_node_hours", "active_hours"]:
            if column not in node_hours_pdf.columns:
                continue
            monthly_column = f"monthly_{column}"
            node_hours_pdf[monthly_column] = (
                pd.to_numeric(node_hours_pdf[column], errors="coerce") / lookback_days * DAYS_PER_MONTH
            ).round(2)

        print(
            f"Measured {node_hours_pdf['total_node_hours'].sum():,.0f} node-hours across "
            f"{len(node_hours_pdf)} clusters over the last {CONFIG['usage_lookback_days']} days."
        )
else:
    print("[info] system.compute.node_timeline is unavailable, so assumed monthly hours will be used instead.")

display_pdf(node_hours_pdf, "No node-hour telemetry available")

NODE_HOURS_INDEX: Dict[str, Dict[str, Any]] = (
    {str(row["cluster_id"]): row for row in node_hours_pdf.to_dict("records")} if not node_hours_pdf.empty else {}
)

# COMMAND ----------

# Historical cluster definitions, which reach further back than the clusters/list API.
historical_clusters_pdf = pd.DataFrame()

if compute_clusters_cols:
    historical_clusters_df = safe_sql(
        "historical_cluster_definitions",
        f"""
        WITH ranked AS (
          SELECT
            cluster_id, cluster_name, owned_by, cluster_source, driver_node_type, worker_node_type,
            worker_count, min_autoscale_workers, max_autoscale_workers, auto_termination_minutes,
            dbr_version, policy_id, delete_time, change_time,
            ROW_NUMBER() OVER (PARTITION BY cluster_id ORDER BY change_time DESC) AS rn
          FROM system.compute.clusters
          WHERE change_time >= DATE({sql_quote(start_date)})
        )
        SELECT * EXCEPT (rn) FROM ranked WHERE rn = 1
        """,
    )
    historical_clusters_pdf = spark_to_pdf(historical_clusters_df)
    if not historical_clusters_pdf.empty:
        print(f"Found {len(historical_clusters_pdf)} cluster definitions in system.compute.clusters.")

display_pdf(historical_clusters_pdf, "No historical cluster definitions available")

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 3 &mdash; Price the Azure target

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Recommended Azure sizing and cost for every discovered cluster
# MAGIC
# MAGIC Each AWS cluster and job cluster becomes one row: the AWS node type, the recommended Azure VM SKU, the node
# MAGIC count, total vCPU and memory, and the hourly and monthly VM cost.
# MAGIC
# MAGIC `hours_source` tells you how the monthly figure was reached:
# MAGIC
# MAGIC - `measured_node_hours` &mdash; real consumption from `system.compute.node_timeline`. Most accurate.
# MAGIC - `measured_node_hours_driver_estimated` &mdash; real node-hours, but the runtime did not flag which node
# MAGIC   was the driver, so driver time was taken as the cluster's active wall-clock hours.
# MAGIC - `assumed_interactive` / `assumed_job` &mdash; the assumption from the settings cell, because no telemetry
# MAGIC   was available for that cluster.

# COMMAND ----------

def is_single_node_cluster(config: Dict[str, Any]) -> bool:
    """A Databricks single-node cluster runs the driver only."""
    spark_conf = config.get("spark_conf") or {}
    if spark_conf.get("spark.databricks.cluster.profile") == "singleNode":
        return True
    custom_tags = config.get("custom_tags") or {}
    if str(custom_tags.get("ResourceClass", "")).lower() == "singlenode":
        return True
    return to_int(config.get("num_workers")) == 0 and not (config.get("autoscale") or {})


def worker_bounds(config: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    """Return ``(min_workers, max_workers)``, treating a fixed size as min == max."""
    autoscale = config.get("autoscale") or {}
    num_workers = to_int(config.get("num_workers"))
    min_workers = to_int(autoscale.get("min_workers"))
    max_workers = to_int(autoscale.get("max_workers"))
    return (
        min_workers if min_workers is not None else num_workers,
        max_workers if max_workers is not None else num_workers,
    )


def monthly_hours_for_cluster(cluster_id: Optional[str], workload_kind: str) -> Tuple[float, float, str]:
    """Return ``(driver_hours, worker_node_hours, source)`` for one month.

    Measured node-hours win over assumptions. Worker hours are *node*-hours, so they already account
    for the node count; assumed hours are per node and are scaled by the caller.
    """
    telemetry = NODE_HOURS_INDEX.get(str(cluster_id)) if cluster_id else None
    if telemetry:
        total_hours = to_float(telemetry.get("monthly_total_node_hours")) or 0.0
        driver_hours = to_float(telemetry.get("monthly_driver_node_hours")) or 0.0
        worker_hours = to_float(telemetry.get("monthly_worker_node_hours")) or 0.0

        if driver_hours > 0:
            return driver_hours, worker_hours, "measured_node_hours"

        if total_hours > 0:
            # Older runtimes leave the driver flag null, which lands every node in the worker bucket.
            # A cluster always has exactly one driver, so its hours are the cluster's wall-clock
            # active hours; the remainder is worker time.
            driver_hours = min(to_float(telemetry.get("monthly_active_hours")) or 0.0, total_hours)
            return driver_hours, max(total_hours - driver_hours, 0.0), "measured_node_hours_driver_estimated"

    assumed = {
        "interactive": CONFIG["assumed_monthly_hours_interactive"],
        "job": CONFIG["assumed_monthly_hours_job"],
        "warehouse": CONFIG["assumed_monthly_hours_warehouse"],
    }.get(workload_kind, CONFIG["assumed_monthly_hours_interactive"])
    return assumed, assumed, f"assumed_{workload_kind}"


def build_sizing_row(
    config: Dict[str, Any],
    inventory_source: str,
    workload_kind: str,
    item_id: Any,
    item_name: Any,
    cluster_id_for_telemetry: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Produce one fully sized and costed migration row from a cluster definition."""
    min_workers, max_workers = worker_bounds(config)
    single_node = is_single_node_cluster(config)
    aws_attributes = config.get("aws_attributes") or {}

    sizing = size_cluster(
        worker_node_type=config.get("node_type_id"),
        driver_node_type=config.get("driver_node_type_id"),
        min_workers=min_workers,
        max_workers=max_workers,
        single_node=single_node,
        forced_sku=CONFIG["azure_vm_sku_override"],
    )

    driver_hours, worker_node_hours, hours_source = monthly_hours_for_cluster(cluster_id_for_telemetry, workload_kind)
    worker_count_for_cost = sizing["max_workers"]
    if hours_source.startswith("assumed"):
        # Assumed hours are per node, so scale them by the worker count to get worker node-hours.
        worker_node_hours = worker_node_hours * worker_count_for_cost

    driver_rate = price_book.hourly(sizing["azure_driver_vm_sku"])
    worker_rate = price_book.hourly(sizing["azure_worker_vm_sku"])

    monthly_cost = None
    if driver_rate is not None and (worker_count_for_cost == 0 or worker_rate is not None):
        monthly_cost = driver_rate * driver_hours + (worker_rate or 0.0) * worker_node_hours

    peak_hourly = None
    if driver_rate is not None and (worker_count_for_cost == 0 or worker_rate is not None):
        peak_hourly = driver_rate + (worker_rate or 0.0) * worker_count_for_cost

    utilization = NODE_HOURS_INDEX.get(str(cluster_id_for_telemetry)) if cluster_id_for_telemetry else None
    avg_cpu = to_float((utilization or {}).get("avg_cpu_utilization_percent"))
    rightsizing_hint = "no_utilization_data"
    if avg_cpu is not None:
        if avg_cpu < 15:
            rightsizing_hint = "review_downsize_cpu_under_15pct"
        elif avg_cpu > 80:
            rightsizing_hint = "review_upsize_cpu_over_80pct"
        else:
            rightsizing_hint = "utilization_healthy"

    row = {
        "inventory_source": inventory_source,
        "workload_kind": workload_kind,
        "source_cloud": CONFIG["source_cloud"],
        "target_cloud": CONFIG["target_cloud"],
        "item_id": item_id,
        "item_name": item_name,
        "cluster_id": cluster_id_for_telemetry,
        **sizing,
        "spark_version": config.get("spark_version"),
        "runtime_engine": config.get("runtime_engine"),
        "policy_id": config.get("policy_id"),
        "instance_pool_id": config.get("instance_pool_id"),
        "driver_instance_pool_id": config.get("driver_instance_pool_id"),
        "autotermination_minutes": config.get("autotermination_minutes"),
        "aws_availability": aws_attributes.get("availability"),
        "aws_zone_id": aws_attributes.get("zone_id"),
        "azure_region": CONFIG["azure_region"],
        "azure_currency": CONFIG["azure_currency"],
        "azure_pricing_model": CONFIG["azure_pricing_model"],
        "azure_driver_vm_hourly": round_or_none(driver_rate, 6),
        "azure_worker_vm_hourly": round_or_none(worker_rate, 6),
        "azure_cluster_vm_hourly_at_max": round_or_none(peak_hourly, 4),
        "monthly_driver_node_hours": round_or_none(driver_hours, 1),
        "monthly_worker_node_hours": round_or_none(worker_node_hours, 1),
        "hours_source": hours_source,
        "azure_vm_monthly_cost": round_or_none(monthly_cost, 2),
        "avg_cpu_utilization_percent": avg_cpu,
        "avg_memory_utilization_percent": to_float((utilization or {}).get("avg_memory_utilization_percent")),
        "rightsizing_hint": rightsizing_hint,
        "azure_price_status": "priced" if monthly_cost is not None else "no_price_available",
    }
    if extra:
        row.update(extra)
    return row


# Collect every Azure VM SKU the workspace will need, then price them all in one batched pass.
def collect_required_skus(configs: Iterable[Dict[str, Any]]) -> List[str]:
    """Pre-compute the distinct Azure VM SKUs needed, so pricing is fetched in as few calls as possible."""
    skus: set = set()
    for config in configs:
        sizing = size_cluster(
            config.get("node_type_id"), config.get("driver_node_type_id"),
            *worker_bounds(config), is_single_node_cluster(config), CONFIG["azure_vm_sku_override"],
        )
        skus.update(sku for sku in (sizing["azure_worker_vm_sku"], sizing["azure_driver_vm_sku"]) if sku)
    return sorted(skus)


job_cluster_configs = [
    {
        "node_type_id": row.get("node_type_id"),
        "driver_node_type_id": row.get("driver_node_type_id"),
        "num_workers": row.get("num_workers"),
        "autoscale": {"min_workers": row.get("autoscale_min_workers"), "max_workers": row.get("autoscale_max_workers")},
        "spark_version": row.get("spark_version"),
        "runtime_engine": row.get("runtime_engine"),
        "policy_id": row.get("policy_id"),
        "instance_pool_id": row.get("instance_pool_id"),
        "driver_instance_pool_id": row.get("driver_instance_pool_id"),
        "_job_id": row.get("job_id"),
        "_job_name": row.get("job_name"),
        "_task_key": row.get("task_key"),
        "_cluster_key": row.get("cluster_key"),
        "_cluster_scope": row.get("cluster_scope", "job_cluster"),
    }
    for row in job_cluster_rows
]

required_skus = collect_required_skus(list(clusters) + job_cluster_configs)
if CONFIG["azure_sql_warehouse_node_vm"]:
    required_skus = sorted(set(required_skus) | {CONFIG["azure_sql_warehouse_node_vm"]})

if required_skus:
    print(f"Pricing {len(required_skus)} distinct Azure VM SKUs in {CONFIG['azure_region']}...")
    price_book.fetch(required_skus, CONFIG["azure_region"])

interactive_sizing_rows = [
    build_sizing_row(
        cluster,
        inventory_source="interactive_or_recent_cluster",
        workload_kind="interactive",
        item_id=cluster.get("cluster_id"),
        item_name=cluster.get("cluster_name"),
        cluster_id_for_telemetry=cluster.get("cluster_id"),
        extra={
            "cluster_state": cluster.get("state"),
            "cluster_source": cluster.get("cluster_source"),
            "creator_user_name": cluster.get("creator_user_name"),
        },
    )
    for cluster in clusters
]

job_cluster_sizing_rows = [
    build_sizing_row(
        config,
        inventory_source=config["_cluster_scope"],
        workload_kind="job",
        item_id=config["_job_id"],
        item_name=config["_job_name"],
        cluster_id_for_telemetry=None,
        extra={"task_key": config["_task_key"], "cluster_key": config["_cluster_key"]},
    )
    for config in job_cluster_configs
]

cluster_sizing_pdf = pd.DataFrame(interactive_sizing_rows)
job_cluster_sizing_pdf = pd.DataFrame(job_cluster_sizing_rows)
combined_compute_sizing_pdf = (
    pd.concat([cluster_sizing_pdf, job_cluster_sizing_pdf], ignore_index=True)
    if interactive_sizing_rows or job_cluster_sizing_rows
    else pd.DataFrame()
)

HEADLINE_SIZING_COLS = [
    "inventory_source", "item_name", "aws_worker_node_type", "aws_worker_vcpu", "aws_worker_memory_gb",
    "azure_worker_vm_sku", "azure_vcpu_per_worker", "azure_memory_gb_per_worker", "max_workers",
    "max_nodes_including_driver", "azure_total_vcpu_at_max", "azure_total_memory_gb_at_max",
    "azure_cluster_vm_hourly_at_max", "azure_vm_monthly_cost", "hours_source", "mapping_confidence",
    "rightsizing_hint", "azure_price_status",
]

print(f"Sized {len(interactive_sizing_rows)} interactive clusters and {len(job_cluster_sizing_rows)} job clusters.")
print("Headline view (full detail is in the saved outputs):")
display_pdf(select_existing(combined_compute_sizing_pdf, HEADLINE_SIZING_COLS), "No compute sizing rows produced")

# COMMAND ----------

# MAGIC %md
# MAGIC ### SQL warehouse sizing
# MAGIC
# MAGIC Databricks SQL warehouse t-shirt sizes map to a fixed number of cluster nodes. **Serverless** warehouses have
# MAGIC no customer-visible VM cost on Azure, so they are reported with a DBU-only note. Classic and Pro warehouses
# MAGIC run on VMs in your subscription and are costed here.

# COMMAND ----------

# Databricks SQL warehouse t-shirt size -> nodes per cluster (driver + workers).
SQL_WAREHOUSE_NODE_COUNTS: Dict[str, int] = {
    "2X-Small": 1, "X-Small": 2, "Small": 4, "Medium": 8, "Large": 16,
    "X-Large": 32, "2X-Large": 64, "3X-Large": 128, "4X-Large": 256,
}


def warehouse_node_count(cluster_size: Optional[str]) -> Optional[int]:
    """Nodes per warehouse cluster for a Databricks SQL t-shirt size."""
    if not cluster_size:
        return None
    normalized = str(cluster_size).strip().lower().replace(" ", "").replace("-", "")
    for name, nodes in SQL_WAREHOUSE_NODE_COUNTS.items():
        if name.lower().replace(" ", "").replace("-", "") == normalized:
            return nodes
    return None


warehouse_sizing_rows: List[Dict[str, Any]] = []

for warehouse in warehouses:
    warehouse_type = str(warehouse.get("warehouse_type") or "").upper()
    is_serverless = warehouse_type == "SERVERLESS" or bool(warehouse.get("enable_serverless_compute"))
    nodes_per_cluster = warehouse_node_count(warehouse.get("cluster_size"))
    max_clusters = to_int(warehouse.get("max_num_clusters")) or 1
    total_nodes = (nodes_per_cluster or 0) * max_clusters
    vm_sku = None if is_serverless else CONFIG["azure_sql_warehouse_node_vm"]
    node_rate = price_book.hourly(vm_sku) if vm_sku else None
    monthly_hours = CONFIG["assumed_monthly_hours_warehouse"]
    vm_spec = AZURE_VM_INDEX.get(vm_sku or "", {})

    warehouse_sizing_rows.append(
        {
            "inventory_source": "sql_warehouse",
            "workload_kind": "warehouse",
            "source_cloud": CONFIG["source_cloud"],
            "target_cloud": CONFIG["target_cloud"],
            "warehouse_id": warehouse.get("id"),
            "warehouse_name": warehouse.get("name"),
            "warehouse_type": warehouse.get("warehouse_type"),
            "is_serverless": is_serverless,
            "cluster_size": warehouse.get("cluster_size"),
            "nodes_per_cluster": nodes_per_cluster,
            "min_num_clusters": warehouse.get("min_num_clusters"),
            "max_num_clusters": max_clusters,
            "max_total_nodes": total_nodes or None,
            "auto_stop_mins": warehouse.get("auto_stop_mins"),
            "enable_photon": warehouse.get("enable_photon"),
            "spot_instance_policy": warehouse.get("spot_instance_policy"),
            "state": warehouse.get("state"),
            "azure_region": CONFIG["azure_region"],
            "azure_warehouse_size_equivalent": warehouse.get("cluster_size"),
            "azure_node_vm_sku": vm_sku,
            "azure_vcpu_per_node": vm_spec.get("vcpu"),
            "azure_memory_gb_per_node": vm_spec.get("memory_gb"),
            "azure_total_vcpu_at_max": (vm_spec.get("vcpu") or 0) * total_nodes or None,
            "azure_node_vm_hourly": round_or_none(node_rate, 6),
            "azure_warehouse_vm_hourly_at_max": round_or_none(node_rate * total_nodes, 4) if node_rate and total_nodes else None,
            "billable_hours_per_month": monthly_hours,
            "azure_vm_monthly_cost": round_or_none(node_rate * total_nodes * monthly_hours, 2) if node_rate and total_nodes else None,
            "hours_source": "assumed_warehouse",
            "azure_currency": CONFIG["azure_currency"],
            "azure_pricing_model": CONFIG["azure_pricing_model"],
            "azure_price_status": "not_applicable_serverless" if is_serverless else ("priced" if node_rate else "no_price_available"),
            "migration_note": (
                "Serverless SQL warehouses are billed as DBUs only, with no customer-visible VM cost. "
                "Compare the serverless DBU rate instead."
                if is_serverless
                else f"Modelled as {total_nodes or 'n/a'} x {vm_sku}. Validate the warehouse size against query concurrency "
                     "and latency targets in Azure; Photon and warehouse type also change the DBU rate."
            ),
        }
    )

warehouse_sizing_pdf = pd.DataFrame(warehouse_sizing_rows)
print(f"Sized {len(warehouse_sizing_rows)} SQL warehouses.")
display_pdf(warehouse_sizing_pdf, "No SQL warehouse sizing rows produced")

azure_vm_prices_pdf = price_book.to_frame()
pricing_warnings_pdf = price_book.warnings_frame()

print(f"Azure VM price records retrieved: {len(azure_vm_prices_pdf)}")
display_pdf(azure_vm_prices_pdf, "No Azure VM prices retrieved")
display_pdf(pricing_warnings_pdf, "No pricing warnings. Every requested SKU returned a price.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 16. Databricks usage and DBU pricing
# MAGIC
# MAGIC Azure VM cost is only half the bill. This section reads `system.billing.usage` for what you consume today on
# MAGIC AWS, then reprices the same SKUs against Azure Databricks list prices from `system.billing.list_prices`.

# COMMAND ----------

usage_cols = set(table_columns("system.billing.usage"))

# system.billing.usage columns vary by runtime and account, so every optional field is guarded.
usage_metadata_expr = "to_json(usage_metadata)" if "usage_metadata" in usage_cols else "CAST(NULL AS STRING)"
u_usage_metadata_expr = "to_json(u.usage_metadata)" if "usage_metadata" in usage_cols else "CAST(NULL AS STRING)"
product_features_expr = "to_json(product_features)" if "product_features" in usage_cols else "CAST(NULL AS STRING)"
identity_metadata_expr = "to_json(identity_metadata)" if "identity_metadata" in usage_cols else "CAST(NULL AS STRING)"
billing_origin_product_expr = "billing_origin_product" if "billing_origin_product" in usage_cols else "CAST(NULL AS STRING)"
u_billing_origin_product_expr = "u.billing_origin_product" if "billing_origin_product" in usage_cols else "CAST(NULL AS STRING)"
usage_type_expr = "usage_type" if "usage_type" in usage_cols else "CAST(NULL AS STRING)"
u_usage_type_expr = "u.usage_type" if "usage_type" in usage_cols else "CAST(NULL AS STRING)"

source_cloud_sql = sql_quote(CONFIG["source_cloud"])
target_cloud_sql = sql_quote(CONFIG["target_cloud"])
start_date_sql = sql_quote(start_date)


def usage_summary_query(grouping_expr: str, grouping_alias: str, require_not_null: bool = True) -> str:
    """Build a usage summary grouped by one metadata field, such as cluster_id or job_id."""
    not_null_filter = f"      AND {grouping_expr} IS NOT NULL" if require_not_null else ""
    return f"""
    SELECT
      {grouping_expr} AS {grouping_alias},
      sku_name,
      cloud,
      usage_unit,
      {usage_type_expr} AS usage_type,
      {billing_origin_product_expr} AS billing_origin_product,
      SUM(usage_quantity) AS total_usage_quantity,
      MIN(usage_start_time) AS first_seen,
      MAX(usage_end_time) AS last_seen
    FROM system.billing.usage
    WHERE usage_start_time >= DATE({start_date_sql})
{not_null_filter}
    GROUP BY {grouping_expr}, sku_name, cloud, usage_unit, {usage_type_expr}, {billing_origin_product_expr}
    ORDER BY total_usage_quantity DESC
    """


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
    WHERE usage_start_time >= DATE({start_date_sql})
    GROUP BY
      date_trunc('month', usage_start_time), workspace_id, sku_name, cloud, usage_unit,
      {usage_type_expr}, {billing_origin_product_expr},
      get_json_object({usage_metadata_expr}, '$.cluster_id'),
      get_json_object({usage_metadata_expr}, '$.job_id'),
      get_json_object({usage_metadata_expr}, '$.warehouse_id'),
      get_json_object({usage_metadata_expr}, '$.node_type'),
      get_json_object({usage_metadata_expr}, '$.instance_pool_id'),
      {product_features_expr}, {identity_metadata_expr}
    ORDER BY usage_month DESC, total_usage_quantity DESC
    """,
)
display_spark_df(usage_detail, "No billing usage detail available")

monthly_summary = safe_sql(
    "monthly_usage_summary",
    f"""
    SELECT
      date_trunc('month', usage_start_time) AS usage_month,
      sku_name, cloud, usage_unit,
      {usage_type_expr} AS usage_type,
      {billing_origin_product_expr} AS billing_origin_product,
      SUM(usage_quantity) AS total_usage_quantity
    FROM system.billing.usage
    WHERE usage_start_time >= DATE({start_date_sql})
    GROUP BY date_trunc('month', usage_start_time), sku_name, cloud, usage_unit,
             {usage_type_expr}, {billing_origin_product_expr}
    ORDER BY usage_month DESC, total_usage_quantity DESC
    """,
)
display_spark_df(monthly_summary, "No monthly usage summary available")

cluster_usage = safe_sql("cluster_usage_summary", usage_summary_query(f"get_json_object({usage_metadata_expr}, '$.cluster_id')", "cluster_id"))
display_spark_df(cluster_usage, "No cluster usage summary available")

job_usage = safe_sql("job_usage_summary", usage_summary_query(f"get_json_object({usage_metadata_expr}, '$.job_id')", "job_id"))
display_spark_df(job_usage, "No job usage summary available")

warehouse_usage = safe_sql("warehouse_usage_summary", usage_summary_query(f"get_json_object({usage_metadata_expr}, '$.warehouse_id')", "warehouse_id"))
display_spark_df(warehouse_usage, "No SQL warehouse usage summary available")

# COMMAND ----------

# MAGIC %md
# MAGIC ### AWS to Azure Databricks list-price comparison
# MAGIC
# MAGIC `system.billing.list_prices` carries the list price for **every** cloud, so the Azure DBU rate can be read
# MAGIC directly rather than copied from a web page. Rows where `unpriced_usage_record_count > 0` had no matching
# MAGIC Azure SKU and need a manual look, usually because a SKU name differs between clouds.

# COMMAND ----------

price_amount_expr = (
    "try_cast(coalesce(get_json_object(to_json(pricing), '$.effective_list.default'), "
    "get_json_object(to_json(pricing), '$.default')) AS DOUBLE)"
)

price_catalog_history = safe_sql(
    "price_catalog_history_source_and_target",
    f"""
    SELECT sku_name, cloud, currency_code, usage_unit, price_start_time, price_end_time,
           {price_amount_expr} AS list_unit_price, to_json(pricing) AS pricing_json
    FROM system.billing.list_prices
    WHERE cloud IN ({source_cloud_sql}, {target_cloud_sql})
    ORDER BY cloud, sku_name, usage_unit, price_start_time DESC
    """,
)
display_spark_df(price_catalog_history, "No pricing history available")

current_price_catalog = safe_sql(
    "current_price_catalog_source_and_target",
    f"""
    SELECT sku_name, cloud, currency_code, usage_unit, price_start_time, price_end_time,
           {price_amount_expr} AS list_unit_price, to_json(pricing) AS pricing_json
    FROM system.billing.list_prices
    WHERE cloud IN ({source_cloud_sql}, {target_cloud_sql}) AND price_end_time IS NULL
    ORDER BY cloud, sku_name, usage_unit
    """,
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
        try_cast(coalesce(get_json_object(to_json(source_price.pricing), '$.effective_list.default'),
                          get_json_object(to_json(source_price.pricing), '$.default')) AS DOUBLE) AS source_list_unit_price,
        source_price.currency_code AS source_currency_code,
        try_cast(coalesce(get_json_object(to_json(target_price.pricing), '$.effective_list.default'),
                          get_json_object(to_json(target_price.pricing), '$.default')) AS DOUBLE) AS target_list_unit_price,
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
      WHERE u.usage_start_time >= DATE({start_date_sql})
        AND u.cloud = {source_cloud_sql}
    )
    SELECT
      usage_month, workspace_id, sku_name, source_cloud,
      {target_cloud_sql} AS target_cloud,
      usage_unit, usage_type, billing_origin_product, cluster_id, job_id, warehouse_id,
      source_currency_code, target_currency_code,
      SUM(usage_quantity) AS total_usage_quantity,
      SUM(usage_quantity * source_list_unit_price) AS source_databricks_list_cost,
      SUM(usage_quantity * target_list_unit_price) AS target_databricks_list_cost,
      SUM(usage_quantity * target_list_unit_price) - SUM(usage_quantity * source_list_unit_price) AS estimated_databricks_list_cost_delta,
      SUM(CASE WHEN target_list_unit_price IS NULL THEN 1 ELSE 0 END) AS unpriced_usage_record_count
    FROM priced_usage
    GROUP BY usage_month, workspace_id, sku_name, source_cloud, usage_unit, usage_type,
             billing_origin_product, cluster_id, job_id, warehouse_id, source_currency_code, target_currency_code
    ORDER BY usage_month DESC, target_databricks_list_cost DESC
    """,
)
display_spark_df(aws_to_azure_dbu_estimate, "No AWS-to-Azure Databricks list-price estimate available")

sql_errors_pdf = pd.DataFrame(sql_errors)
display_pdf(sql_errors_pdf, "No system table errors recorded")

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 4 &mdash; Results

# COMMAND ----------

# MAGIC %md
# MAGIC ## 17. Executive summary
# MAGIC
# MAGIC One table for the business case: what the Azure target looks like and what it costs per month.

# COMMAND ----------

def monthly_dbu_totals(estimate_df) -> Dict[str, Optional[float]]:
    """Average monthly AWS and Azure Databricks list cost over the observed months."""
    pdf = spark_to_pdf(estimate_df, limit=200000)
    if pdf.empty:
        return {"aws_dbu_monthly": None, "azure_dbu_monthly": None, "currency": None, "months_observed": 0, "unpriced_rows": None}

    for column in ["source_databricks_list_cost", "target_databricks_list_cost", "unpriced_usage_record_count"]:
        if column in pdf.columns:
            pdf[column] = pd.to_numeric(pdf[column], errors="coerce")

    months = pdf["usage_month"].nunique() if "usage_month" in pdf.columns else 1
    months = max(int(months or 1), 1)
    currency = None
    if "target_currency_code" in pdf.columns and pdf["target_currency_code"].notna().any():
        currency = pdf["target_currency_code"].dropna().iloc[0]

    return {
        "aws_dbu_monthly": round(pdf.get("source_databricks_list_cost", pd.Series(dtype=float)).sum() / months, 2),
        "azure_dbu_monthly": round(pdf.get("target_databricks_list_cost", pd.Series(dtype=float)).sum() / months, 2),
        "currency": currency,
        "months_observed": months,
        "unpriced_rows": int(pdf.get("unpriced_usage_record_count", pd.Series(dtype=float)).fillna(0).gt(0).sum()),
    }


dbu_totals = monthly_dbu_totals(aws_to_azure_dbu_estimate)


def sum_column(pdf: pd.DataFrame, column: str) -> Optional[float]:
    """Sum a numeric column, returning ``None`` when the frame or column is empty."""
    if pdf is None or pdf.empty or column not in pdf.columns:
        return None
    total = pd.to_numeric(pdf[column], errors="coerce").sum()
    return None if pd.isna(total) else round(float(total), 2)


compute_vm_monthly = sum_column(combined_compute_sizing_pdf, "azure_vm_monthly_cost")
warehouse_vm_monthly = sum_column(warehouse_sizing_pdf, "azure_vm_monthly_cost")
total_vm_monthly = sum(value for value in [compute_vm_monthly, warehouse_vm_monthly] if value is not None) or None
azure_dbu_monthly = dbu_totals["azure_dbu_monthly"]
total_azure_monthly = sum(value for value in [total_vm_monthly, azure_dbu_monthly] if value is not None) or None

measured_rows = int(combined_compute_sizing_pdf.get("hours_source", pd.Series(dtype=str)).astype(str).str.startswith("measured").sum()) if not combined_compute_sizing_pdf.empty else 0
priced_rows = int((combined_compute_sizing_pdf.get("azure_price_status", pd.Series(dtype=str)) == "priced").sum()) if not combined_compute_sizing_pdf.empty else 0
low_confidence_rows = int(combined_compute_sizing_pdf.get("mapping_confidence", pd.Series(dtype=str)).isin(["low", "review", "none"]).sum()) if not combined_compute_sizing_pdf.empty else 0

top_vm_skus = ""
if not combined_compute_sizing_pdf.empty and "azure_worker_vm_sku" in combined_compute_sizing_pdf.columns:
    counts = combined_compute_sizing_pdf["azure_worker_vm_sku"].dropna().value_counts().head(5)
    top_vm_skus = ", ".join(f"{sku} x{count}" for sku, count in counts.items())

currency = CONFIG["azure_currency"]

executive_summary_pdf = pd.DataFrame(
    [
        {"category": "Scope", "metric": "Source workspace", "value": workspace_url or "not connected"},
        {"category": "Scope", "metric": "Observation window", "value": f"{CONFIG['usage_lookback_days']} days from {start_date}"},
        {"category": "Scope", "metric": "Interactive clusters sized", "value": len(cluster_sizing_pdf)},
        {"category": "Scope", "metric": "Job clusters sized", "value": len(job_cluster_sizing_pdf)},
        {"category": "Scope", "metric": "SQL warehouses sized", "value": len(warehouse_sizing_pdf)},
        {"category": "Target", "metric": "Azure region", "value": f"{CONFIG['azure_region']} ({CONFIG['azure_region_label']})"},
        {"category": "Target", "metric": "Most recommended Azure VM SKUs", "value": top_vm_skus or NOT_AVAILABLE},
        {"category": "Target", "metric": "Total Azure vCPU at max scale", "value": sum_column(combined_compute_sizing_pdf, "azure_total_vcpu_at_max") or NOT_AVAILABLE},
        {"category": "Target", "metric": "Total Azure memory at max scale (GB)", "value": sum_column(combined_compute_sizing_pdf, "azure_total_memory_gb_at_max") or NOT_AVAILABLE},
        {"category": "Cost", "metric": "Azure VM cost, clusters (monthly)", "value": money(compute_vm_monthly, currency)},
        {"category": "Cost", "metric": "Azure VM cost, SQL warehouses (monthly)", "value": money(warehouse_vm_monthly, currency)},
        {"category": "Cost", "metric": "Azure VM cost, total (monthly)", "value": money(total_vm_monthly, currency)},
        {"category": "Cost", "metric": "Databricks DBU list cost on AWS today (monthly)", "value": money(dbu_totals["aws_dbu_monthly"], dbu_totals["currency"] or currency)},
        {"category": "Cost", "metric": "Databricks DBU list cost on Azure (monthly)", "value": money(azure_dbu_monthly, dbu_totals["currency"] or currency)},
        {"category": "Cost", "metric": "Estimated Azure total (VM + DBU, monthly)", "value": money(total_azure_monthly, currency)},
        {"category": "Cost", "metric": "Pricing model", "value": PRICING_MODEL_LABELS[CONFIG["azure_pricing_model"]]},
        {"category": "Confidence", "metric": "Rows using measured node-hours", "value": f"{measured_rows} of {len(combined_compute_sizing_pdf)}"},
        {"category": "Confidence", "metric": "Rows with a live Azure VM price", "value": f"{priced_rows} of {len(combined_compute_sizing_pdf)}"},
        {"category": "Confidence", "metric": "Rows needing a mapping review", "value": low_confidence_rows},
        {"category": "Confidence", "metric": "Region list source", "value": CONFIG["azure_region_source"]},
        {"category": "Confidence", "metric": "Collection errors (API / SQL / pricing)", "value": f"{len(api_errors)} / {len(sql_errors)} / {len(price_book.warnings)}"},
    ]
)

print("=" * 78)
print(f"EXECUTIVE SUMMARY - migration to Azure Databricks in {CONFIG['azure_region']} ({CONFIG['azure_region_label']})")
print("=" * 78)
print(f"  Azure VM cost per month        : {money(total_vm_monthly, currency)}")
print(f"  Azure Databricks DBU per month : {money(azure_dbu_monthly, dbu_totals['currency'] or currency)}")
print(f"  Estimated Azure total          : {money(total_azure_monthly, currency)}")
print(f"  Compared with AWS DBU today    : {money(dbu_totals['aws_dbu_monthly'], dbu_totals['currency'] or currency)} (Databricks list price only)")
print(f"  Pricing model                  : {PRICING_MODEL_LABELS[CONFIG['azure_pricing_model']]}")
print("  These are public list prices. They exclude your discounts, storage, networking and support.")
print("=" * 78)

display_pdf(executive_summary_pdf, "No executive summary could be produced")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 18. Save the outputs
# MAGIC
# MAGIC Everything is written twice: Delta tables under `OUTPUT_BASE_PATH` and CSV files under `LOCAL_OUTPUT_DIR`,
# MAGIC so the results can be opened in Excel or joined to a wider estimate.

# COMMAND ----------

output_base_path = CONFIG["output_base_path"]
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
    "cluster_node_hours": node_hours_pdf,
    "historical_cluster_definitions": historical_clusters_pdf,
}

sizing_outputs = {
    "executive_summary": executive_summary_pdf,
    "run_settings": run_settings_pdf,
    "azure_compute_sizing_from_api": combined_compute_sizing_pdf,
    "azure_interactive_cluster_sizing_from_api": cluster_sizing_pdf,
    "azure_job_cluster_sizing_from_api": job_cluster_sizing_pdf,
    "azure_sql_warehouse_sizing_from_api": warehouse_sizing_pdf,
    "azure_node_type_reference_from_api": node_types_summary_pdf,
    "azure_vm_catalog": azure_vm_catalog_pdf,
    "azure_vm_prices": azure_vm_prices_pdf,
    "azure_pricing_warnings": pricing_warnings_pdf,
    "azure_regions_allowed": azure_regions_pdf,
    "quick_estimate": quick_estimate_pdf,
    "quick_pricing_options": quick_pricing_options_pdf,
    "azure_region_comparison": region_comparison_pdf,
    "aws_to_azure_mapping_examples": mapping_examples_pdf,
}

print("Writing API inventory outputs")
write_pdf_outputs(api_outputs, api_output_path, api_local_dir)
print("Writing Azure sizing and pricing outputs")
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
    print("Writing billing outputs")
    for name, df in spark_outputs.items():
        if df is None:
            continue
        try:
            df.write.mode("overwrite").format("delta").save(f"{billing_output_path}/{name}")
            df.coalesce(1).write.mode("overwrite").option("header", "true").csv(f"{billing_output_path}_csv/{name}")
        except Exception as exc:
            print(f"[warn] Could not write billing output {name}: {str(exc)[:200]}")

    if not sql_errors_pdf.empty:
        sdf = spark_df_from_pdf(sql_errors_pdf, force_string=True)
        if sdf is not None:
            try:
                sdf.write.mode("overwrite").format("delta").save(f"{billing_output_path}/billing_sql_errors")
            except Exception as exc:
                print(f"[warn] Could not write billing_sql_errors: {str(exc)[:200]}")

    print(f"  Delta: {billing_output_path}")
    print(f"  CSV  : {billing_output_path}_csv")
else:
    print("Spark is not available, so billing Delta/CSV outputs were skipped.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 19. Customer notes and assumptions
# MAGIC
# MAGIC ### What the numbers are
# MAGIC
# MAGIC - **Azure VM cost** is the compute cost of running the recommended VMs in your chosen region, taken live from
# MAGIC   the public Azure Retail Prices API at `https://prices.azure.com`.
# MAGIC - **Databricks DBU cost** is the Databricks platform charge, read from `system.billing.list_prices`, which
# MAGIC   carries the current list price for every cloud.
# MAGIC - **Monthly figures** use **730 hours per month** by default (365 &times; 24 &divide; 12), or the value in the
# MAGIC   **A4 Hours per month** widget. Where real telemetry exists, monthly cost uses **measured node-hours**
# MAGIC   scaled from the observation window to an average 30.44-day month, which is more accurate than any
# MAGIC   assumption.
# MAGIC
# MAGIC ### How the Azure size is chosen
# MAGIC
# MAGIC 1. AWS vCPU and memory come from **this workspace's own node-types API** where possible, otherwise they are
# MAGIC    derived from the instance name. The `aws_spec_source` column tells you which.
# MAGIC 2. The VM family follows the workload's memory-per-vCPU ratio: &ge;&nbsp;7&nbsp;GB &rarr; memory optimized
# MAGIC    (E series), &le;&nbsp;2.5&nbsp;GB &rarr; compute optimized (F series), otherwise general purpose (D series).
# MAGIC    AWS storage-optimized nodes go to the L series so local NVMe is preserved. GPU nodes go to NC/ND VMs.
# MAGIC 3. Within that family the notebook picks the **smallest VM that meets or exceeds** the AWS vCPU and memory.
# MAGIC 4. Node counts carry over unchanged, so the comparison is genuinely like for like.
# MAGIC 5. `mapping_confidence` flags anything that needs a human look: `high` is a clean capacity match, `medium`
# MAGIC    fell back to another family, `review` relaxed the memory requirement, `low` found no single VM big enough.
# MAGIC
# MAGIC ### What is **not** included
# MAGIC
# MAGIC - Your EA / MCA / CSP discounts, Azure consumption commitments and any private pricing.
# MAGIC - Azure Hybrid Benefit and any existing reservations or savings plans you already own.
# MAGIC - Managed disks, ADLS Gen2 storage and transactions, bandwidth, NAT Gateway, Private Link, Azure Monitor,
# MAGIC   backup and support plans.
# MAGIC - AWS-side costs (EC2, EBS, S3, data transfer) &mdash; only the Databricks DBU side of AWS is shown.
# MAGIC - Migration effort: data transfer, code changes, Unity Catalog setup, testing and parallel running.
# MAGIC - Serverless SQL warehouses and serverless jobs, which have no customer-visible VM cost.
# MAGIC
# MAGIC ### Things worth checking before you commit
# MAGIC
# MAGIC - **Region availability.** Confirm every recommended VM SKU is offered in your region, and that Azure
# MAGIC   Databricks itself is available there: <https://learn.microsoft.com/azure/databricks/resources/supported-regions>
# MAGIC - **Quota.** New subscriptions have low vCPU quotas per VM family. Raise a quota request early.
# MAGIC - **Right-sizing.** Rows flagged `review_downsize_cpu_under_15pct` were under 15% average CPU. Migration is
# MAGIC   the natural moment to resize them, which is usually the single biggest saving available.
# MAGIC - **Commitments.** Compare the `savings_plan` and `reserved` columns against pay-as-you-go. A 3-year
# MAGIC   reservation on steady-state clusters is typically the largest lever after right-sizing.
# MAGIC - **Spot for workers.** Azure Spot suits fault-tolerant job-cluster workers, never drivers.
# MAGIC - **Photon.** Photon changes the DBU rate but usually cuts total runtime. Model it per workload.
# MAGIC
# MAGIC ### Reproducing this run
# MAGIC
# MAGIC The `run_settings` output records every setting used, and `azure_vm_prices` records every price with the
# MAGIC timestamp it was retrieved. Azure list prices change, so re-run before any final commercial decision.
# MAGIC
# MAGIC ### Official references
# MAGIC
# MAGIC - [Azure Retail Prices API](https://learn.microsoft.com/rest/api/cost-management/retail-prices/azure-retail-prices)
# MAGIC - [Azure Databricks pricing](https://azure.microsoft.com/pricing/details/databricks/)
# MAGIC - [Azure Virtual Machines pricing](https://azure.microsoft.com/pricing/details/virtual-machines/linux/)
# MAGIC - [Databricks billing system table reference](https://docs.databricks.com/aws/en/admin/system-tables/billing)
# MAGIC - [Databricks compute system table reference](https://docs.databricks.com/aws/en/admin/system-tables/compute)
# MAGIC - [Databricks REST API reference](https://docs.databricks.com/api/workspace/introduction)
# MAGIC
# MAGIC > **Disclaimer.** Every figure here is an estimate produced from public list prices and the inventory this
# MAGIC > notebook could read. It is not a quotation. Validate against the official Azure pricing pages and your
# MAGIC > Microsoft agreement before making any commercial commitment.

# COMMAND ----------

print("Run complete.")
print(f"  Notebook version : {NOTEBOOK_VERSION}")
print(f"  Run timestamp    : {CONFIG['run_timestamp_utc']} UTC")
print(f"  Azure region     : {CONFIG['azure_region']} ({CONFIG['azure_region_label']})")
print(f"  Outputs (Delta)  : {output_base_path}")
print(f"  Outputs (CSV)    : {Path(CONFIG['local_output_dir']).resolve()}")
print(f"  Issues           : {len(api_errors)} API, {len(sql_errors)} system table, {len(price_book.warnings)} pricing")
if api_errors or sql_errors or price_book.warnings:
    print("  Each issue table lists a recommended action. None of them stop the run.")
