# AWS → Azure Databricks Migration Sizing & Pricing

A single Databricks notebook that a customer can run **inside their own AWS Databricks workspace** to answer
two questions:

1. **What Azure Databricks cluster do I need?** — the equivalent Azure VM SKU, node count, vCPU and memory
   for every cluster, job cluster and SQL warehouse in the workspace.
2. **What will it cost?** — hourly and monthly Azure VM pricing pulled live from the **public Azure Retail
   Prices API**, plus the Databricks DBU cost from the billing system tables.

The default target region is **`uaenorth`** (UAE North). Any other region is one dropdown away.

## Run it in three steps

1. Import `aws_databricks_migration_discovery.ipynb` (or the `.py`, as a Databricks source notebook) into the
   AWS Databricks workspace you want to assess.
2. Attach it to any all-purpose cluster on DBR 12.2 LTS or later and press **Run all**.
3. A widget bar appears at the top. Pick your **Azure region** and press **Run all** once more.

No code editing, no Azure credentials, no Azure subscription and no secrets are required. Everything the
customer can change is a widget.

If the workspace has no outbound internet access, live Azure pricing is skipped, the notebook says so, and
sizing still runs against a built-in region list.

## What you get

| Section | Output |
| --- | --- |
| 3 | Every valid Azure region value, printed exactly as the notebook accepts it |
| 7–9 | Inventory of clusters, SQL warehouses, instance pools, jobs and job clusters |
| 11 | AWS node type → Azure VM SKU recommendation, with the reason for each mapping |
| 12–13 | Live Azure VM prices, and a quick estimate you can drive entirely from widgets |
| 14 | Real node-hours and CPU/memory utilisation from `system.compute.node_timeline` |
| 15 | Recommended Azure size **and monthly cost** for every discovered cluster and warehouse |
| 16 | Databricks DBU cost today on AWS, repriced against Azure list prices |
| 17 | Executive summary: total monthly Azure VM + DBU cost |
| 18 | Delta tables and CSV exports of everything above |
| 19 | Customer notes, assumptions and exclusions |

## Azure region selection

The **A1 Azure region** widget defaults to `uaenorth`.

Region values are the exact `armRegionName` strings the Azure Retail Prices API accepts — `uaenorth`, not
"UAE North"; `eastus`, not "East US". Section 3 discovers the list live from the pricing API and prints all
of it, so the values shown to the customer are the same values validation accepts. A verified 69-region
fallback list is used when the API is unreachable.

Validation is forgiving about spelling but strict about the result:

```
AZURE_REGION = "UAE North"      ->  accepted, normalised to uaenorth
AZURE_REGION = "middleeast-1"   ->  SettingsError: 'middleeast-1' is not a valid Azure region name.
                                    Did you mean: denmarkeast?
                                    All 69 allowed values: ...
```

A few of the accepted values, for reference:

```
uaenorth  uaecentral  eastus  eastus2  westus  westus2  westus3  centralus
westeurope  northeurope  uksouth  ukwest  francecentral  germanywestcentral
southeastasia  australiaeast  japaneast  centralindia  qatarcentral  israelcentral
```

## How the Azure size is chosen

1. AWS vCPU and memory come from the workspace's own `clusters/list-node-types` API where available,
   otherwise they are derived from the instance name. The `aws_spec_source` column records which.
2. The VM family follows the memory-per-vCPU ratio — ≥ 7 GB → **E** series (memory optimized),
   ≤ 2.5 GB → **F** series (compute optimized), otherwise **D** series (general purpose). AWS
   storage-optimized nodes map to the **L** series to keep local NVMe; GPU nodes map to **NC/ND**.
3. Within that family the notebook picks the smallest VM that meets or exceeds the AWS vCPU **and** memory.
4. Node counts carry over unchanged, so the comparison is like for like.
5. Every row carries a `mapping_confidence` (`high` / `medium` / `review` / `low`) and a plain-English
   `mapping_note` explaining the choice.

Set **B5 VM family preference** or **B6 Force a VM SKU** to override the automatic choice, or
**C1 Sizing strategy** to `cost_optimized` (allows up to 20% less memory) or `performance` (one size up).

## Pricing

Prices come from `https://prices.azure.com/api/retail/prices`, which is public and anonymous — no Azure
login, subscription or SDK is needed. Linux, on-demand meters only; Windows and DevTest meters are excluded.

Six pricing models are retrieved for every SKU and shown side by side:

| Model | Meaning |
| --- | --- |
| `payg` | Pay-as-you-go (default) |
| `spot` | Azure Spot, evictable — suitable for job-cluster workers only |
| `savings_plan_1y` / `savings_plan_3y` | Azure savings plan for compute |
| `reserved_1y` / `reserved_3y` | Reserved instances, amortised to an hourly rate |

Monthly cost uses **730 hours per month** by default (widget **A4**). Where `system.compute.node_timeline`
is readable, monthly cost is instead based on **measured node-hours** scaled to an average 30.44-day month —
the `hours_source` column tells you which basis was used per row.

Every price is an unnegotiated public list price. It excludes your EA/MCA/CSP discounts, Azure Hybrid
Benefit, storage, networking, and support. It is an estimate, not a quotation.

## Widgets

| Widget | Env var | Default |
| --- | --- | --- |
| A1 Azure region | `AZURE_REGION` | `uaenorth` |
| A2 Currency | `AZURE_CURRENCY` | `USD` |
| A3 VM pricing model | `AZURE_PRICING_MODEL` | `payg` |
| A4 Hours per month | `HOURS_PER_MONTH` | `730` |
| B1 Workload size | `WORKLOAD_SIZE` | `medium` |
| B2 Worker count | `QUICK_NUM_WORKERS` | `4` |
| B3 Custom vCPU per node | `QUICK_CUSTOM_VCPU_PER_NODE` | empty |
| B4 Custom GB per node | `QUICK_CUSTOM_MEMORY_GB_PER_NODE` | empty |
| B5 VM family preference | `AZURE_VM_FAMILY_PREFERENCE` | `auto` |
| B6 Force a VM SKU | `AZURE_VM_SKU_OVERRIDE` | empty |
| C1 Sizing strategy | `AZURE_SIZING_STRATEGY` | `like_for_like` |
| C2 Prefer local NVMe VMs | `AZURE_PREFER_LOCAL_SSD` | `true` |
| C3 Usage lookback days | `USAGE_LOOKBACK_DAYS` | `90` |
| C4 Compare extra regions | `AZURE_COMPARISON_REGIONS` | empty |
| D1 Collect REST inventory | `RUN_API_INVENTORY` | `true` |
| D2 Query system tables | `RUN_BILLING_USAGE` | `true` |
| D3 Fetch live Azure prices | `RUN_AZURE_PRICING` | `true` |

Precedence is **widget → environment variable → constant in the settings cell**, so the same notebook works
as a widget-driven demo in Databricks and as a scripted run in CI.

Additional environment-only settings: `DATABRICKS_HOST`, `DATABRICKS_TOKEN`, `DATABRICKS_CONFIG_PROFILE`,
`SOURCE_CLOUD` (`AWS`), `TARGET_CLOUD` (`AZURE`), `OUTPUT_BASE_PATH`, `LOCAL_OUTPUT_DIR`,
`DATABRICKS_API_MAX_RETRIES`, `DATABRICKS_API_TIMEOUT_SECONDS`, `ASSUMED_MONTHLY_HOURS_INTERACTIVE`,
`ASSUMED_MONTHLY_HOURS_JOB`, `ASSUMED_MONTHLY_HOURS_WAREHOUSE`.

## Permissions

The notebook never fails on a permission error. Anything it cannot read is recorded in the `api_errors` or
`billing_sql_errors` output with a recommended action, and the run continues.

**REST inventory** — permission to view clusters, SQL warehouses, instance pools, jobs and node types. A
workspace admin sees everything; a normal user sees what they have access to.

**System tables** — `USE CATALOG system`, plus `SELECT` on `system.billing.usage`,
`system.billing.list_prices`, `system.compute.clusters` and `system.compute.node_timeline`.

**Azure pricing** — outbound HTTPS to `prices.azure.com`. Nothing else. No Azure identity of any kind.

Only workspace metadata is read. No table data, query text or notebook content is touched, and the API token
is never printed or written to any output.

## Running outside Databricks

REST inventory and Azure pricing work from a plain Python process; system tables need Spark.

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt

export DATABRICKS_HOST="https://<workspace-host>"
export DATABRICKS_TOKEN="<personal-access-token>"
export RUN_BILLING_USAGE=false
export AZURE_REGION=uaenorth

python aws_databricks_migration_discovery.py
```

`~/.databrickscfg` is also supported; set `DATABRICKS_CONFIG_PROFILE` for a non-`DEFAULT` profile.

## Outputs

Delta tables under `OUTPUT_BASE_PATH` (default `dbfs:/tmp/aws_databricks_migration_discovery`) and CSVs under
`LOCAL_OUTPUT_DIR` (default `./outputs/aws_databricks_migration_discovery`):

- `api_inventory/` — clusters, warehouses, pools, node types, jobs, job clusters, job tasks, node-hours,
  historical cluster definitions, API errors.
- `azure_migration_sizing/` — `executive_summary`, `run_settings`, `azure_compute_sizing_from_api`,
  `azure_interactive_cluster_sizing_from_api`, `azure_job_cluster_sizing_from_api`,
  `azure_sql_warehouse_sizing_from_api`, `azure_vm_catalog`, `azure_vm_prices`, `azure_pricing_warnings`,
  `azure_regions_allowed`, `quick_estimate`, `quick_pricing_options`, `azure_region_comparison`,
  `aws_to_azure_mapping_examples`.
- `billing_usage_and_pricing/` — usage detail and summaries, price catalogs, and the AWS→Azure Databricks
  list-price estimate.

`run_settings` records every setting used and `azure_vm_prices` records the retrieval timestamp, so any run
can be reproduced and audited.

## Repo layout

| File | Purpose |
| --- | --- |
| `aws_databricks_migration_discovery.py` | **Source of truth.** Databricks source format. Edit this. |
| `aws_databricks_migration_discovery.ipynb` | Generated Jupyter notebook. |
| `aws_databricks_migration_discovery_preview.html` | Generated static preview, readable without Databricks. |
| `tools/build_notebook.py` | Regenerates the two files above from the `.py`. |
| `requirements.txt` | `pandas` and `requests` for local runs. |

After editing the `.py`:

```bash
python tools/build_notebook.py           # regenerate
python tools/build_notebook.py --check   # verify nothing is stale
```

## Disclaimer

Every figure produced by this notebook is an **estimate** built from public list prices and the inventory the
notebook was able to read. It is not a quotation and carries no commercial commitment. Validate against the
official Azure pricing pages and your Microsoft agreement before making any decision.

- [Azure Retail Prices API](https://learn.microsoft.com/rest/api/cost-management/retail-prices/azure-retail-prices)
- [Azure Databricks pricing](https://azure.microsoft.com/pricing/details/databricks/)
- [Azure Virtual Machines pricing](https://azure.microsoft.com/pricing/details/virtual-machines/linux/)
- [Azure Databricks supported regions](https://learn.microsoft.com/azure/databricks/resources/supported-regions)
- [Databricks billing system tables](https://docs.databricks.com/aws/en/admin/system-tables/billing)
- [Databricks compute system tables](https://docs.databricks.com/aws/en/admin/system-tables/compute)
