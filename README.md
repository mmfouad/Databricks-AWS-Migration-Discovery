# Databricks AWS Migration Discovery

This repo contains a Databricks discovery notebook for assessing an AWS Databricks workspace before migrating workloads to Azure Databricks.

Files:

- `aws_databricks_migration_discovery.ipynb`: notebook for Databricks or VS Code.
- `aws_databricks_migration_discovery.py`: Databricks source export of the same notebook.
- `requirements.txt`: local/VS Code dependencies for REST API inventory.

## What The Notebook Collects

The notebook keeps API inventory separate from billing and pricing evidence.

API inventory:

- Clusters, SQL warehouses, instance pools, node types, jobs, job tasks, and job cluster definitions.
- Pagination for supported list APIs.
- Retries for transient REST failures such as HTTP 429 and 5xx.
- Clean API error tables for missing permissions or unavailable endpoints.

Billing and pricing:

- Usage summaries from `system.billing.usage`.
- Current and historical Databricks list prices from `system.billing.list_prices`.
- AWS source usage joined to AWS and Azure Databricks list prices where the SKU and usage unit match.
- A Databricks list-price estimate table for Azure migration modeling.

Azure sizing outputs:

- `azure_compute_sizing_from_api`
- `azure_interactive_cluster_sizing_from_api`
- `azure_job_cluster_sizing_from_api`
- `azure_sql_warehouse_sizing_from_api`
- `azure_node_type_reference_from_api`

These are mapping tables for an Azure sizing workbook. They intentionally leave Azure VM SKU, Azure region, and target pricing tier as assessment columns because those depend on the target Azure region, workload performance target, availability design, and commercial terms.

## Running In Databricks

Run `aws_databricks_migration_discovery.ipynb` inside the source AWS Databricks workspace.

The notebook can use the Databricks notebook context for `DATABRICKS_HOST` and API token. You can still override credentials with environment variables if needed.

Quick AWS Databricks run:

1. Open the source AWS Databricks workspace that you want to inventory.
2. Import `aws_databricks_migration_discovery.ipynb` into Workspace, or import `aws_databricks_migration_discovery.py` as a Databricks source notebook.
3. Attach the notebook to an all-purpose cluster that can query Unity Catalog system tables. For API-only discovery, any cluster with Python, `pandas`, and `requests` is enough.
4. If `pandas` or `requests` are unavailable on the cluster, run this in a first notebook cell and then restart Python when prompted:

   ```python
   %pip install pandas requests
   ```

5. Confirm the identity running the notebook can view clusters, SQL warehouses, instance pools, jobs, and node types.
6. For billing and pricing discovery, confirm the identity can query `system.billing.usage` and `system.billing.list_prices`.
7. Run all cells. Review the displayed inventory tables first, then the saved Delta and CSV outputs.

Optional overrides can be set as cluster environment variables or by editing the `CONFIG` cell before running:

- `USAGE_LOOKBACK_DAYS=90`
- `RUN_API_INVENTORY=true`
- `RUN_BILLING_USAGE=true`
- `OUTPUT_BASE_PATH=dbfs:/tmp/aws_databricks_migration_discovery`
- `SOURCE_CLOUD=AWS`
- `TARGET_CLOUD=AZURE`

Default Delta output paths:

- `dbfs:/tmp/aws_databricks_migration_discovery/api_inventory`
- `dbfs:/tmp/aws_databricks_migration_discovery/azure_migration_sizing`
- `dbfs:/tmp/aws_databricks_migration_discovery/billing_usage_and_pricing`

Billing CSV exports are also written under:

- `dbfs:/tmp/aws_databricks_migration_discovery/billing_usage_and_pricing_csv`

## Running In VS Code

For REST API inventory only:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

export DATABRICKS_HOST="https://<workspace-host>"
export DATABRICKS_TOKEN="<personal-access-token-or-oauth-token>"
export RUN_BILLING_USAGE=false

python aws_databricks_migration_discovery.py
```

If your shell still cannot find `pip` after activation, use the venv Python directly:

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python aws_databricks_migration_discovery.py
```

Local CSV outputs are written under:

- `./outputs/aws_databricks_migration_discovery/api_inventory`
- `./outputs/aws_databricks_migration_discovery/azure_migration_sizing`

For billing/pricing tables from VS Code, use a Databricks-backed notebook/kernel, the Databricks VS Code extension, or Databricks Connect with access to the source workspace. A local Python process without Spark access can collect REST API inventory but cannot query `system.billing.*`.

The notebook also supports `~/.databrickscfg`. Set `DATABRICKS_CONFIG_PROFILE` if you want a profile other than `DEFAULT`.

## Configuration

Environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `DATABRICKS_HOST` | unset | Workspace URL for REST API calls. |
| `DATABRICKS_WORKSPACE_URL` | unset | Alternate workspace URL variable. |
| `DATABRICKS_TOKEN` | unset | Token for REST API calls. |
| `DATABRICKS_CONFIG_PROFILE` | `DEFAULT` | Profile from `~/.databrickscfg`. |
| `SOURCE_CLOUD` | `AWS` | Source cloud for pricing/usage comparisons. |
| `TARGET_CLOUD` | `AZURE` | Target cloud for migration pricing comparison. |
| `USAGE_LOOKBACK_DAYS` | `90` | Billing usage lookback window. |
| `RUN_API_INVENTORY` | `true` | Enable REST API inventory. |
| `RUN_BILLING_USAGE` | `true` | Enable Spark SQL billing/pricing queries. |
| `OUTPUT_BASE_PATH` | `dbfs:/tmp/aws_databricks_migration_discovery` | Delta output base path. |
| `LOCAL_OUTPUT_DIR` | `./outputs/aws_databricks_migration_discovery` | Local CSV output base path. |
| `DATABRICKS_API_MAX_RETRIES` | `5` | REST retry attempts. |
| `DATABRICKS_API_TIMEOUT_SECONDS` | `30` | REST request timeout. |

## Required Permissions

For API inventory, the caller needs workspace permissions to view:

- Clusters and recently terminated clusters.
- SQL warehouses.
- Instance pools.
- Jobs and job task settings.
- Workspace node types.

For full discovery, use a workspace admin or an account/workspace identity that has broad read access. Non-admin users may only see objects they can view, and the notebook will record skipped endpoints in the `api_errors` output instead of failing the run.

For billing and pricing, the Spark identity needs Unity Catalog/system table access:

- `USE CATALOG system`
- `USE SCHEMA billing`
- `SELECT` on `system.billing.usage`
- `SELECT` on `system.billing.list_prices`

If these queries fail, ask a Databricks account admin to enable system tables if required and grant access. The notebook writes failures to `billing_sql_errors`.

## Pricing Scope

The notebook uses Databricks system tables for Databricks pricing:

- `system.billing.usage` provides usage quantities by SKU, cloud, workspace, and metadata.
- `system.billing.list_prices` provides current and historical Databricks list prices by cloud, SKU, currency, and usage unit.

The Azure estimate is a Databricks list-price comparison only. It does not include:

- Private pricing, committed-use discounts, enterprise agreements, or marketplace terms.
- AWS infrastructure costs such as EC2, EBS, S3, data transfer, NAT, or PrivateLink.
- Azure infrastructure costs such as VM compute, managed disks, ADLS Gen2, bandwidth, Private Link, monitoring, or backup.

Use the generated pricing outputs together with Azure VM/storage/network pricing and the Azure Databricks pricing page when building the final migration business case.

Official references:

- [Databricks billing system table reference](https://docs.databricks.com/aws/en/admin/system-tables/billing)
- [Databricks pricing system table reference](https://docs.databricks.com/aws/en/admin/system-tables/pricing)
- [Azure Databricks pricing](https://azure.microsoft.com/pricing/details/databricks/)
- [Databricks REST API reference](https://docs.databricks.com/api/workspace/introduction)
