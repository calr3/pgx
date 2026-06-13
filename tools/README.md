# tools/

Operational helpers for running pgx training on GCP. Not part of the `pgx`
package; standalone scripts driven by the `gcloud` CLI.

## find_cheapest_tpu_spot.py

Find the cheapest GCP zone that can **actually** provision a given Cloud TPU on
spot pricing.

The motivation: the Cloud Billing Catalog will happily quote you a low spot
price in a region that has **no such hardware**. When picking a zone for the
offline-batch AlphaZero job, we found the three cheapest "v5e spot" SKUs were
all mirages — asia-south1 (Mumbai) is v6e-only, me-west1 (Israel) and
northamerica-northeast2 (Toronto) list v5e spot prices but offer no v5e
accelerators. This script cross-references price against real per-zone
accelerator-type availability so the ranking only contains zones you can create
in.

### Usage

```bash
# Default: v5litepod-4 (v5e), spot, active gcloud project
tools/find_cheapest_tpu_spot.py

# Other slices / generations (generation is inferred from the type name)
tools/find_cheapest_tpu_spot.py -a v5litepod-8
tools/find_cheapest_tpu_spot.py -a v6e-4

# Bound the availability scan to cheap regions; machine-readable output
tools/find_cheapest_tpu_spot.py --max-price 0.30 --json
```

Output is a price-ascending table; the cheapest row that has at least one
available zone is marked `*` and a ready-to-run `gcloud ... create` command is
printed for it.

### How it works

- **Pricing**: queries the public Cloud Billing Catalog (Compute Engine service
  `6F81-5844-456A`) for SKUs whose description matches the generation (e.g.
  `TpuV5e`) with `usageType=Preemptible` (= Spot), taking the per-chip-hour rate
  per region.
- **Availability**: for each priced region, probes every zone with
  `gcloud compute tpus accelerator-types list` (in parallel) and keeps only the
  zones that actually offer the requested type.

### Caveats

- "Available" means the type is *offered* in the zone. Momentary spot capacity
  is only confirmed at **create** time — a create may still fail on a stockout
  with `code 8 "no capacity"` or the opaque `code 13 "an internal error"`. If
  so, try the next zone in the list.
- Requires `gcloud` authenticated with access to the project and the Billing
  Catalog API; uses `gcloud auth print-access-token` for the catalog call.
