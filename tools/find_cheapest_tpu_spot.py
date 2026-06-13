#!/usr/bin/env python3
# Copyright 2026 The Pgx Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Find the cheapest GCP zone that can ACTUALLY provision a Cloud TPU on spot.

A low spot price in the Cloud Billing Catalog does NOT mean the hardware is
offered in that region. For example, asia-south1 (Mumbai) lists the cheapest
v5e spot SKU but only has v6e accelerators; me-west1 and northamerica-northeast2
likewise advertise low v5e spot prices with no v5e hardware at all. This tool
cross-references spot pricing against real per-zone accelerator-type
availability, so the ranking only contains zones you can actually create in.

Caveat: "available" means the accelerator type is *offered* in the zone.
Momentary spot capacity is only ever confirmed at create time — a create can
still fail on a stockout, either with an explicit
    code 8  "There is no more capacity in the zone ..."
or an opaque
    code 13 "an internal error has occurred"
(both observed for spot v5e). When that happens, try the next zone down the list.

Pricing comes from the public Cloud Billing Catalog (Compute Engine service,
id 6F81-5844-456A); availability and capacity come from the TPU API, queried
via the `gcloud` CLI using your active credentials.

Examples:
  tools/find_cheapest_tpu_spot.py                          # v5litepod-4 (v5e), spot
  tools/find_cheapest_tpu_spot.py -a v5litepod-8
  tools/find_cheapest_tpu_spot.py -a v6e-4                 # gen inferred as v6e
  tools/find_cheapest_tpu_spot.py --max-price 0.30 --json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# Compute Engine service id in the Cloud Billing Catalog (TPUs bill under it).
COMPUTE_ENGINE_SERVICE = "6F81-5844-456A"

# Per TPU generation: the substring used in spot SKU descriptions, and the
# tpu-vm runtime version to suggest in the create command.
GEN_INFO = {
    "v5e": {"sku": "TpuV5e", "runtime": "v2-alpha-tpuv5-lite"},
    "v5p": {"sku": "TpuV5p", "runtime": "v2-alpha-tpuv5"},
    "v6e": {"sku": "TpuV6e", "runtime": "v2-alpha-tpuv6e"},
    "v4":  {"sku": "TpuV4",  "runtime": "tpu-vm-v4-base"},
}

# Accelerator-type prefix -> generation key above.
PREFIX_TO_GEN = {
    "v5litepod": "v5e",
    "v5p": "v5p",
    "v6e": "v6e",
    "v4": "v4",
}


def infer_gen(accelerator_type: str) -> str:
    for prefix, gen in PREFIX_TO_GEN.items():
        if accelerator_type.startswith(prefix):
            return gen
    raise SystemExit(
        f"Cannot infer TPU generation from --accelerator-type={accelerator_type!r}; "
        f"pass --tpu-gen explicitly (one of {', '.join(GEN_INFO)})."
    )


def chip_count(accelerator_type: str) -> int | None:
    """Number of chips in the slice, e.g. v5litepod-4 -> 4. None if not numeric."""
    tail = accelerator_type.rsplit("-", 1)[-1]
    return int(tail) if tail.isdigit() else None


def gcloud(args: list[str]) -> str:
    """Run a gcloud command, returning stdout (empty string on failure)."""
    try:
        out = subprocess.run(
            ["gcloud", *args],
            capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return out.stdout if out.returncode == 0 else ""


def access_token() -> str:
    tok = gcloud(["auth", "print-access-token"]).strip()
    if not tok:
        raise SystemExit(
            "Could not get an access token. Run `gcloud auth login` (and "
            "`gcloud auth application-default login` if needed) first."
        )
    return tok


def active_project() -> str:
    proj = gcloud(["config", "get-value", "project"]).strip()
    if not proj or proj == "(unset)":
        raise SystemExit("No active gcloud project; pass --project.")
    return proj


def fetch_spot_prices(sku_keyword: str, currency: str) -> dict[str, float]:
    """region -> spot price per chip-hour, from the Billing Catalog."""
    token = access_token()
    base = (
        f"https://cloudbilling.googleapis.com/v1/services/"
        f"{COMPUTE_ENGINE_SERVICE}/skus?currencyCode={currency}&pageSize=5000"
    )
    prices: dict[str, float] = {}
    url = base
    while url:
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.load(resp)
        for sku in data.get("skus", []):
            desc = sku.get("description", "")
            if sku_keyword not in desc:
                continue
            if sku.get("category", {}).get("usageType") != "Preemptible":
                continue  # Preemptible == Spot; skips OnDemand/Commit SKUs
            rate = sku["pricingInfo"][0]["pricingExpression"]["tieredRates"][-1]
            unit = rate["unitPrice"]
            price = int(unit.get("units", 0)) + unit.get("nanos", 0) / 1e9
            for region in sku.get("serviceRegions", []):
                # Keep the cheapest if a region appears in multiple SKUs.
                if region not in prices or price < prices[region]:
                    prices[region] = price
        token_next = data.get("nextPageToken")
        url = base + f"&pageToken={token_next}" if token_next else None
    return prices


def zones_by_region(project: str) -> dict[str, list[str]]:
    out = gcloud([
        "compute", "zones", "list", "--project", project,
        "--format=csv[no-heading](name,region)",
    ])
    mapping: dict[str, list[str]] = {}
    for line in out.splitlines():
        if not line.strip():
            continue
        name, region_url = line.split(",", 1)
        region = region_url.rstrip("/").rsplit("/", 1)[-1]
        mapping.setdefault(region, []).append(name)
    return mapping


def zone_has_type(zone: str, accelerator_type: str, project: str) -> bool:
    out = gcloud([
        "compute", "tpus", "accelerator-types", "list",
        "--zone", zone, "--project", project, "--format=value(type)",
    ])
    return accelerator_type in out.split()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("-a", "--accelerator-type", default="v5litepod-4",
                    help="TPU accelerator type (default: v5litepod-4).")
    ap.add_argument("--tpu-gen", choices=sorted(GEN_INFO),
                    help="TPU generation for pricing (default: inferred from -a).")
    ap.add_argument("--price-sku-keyword",
                    help="Override the spot SKU description substring (e.g. TpuV5e).")
    ap.add_argument("--project", help="GCP project (default: active gcloud project).")
    ap.add_argument("--currency", default="USD", help="Currency code (default: USD).")
    ap.add_argument("--max-price", type=float,
                    help="Skip regions whose $/chip-hr exceeds this (speeds up the "
                         "availability scan).")
    ap.add_argument("--workers", type=int, default=16,
                    help="Parallel zone-availability probes (default: 16).")
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    args = ap.parse_args()

    gen = args.tpu_gen or infer_gen(args.accelerator_type)
    sku_keyword = args.price_sku_keyword or GEN_INFO[gen]["sku"]
    runtime = GEN_INFO[gen]["runtime"]
    project = args.project or active_project()
    chips = chip_count(args.accelerator_type)

    if not args.json:
        print(f"# TPU {gen} ({args.accelerator_type}) spot, project={project}",
              file=sys.stderr)
        print("# fetching prices from the Cloud Billing Catalog...", file=sys.stderr)
    prices = fetch_spot_prices(sku_keyword, args.currency)
    if not prices:
        raise SystemExit(
            f"No Preemptible (spot) SKUs matched {sku_keyword!r}. "
            f"Try --price-sku-keyword."
        )

    region_zones = zones_by_region(project)

    # Candidate regions: priced, optionally under --max-price, ordered cheapest first.
    candidates = sorted(prices.items(), key=lambda kv: kv[1])
    if args.max_price is not None:
        candidates = [(r, p) for r, p in candidates if p <= args.max_price]

    # Probe every zone of every candidate region for availability, in parallel.
    probe_zones = [
        (region, zone)
        for region, _ in candidates
        for zone in region_zones.get(region, [])
    ]
    if not args.json:
        print(f"# probing {len(probe_zones)} zones across {len(candidates)} "
              f"priced regions for availability...", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        avail = list(pool.map(
            lambda rz: zone_has_type(rz[1], args.accelerator_type, project),
            probe_zones,
        ))
    available_by_region: dict[str, list[str]] = {}
    for (region, zone), ok in zip(probe_zones, avail):
        if ok:
            available_by_region.setdefault(region, []).append(zone)

    results = []
    for region, price in candidates:
        zones = sorted(available_by_region.get(region, []))
        results.append({
            "region": region,
            "price_per_chip_hour": round(price, 4),
            "slice_price_per_hour": round(price * chips, 4) if chips else None,
            "available_zones": zones,
        })

    if args.json:
        print(json.dumps({
            "accelerator_type": args.accelerator_type,
            "tpu_gen": gen,
            "currency": args.currency,
            "project": project,
            "results": results,
        }, indent=2))
        return 0

    # Human-readable table.
    slice_hdr = f"slice $/hr (x{chips})" if chips else "slice $/hr"
    print()
    print(f"{'$/chip-hr':>10}  {slice_hdr:>16}  {'region':22}  available zones")
    print(f"{'-'*10}  {'-'*16}  {'-'*22}  {'-'*30}")
    recommended = None
    for r in results:
        zones = r["available_zones"]
        zlabel = ", ".join(zones) if zones else "— none (price-only, not provisionable) —"
        slice_p = f"{r['slice_price_per_hour']:.4f}" if r["slice_price_per_hour"] else "—"
        mark = " "
        if zones and recommended is None:
            recommended = r
            mark = "*"
        print(f"{r['price_per_chip_hour']:10.4f}  {slice_p:>16}  "
              f"{r['region']:22}  {mark}{zlabel}")

    print()
    if recommended is None:
        print("No priced region has the accelerator type available. "
              "Widen --max-price or check the type name.")
        return 1

    pick_zone = recommended["available_zones"][0]
    print(f"Cheapest provisionable: {recommended['region']} "
          f"@ ${recommended['price_per_chip_hour']:.4f}/chip-hr "
          f"(zone {pick_zone})")
    print("Create with:")
    print(
        f"  gcloud compute tpus tpu-vm create TPU_NAME \\\n"
        f"    --project={project} --zone={pick_zone} \\\n"
        f"    --accelerator-type={args.accelerator_type} "
        f"--version={runtime} --spot"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
