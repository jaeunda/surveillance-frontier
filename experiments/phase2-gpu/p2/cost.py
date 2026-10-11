"""Spending of the paid steps (README Part II "Budget", Part III "Formal machines"): instance-hours from EC2 launch to
termination (ec2*.json written by collect_env.py --ec2-times after termination) times the official on-demand price
recorded with them. Storage, transfer and bills are reported separately by hand."""

from pathlib import Path

from .common import RESULTS, read_json


def ec2_rows(roots):
    rows = []
    for root in roots:
        for f in sorted(Path(root).rglob("ec2*.json")):
            e = read_json(f)
            price = (e.get("on_demand_price") or {}).get("price_usd_per_hour")
            hours = e.get("instance_hours")
            rows.append({"record": str(f.relative_to(RESULTS)), "instance_id": e.get("instance_id"),
                         "instance_type": e.get("instance_type"), "purchase": e.get("purchase_option"),
                         "on_demand_usd_per_h": price, "instance_hours": hours,
                         "compute_usd_on_demand": price * hours if price and hours else None,
                         "spot_price_at_launch": (e.get("spot_price_at_launch") or {}).get("price_usd_per_hour")})
    return rows


def recorded_usd(roots):
    rows = ec2_rows(roots)
    return sum(r["compute_usd_on_demand"] or 0 for r in rows), [r["record"] for r in rows if r["compute_usd_on_demand"] is None]
