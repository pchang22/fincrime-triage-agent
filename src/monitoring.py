"""
Stage 4: Transaction monitoring — catching suspicious BEHAVIOUR.

Stage 3 asked "is this person on a sanctions list?"
Stage 4 asks  "is this person's money doing something suspicious?"

Five rules, one per classic anti-money-laundering (AML) red flag:

  structuring          cash deposits split to stay under the $10,000 reporting line
  rapid_movement       big money comes in and almost all of it leaves within 48h
  high_risk_geography  money sent to / received from high-risk jurisdictions
  velocity_spike       a sudden burst of activity far above the customer's normal
  round_amounts        repeated transfers of suspiciously round sums

Every alert, plus Stage 3's sanctions alerts, is merged into ONE case per
customer: the queue an analyst (or, in Stage 5, an AI agent) works through.

Inputs:  data/transactions.csv, data/customers.csv, data/answer_key.csv,
         data/screening_alerts.csv (from Stage 3)
Outputs: data/monitoring_alerts.csv   one row per rule that fired
         data/cases.csv               one row per customer needing review
"""

from datetime import timedelta
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")

# ---- Rule settings: the "dials" a bank tunes (and must justify) ----
REPORTING_THRESHOLD = 10_000   # US: cash over $10k triggers a Currency Transaction Report
STRUCT_FLOOR = 8_000           # "just under" the threshold starts here
STRUCT_MIN_DEPOSITS = 3        # ...this many of them...
STRUCT_WINDOW_DAYS = 7         # ...within this many days

RAPID_MIN_INBOUND = 10_000     # only look at large incoming amounts
RAPID_WINDOW_HOURS = 48
RAPID_MIN_PASS_THROUGH = 0.90  # >= 90% of it leaves again

HIGH_RISK_COUNTRIES = {"Iran", "North Korea", "Myanmar", "Cuba"}

VELOCITY_MIN_TXNS = 8          # busiest 7 days must have at least this many txns,
VELOCITY_MIN_AMOUNT = 10_000   # move at least this much money,
VELOCITY_MULTIPLE = 5          # and be 5x the customer's normal weekly pace.
# Tuning note: with only "5 txns at 5x normal", 13 of 470 normal customers were
# flagged: quiet customers who made 5 small purchases in one week ($500-$7,500).
# Count alone is noise; the planted spikes were 11-12 txns moving ~$28k.

ROUND_UNIT = 1_000             # "round" = an exact multiple of $1,000
ROUND_MIN_COUNT = 3


def money(x):
    return f"${x:,.0f}"


def dates(ts):
    return f"{ts.min():%b %d}–{ts.max():%b %d}" if len(ts) > 1 else f"{ts.min():%b %d}"


# ----------------------------------------------------------------------------
# The five rules. Each returns a list of alerts (dicts).
# ----------------------------------------------------------------------------
def rule_structuring(txns):
    cash = txns[(txns.channel == "cash") & (txns.direction == "in")
                & txns.amount.between(STRUCT_FLOOR, REPORTING_THRESHOLD - 0.01)]
    alerts = []
    for cid, g in cash.groupby("customer_id"):
        g = g.sort_values("timestamp")
        # slide a 7-day window over this customer's near-threshold deposits
        best = None
        for start in g.timestamp:
            window = g[(g.timestamp >= start) &
                       (g.timestamp < start + timedelta(days=STRUCT_WINDOW_DAYS))]
            if best is None or len(window) > len(best):
                best = window
        if len(best) >= STRUCT_MIN_DEPOSITS:
            alerts.append({
                "customer_id": cid, "rule": "structuring",
                "risk_score": min(95, 60 + 10 * (len(best) - STRUCT_MIN_DEPOSITS + 1)),
                "details": (f"{len(best)} cash deposits between {money(STRUCT_FLOOR)} and "
                            f"{money(REPORTING_THRESHOLD)} within {STRUCT_WINDOW_DAYS} days "
                            f"({dates(best.timestamp)}), total {money(best.amount.sum())}"),
                "txn_ids": ";".join(best.txn_id),
            })
    return alerts


def rule_rapid_movement(txns):
    alerts = []
    big_in = txns[(txns.direction == "in") & (txns.amount >= RAPID_MIN_INBOUND)]
    outs = txns[txns.direction == "out"]
    for _, t in big_in.iterrows():
        o = outs[(outs.customer_id == t.customer_id) & (outs.timestamp > t.timestamp) &
                 (outs.timestamp <= t.timestamp + timedelta(hours=RAPID_WINDOW_HOURS))]
        share = o.amount.sum() / t.amount
        if share >= RAPID_MIN_PASS_THROUGH:
            hours = (o.timestamp.max() - t.timestamp).total_seconds() / 3600
            alerts.append({
                "customer_id": t.customer_id, "rule": "rapid_movement",
                "risk_score": round(70 + 20 * min(1, (share - 0.9) / 0.1)),
                "details": (f"{money(t.amount)} received {t.timestamp:%b %d} from "
                            f"{t.counterparty_country}; {share:.0%} sent out again within "
                            f"{hours:.0f}h in {len(o)} payment(s) to "
                            f"{', '.join(sorted(o.counterparty_country.unique()))}"),
                "txn_ids": ";".join([t.txn_id, *o.txn_id]),
            })
    return alerts


def rule_high_risk_geography(txns):
    hr = txns[txns.counterparty_country.isin(HIGH_RISK_COUNTRIES)]
    alerts = []
    for cid, g in hr.groupby("customer_id"):
        directions = ", ".join(f"{n} {d}" for d, n in g.direction.value_counts().items())
        alerts.append({
            "customer_id": cid, "rule": "high_risk_geography",
            "risk_score": 95 if len(g) > 1 or g.amount.sum() > 10_000 else 85,
            "details": (f"{len(g)} payment(s) ({directions}) involving "
                        f"{', '.join(sorted(g.counterparty_country.unique()))}, "
                        f"total {money(g.amount.sum())} ({dates(g.timestamp)})"),
            "txn_ids": ";".join(g.txn_id),
        })
    return alerts


def rule_velocity_spike(txns):
    alerts = []
    period_days = (txns.timestamp.max() - txns.timestamp.min()).days + 1
    n_weeks = period_days / 7
    for cid, g in txns.groupby("customer_id"):
        if len(g) < VELOCITY_MIN_TXNS:
            continue
        daily = g.set_index("timestamp").resample("D").size()
        rolling = daily.rolling(7, min_periods=1).sum()
        peak = int(rolling.max())
        baseline = (len(g) - peak) / max(n_weeks - 1, 1)   # normal txns per week
        ratio = peak / max(baseline, 0.5)
        if peak >= VELOCITY_MIN_TXNS and ratio >= VELOCITY_MULTIPLE:
            end = rolling.idxmax()
            burst = g[(g.timestamp > end - timedelta(days=7)) & (g.timestamp <= end + timedelta(days=1))]
            if burst.amount.sum() < VELOCITY_MIN_AMOUNT:
                continue
            alerts.append({
                "customer_id": cid, "rule": "velocity_spike",
                "risk_score": min(90, round(60 + 3 * (ratio - VELOCITY_MULTIPLE))),
                "details": (f"{peak} transactions in 7 days ending {end:%b %d} "
                            f"(total {money(burst.amount.sum())}) vs. a normal pace of "
                            f"~{baseline:.1f}/week ({ratio:.0f}x)"),
                "txn_ids": ";".join(burst.txn_id),
            })
    return alerts


def rule_round_amounts(txns):
    r = txns[(txns.amount >= ROUND_UNIT) & (txns.amount % ROUND_UNIT == 0)]
    alerts = []
    for cid, g in r.groupby("customer_id"):
        if len(g) >= ROUND_MIN_COUNT:
            amounts = ", ".join(f"{n}x {money(a)}" for a, n in g.amount.value_counts().items())
            alerts.append({
                "customer_id": cid, "rule": "round_amounts",
                "risk_score": min(80, 50 + 10 * (len(g) - ROUND_MIN_COUNT + 1)),
                "details": f"{len(g)} transfers of exact round amounts ({amounts}), {dates(g.timestamp)}",
                "txn_ids": ";".join(g.txn_id),
            })
    return alerts


RULES = [rule_structuring, rule_rapid_movement, rule_high_risk_geography,
         rule_velocity_spike, rule_round_amounts]


# ----------------------------------------------------------------------------
# Build the case queue
# ----------------------------------------------------------------------------
def build_cases(monitoring, screening, customers):
    """One case per customer. Screening and monitoring alerts are combined,
    because one person with two red flags is riskier than either alone."""
    rows = []
    for _, s in screening.iterrows():
        rows.append({
            "customer_id": s.customer_id, "rule": f"sanctions_{s.decision}",
            "risk_score": int(s.score_v4),
            "details": (f"Name screening: '{s.customer_name}' scored {int(s.score_v4)}/100 "
                        f"against OFAC SDN entry '{s.matched_list_name}' "
                        f"(ent_num {s.matched_ent_num}, program {s.matched_program}"
                        f"{', alias' if str(s.matched_is_alias) == 'True' else ''})"),
            "txn_ids": "",
        })
    all_alerts = pd.concat([monitoring, pd.DataFrame(rows)], ignore_index=True)

    cases = []
    for cid, g in all_alerts.groupby("customer_id"):
        cust = customers.loc[cid]
        top = int(g.risk_score.max())
        cases.append({
            "customer_id": cid,
            "customer_name": cust["name"],
            "customer_type": cust["customer_type"],
            "country": cust["country"],
            "alert_count": len(g),
            "rules_triggered": ";".join(g.rule),
            # one extra red flag adds 5 points, capped at 100
            "case_risk": min(100, top + 5 * (len(g) - 1)),
            "alert_details": " | ".join(g.details),
            "txn_ids": ";".join(t for t in g.txn_ids if t),
        })
    cases = pd.DataFrame(cases).sort_values("case_risk", ascending=False).reset_index(drop=True)
    cases.insert(0, "case_id", [f"CASE-{i:04d}" for i in range(1, len(cases) + 1)])
    return cases


def main():
    txns = pd.read_csv(DATA_DIR / "transactions.csv", parse_dates=["timestamp"])
    customers = pd.read_csv(DATA_DIR / "customers.csv", encoding="utf-8-sig").set_index("customer_id")
    key = pd.read_csv(DATA_DIR / "answer_key.csv", encoding="utf-8-sig").set_index("customer_id")
    screening = pd.read_csv(DATA_DIR / "screening_alerts.csv", encoding="utf-8-sig")
    print(f"Monitoring {len(txns):,} transactions for {len(customers):,} customers...\n")

    monitoring = pd.DataFrame([a for rule in RULES for a in rule(txns)])
    monitoring.to_csv(DATA_DIR / "monitoring_alerts.csv", index=False, encoding="utf-8-sig")

    # --- Scorecard: did each rule catch the customers planted with its pattern? ---
    print("Rule scorecard (planted = 3 customers per pattern):")
    print(f"  {'rule':<22}{'caught':>8}{'other flags':>13}  example of what an 'other flag' was")
    for rule in [r.__name__.replace("rule_", "") for r in RULES]:
        flagged = set(monitoring.loc[monitoring.rule == rule, "customer_id"])
        planted = set(key.index[key.reason == rule])
        other = flagged - planted
        example = ""
        if other:
            ex = sorted(other)[0]
            example = f"{ex} ({key.loc[ex, 'reason']})"
        print(f"  {rule:<22}{len(flagged & planted):>5}/{len(planted):<2}{len(other):>13}  {example}")

    behaviour = set(key.index[key.category == "behaviour"])
    caught = behaviour & set(monitoring.customer_id)
    print(f"\nBehaviour cases caught overall: {len(caught)}/{len(behaviour)}")

    # --- The combined case queue ---
    cases = build_cases(monitoring, screening, customers)
    cases.to_csv(DATA_DIR / "cases.csv", index=False, encoding="utf-8-sig")

    cases["truth"] = cases.customer_id.map(key.category)
    should = set(key.index[key.should_flag])
    print(f"\nCase queue: {len(cases)} customers out of {len(customers)} "
          f"({len(cases) / len(customers):.1%}) need review")
    print(f"  contains {len(should & set(cases.customer_id))}/{len(should)} of everyone who "
          f"should be flagged; {(~cases.customer_id.isin(should)).sum()} are false alarms")
    print("\nTop 5 cases by risk:")
    for _, c in cases.head(5).iterrows():
        print(f"  {c.case_id}  risk {c.case_risk:>3}  {c.customer_name[:28]:<29} {c.rules_triggered}")
    print("\nSaved data/monitoring_alerts.csv and data/cases.csv")


if __name__ == "__main__":
    main()
