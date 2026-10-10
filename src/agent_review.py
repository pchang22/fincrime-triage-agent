"""
Stage 5: Case reviewer — the "make the queue disappear" step.

Reads every case from Stage 4 and decides: CLOSE (false alarm) or ESCALATE
(a human investigator must look). Two interchangeable reviewers:

  rules    a deterministic rulebook (no API key needed)
  claude   an AI model via the Anthropic API (needs an API key)

Run:   python src/agent_review.py            (uses claude if a key exists, else rules)
       python src/agent_review.py rules      (force the rulebook)
       python src/agent_review.py claude     (force the AI)

Whichever reviewer is used, GUARDRAILS in code have the final say: a case
may only be auto-closed if the reviewer is highly confident AND it has no
suspicious-behaviour alerts AND no strong sanctions match AND the customer's
date of birth doesn't match a listed one AND (when no DOB comparison is
possible) the listed first name and surname don't both appear in the name.
The AI advises; the code enforces policy.

Inputs:  data/cases.csv, data/screening_alerts.csv, data/customers.csv,
         data/transactions.csv, data/answer_key.csv (grading only — never
         shown to the reviewer)
Output:  data/review_decisions.csv
"""

import json
import os
import sys
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from load_sdn import normalize_name

DATA_DIR = Path("data")
MODEL = "claude-sonnet-5-5"     # cheaper alternative: "claude-haiku-5-5"
MAX_TXNS_SHOWN = 15


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load():
    read = lambda f, **kw: pd.read_csv(DATA_DIR / f, encoding="utf-8-sig", **kw)
    cases = read("cases.csv")
    screening = read("screening_alerts.csv").set_index("customer_id")
    customers = read("customers.csv").set_index("customer_id")
    txns = read("transactions.csv").set_index("txn_id")
    key = read("answer_key.csv").set_index("customer_id")
    return cases, screening, customers, txns, key


def behaviour_rules(case):
    return [r for r in str(case.rules_triggered).split(";") if not r.startswith("sanctions_")]


# ----------------------------------------------------------------------------
# Reviewer 1: the rulebook
# ----------------------------------------------------------------------------
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def compare_names(customer_name, list_name):
    """Split the OFAC name ('SURNAME, Given Names') and check, separately,
    whether the customer's name contains the listed given name and surname."""
    cust = normalize_name(customer_name, False).split()
    if "," not in str(list_name):
        return None                                   # a company: no name parts
    surname = normalize_name(list_name.split(",", 1)[0], False).split()
    given = normalize_name(list_name.split(",", 1)[1], False).split()
    if not cust or not surname or not given:
        return None
    best = lambda word: max(similar(word, c) for c in cust)
    return {
        "given": given[0], "surname": " ".join(surname),
        "given_score": best(given[0]),                 # 1.0 = exact word present
        "surname_score": max(best(s) for s in surname),
    }


DOB_MISMATCH_YEARS = 3   # DOBs this many years apart = different people


def compare_dob(customer, screen):
    """Compare the customer's date of birth with the listed person's DOB(s).
    status: match (same year) | near (1-2 years) | mismatch (3+) | unknown."""
    out = {"status": "unknown", "gap": None,
           "customer_dob": customer.get("date_of_birth") if customer is not None else None,
           "listed_dob": getattr(screen, "matched_dob", None) if screen is not None else None}
    years = getattr(screen, "matched_dob_years", None) if screen is not None else None
    cust_dob = out["customer_dob"]
    if not isinstance(years, str) or not isinstance(cust_dob, str):
        return out
    cust_year = int(cust_dob[:4])
    gap = min(abs(cust_year - int(y)) for y in years.split("|"))
    out["gap"] = gap
    out["status"] = ("match" if gap == 0 else
                     "near" if gap < DOB_MISMATCH_YEARS else "mismatch")
    return out


def dob_sentence(dob):
    if dob["status"] == "unknown":
        why = "no DOB listed for the sanctioned person" if dob["customer_dob"] else "no customer DOB on file"
        return f"DOB check: not possible ({why})."
    return (f"DOB check: customer {dob['customer_dob']} vs listed {dob['listed_dob']} -> "
            f"{dob['status']} (closest listed year {dob['gap']} year(s) apart).")


def review_rules(case, screen, dob):
    """Deterministic policy. Returns (decision, confidence, reason)."""
    behaviour = behaviour_rules(case)
    if behaviour:
        return ("escalate", "high",
                f"Suspicious activity ({', '.join(behaviour)}) requires investigator "
                f"review and a possible SAR decision.")
    if screen is None:
        return "escalate", "low", "No screening details found; cannot assess."
    if screen.decision == "strong_match":
        return ("escalate", "high",
                f"Strong name match ({int(screen.score_v4)}/100) to SDN '{screen.matched_list_name}'. "
                f"Treat as a likely true hit: hold activity pending investigation.")

    # Date of birth is the strongest evidence, so check it first.
    if dob["status"] in ("match", "near"):
        return ("escalate", "high",
                f"Possible name match to '{screen.matched_list_name}' and the birth year "
                f"lines up. {dob_sentence(dob)} Treat as a potential true hit.")
    if dob["status"] == "mismatch":
        return ("close", "high",
                f"Name resembles '{screen.matched_list_name}', but {dob_sentence(dob)} "
                f"A {dob['gap']}-year gap means a different person.")

    # No DOB to compare: fall back to judging the names alone.
    parts = compare_names(screen.customer_name, screen.matched_list_name)
    if parts is None:
        return "escalate", "low", "Matched an entity name; needs manual comparison."
    g_ok = parts["given_score"] >= 0.85        # same first name, or a typo of it
    s_ok = parts["surname_score"] >= 0.80      # surname present, or a typo of it
    if g_ok and s_ok:
        return ("escalate", "medium",
                f"Given name '{parts['given']}' and surname '{parts['surname']}' both appear "
                f"(possibly with variations or missing names). Plausible true match.")
    if g_ok and not s_ok:
        return ("close", "high",
                f"Shares only the given name '{parts['given']}'; surname differs clearly from "
                f"listed '{parts['surname']}' (similarity {parts['surname_score']:.2f}). "
                f"Score is driven by a common first name.")
    if s_ok and not g_ok:
        # Not closed: a rule can't tell 'Madison' vs 'Mason' (different names,
        # similarity 0.83) from 'Mohamad' vs 'Muhammad' (same name transliterated,
        # 0.80). Judgment needed -> a human (or the AI reviewer) decides.
        return ("escalate", "medium",
                f"Surname matches '{parts['surname']}' but given name differs from listed "
                f"'{parts['given']}' (similarity {parts['given_score']:.2f}). Could be a "
                f"different person or a transliteration; needs judgment.")
    return "escalate", "low", "Unclear name evidence; human review needed."


# ----------------------------------------------------------------------------
# Reviewer 2: the AI model
# ----------------------------------------------------------------------------
POLICY = """You are a Level-1 financial-crime operations analyst at a U.S. bank.
You review one alert case at a time and decide whether to CLOSE it as a false
alarm or ESCALATE it to a human investigator.

Policy:
1. Transaction-monitoring alerts (structuring, rapid movement, high-risk
   geography, velocity spike, round amounts): we have no customer due-diligence
   information in this exercise, so escalate them, and explain the risk.
2. Sanctions name alerts. Date of birth is the strongest evidence:
   a. If the customer's DOB matches, or is within 2 years of, any DOB listed for
      the sanctioned person: escalate.
   b. If the customer's DOB differs from EVERY listed DOB by 3 or more years:
      you may close it as a different person, even if the names are similar.
   c. If either DOB is missing, judge on names alone: close ONLY if the name
      evidence clearly shows a different person (e.g. surname clearly different
      and only a common given name shared). Spelling variants, transliterations,
      nicknames, reordered names, missing middle names, and a dropped second
      surname (common in Spanish naming) are NOT grounds to close.
3. Do not clear a sanctions alert on the customer's country of residence or the
   listed nationality alone: sanctioned people often live abroad.
4. When in doubt, escalate. Missing a sanctioned person or a launderer costs far
   more than an extra review.

Write the reason the way an analyst writes a case note: specific, 2-3 sentences."""

SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["close", "escalate"]},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "reason": {"type": "string"},
        "next_step": {"type": "string"},
    },
    "required": ["decision", "confidence", "reason", "next_step"],
    "additionalProperties": False,
}


def describe_case(case, customer, txns, screen, dob):
    """Turn a case into the plain-text brief the AI reads."""
    cust_dob = customer.get("date_of_birth")
    lines = [
        f"Case {case.case_id} | customer {case.customer_id}: {case.customer_name} "
        f"({case.customer_type}, resident in {case.country}, account opened "
        f"{customer.account_opened}, date of birth "
        f"{cust_dob if isinstance(cust_dob, str) else 'not on file'})",
        f"Alerts ({case.alert_count}):",
        *[f"  - {d}" for d in str(case.alert_details).split(" | ")],
    ]
    if screen is not None:
        listed = getattr(screen, "matched_dob", None)
        nat = getattr(screen, "matched_nationality", None)
        lines += [
            "Listed (sanctioned) person's identifiers:",
            f"  - DOB: {listed if isinstance(listed, str) else 'not listed'}",
            f"  - Nationality: {nat if isinstance(nat, str) else 'not listed'}",
            f"  - {dob_sentence(dob)}",
        ]
    ids = [t for t in str(case.txn_ids).split(";") if t and t != "nan"][:MAX_TXNS_SHOWN]
    if ids:
        lines.append("Flagged transactions:")
        for tid in ids:
            t = txns.loc[tid]
            lines.append(f"  - {t.timestamp} {t.direction:>3} {t.channel:<5} "
                         f"${t.amount:>11,.2f}  {t.counterparty_name} ({t.counterparty_country})")
    return "\n".join(lines)


def load_api_key():
    """Look for ANTHROPIC_API_KEY in the environment, then in a .env file."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    env = Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith("ANTHROPIC_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def make_ai_reviewer(api_key):
    import anthropic                       # only needed in AI mode
    client = anthropic.Anthropic(api_key=api_key)

    def review(case, customer, txns, screen, dob):
        response = client.messages.create(
            model=MODEL,
            max_tokens=600,
            system=POLICY,
            messages=[{"role": "user", "content": describe_case(case, customer, txns, screen, dob)}],
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        )
        text = next(b.text for b in response.content if b.type == "text")
        out = json.loads(text)
        return out["decision"], out["confidence"], f"{out['reason']} Next step: {out['next_step']}"

    return review


# ----------------------------------------------------------------------------
# Guardrails: code, not the reviewer, has the final say on auto-closing
# ----------------------------------------------------------------------------
def apply_guardrails(case, decision, confidence, screen, dob):
    if decision != "close":
        return decision, ""
    if behaviour_rules(case):
        return "escalate", "guardrail: behaviour alerts can't be auto-closed"
    if "sanctions_strong_match" in str(case.rules_triggered):
        return "escalate", "guardrail: strong sanctions matches can't be auto-closed"
    if dob["status"] in ("match", "near"):
        return "escalate", "guardrail: customer DOB matches (or is near) a listed DOB"
    if screen is not None and dob["status"] == "unknown":
        parts = compare_names(screen.customer_name, screen.matched_list_name)
        if parts and parts["given_score"] >= 0.85 and parts["surname_score"] >= 0.80:
            return "escalate", ("guardrail: listed first name AND surname both appear "
                                "in the customer's name")
    if confidence != "high":
        return "escalate", f"guardrail: auto-close needs high confidence (got {confidence})"
    return "close", ""


# ----------------------------------------------------------------------------
def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "auto"
    api_key = load_api_key()
    if mode == "auto":
        mode = "claude" if api_key else "rules"
    if mode == "claude" and not api_key:
        raise SystemExit("No ANTHROPIC_API_KEY found (environment or .env file).")
    if mode == "claude":
        gitignore = Path(".gitignore")
        if not gitignore.exists() or ".env" not in gitignore.read_text().split():
            print("WARNING: add '.env' to .gitignore so your API key never reaches GitHub.\n")
        ai_review = make_ai_reviewer(api_key)

    cases, screening, customers, txns, key = load()
    reviewer_name = MODEL if mode == "claude" else "rules"
    print(f"Reviewing {len(cases)} cases with: {reviewer_name}\n")

    rows = []
    for case in cases.itertuples():
        screen = screening.loc[case.customer_id] if case.customer_id in screening.index else None
        customer = customers.loc[case.customer_id]
        dob = compare_dob(customer, screen)
        try:
            if mode == "claude":
                decision, confidence, reason = ai_review(case, customer, txns, screen, dob)
            else:
                decision, confidence, reason = review_rules(case, screen, dob)
        except Exception as e:                     # an AI/API failure must never close a case
            decision, confidence, reason = "escalate", "low", f"Reviewer error, escalated: {e}"
        final, override = apply_guardrails(case, decision, confidence, screen, dob)
        rows.append({
            "case_id": case.case_id, "customer_id": case.customer_id,
            "customer_name": case.customer_name, "case_risk": case.case_risk,
            "rules_triggered": case.rules_triggered, "reviewer": reviewer_name,
            "dob_check": dob["status"] if screen is not None else "",
            "reviewer_decision": decision, "confidence": confidence,
            "final_decision": final, "guardrail_note": override, "reason": reason,
        })
        mark = "CLOSE   " if final == "close" else "ESCALATE"
        print(f"  {case.case_id} {mark} ({confidence:<6}) {case.customer_name[:30]}")

    out = pd.DataFrame(rows)
    out.to_csv(DATA_DIR / "review_decisions.csv", index=False, encoding="utf-8-sig")

    # --- Grade against the answer key (the reviewer never saw this) ---
    out["truth"] = out.customer_id.map(key.should_flag)
    closed = out[out.final_decision == "close"]
    escalated = out[out.final_decision == "escalate"]
    bad_closes = closed[closed.truth]
    false_alarms = out[~out.truth]
    print(f"\nResults ({reviewer_name}):")
    print(f"  Queue in:                {len(out)} cases")
    print(f"  Auto-closed:             {len(closed)}  "
          f"({len(closed[~closed.truth])}/{len(false_alarms)} false alarms cleared)")
    print(f"  Escalated to a human:    {len(escalated)}")
    print(f"  Real cases wrongly closed: {len(bad_closes)}   <- must be 0")
    print(f"  Guardrail overrides:     {(out.guardrail_note != '').sum()}")
    sanc = out[out.dob_check != ""]
    print(f"  DOB evidence (sanctions cases): {sanc.dob_check.value_counts().to_dict()}")
    if len(bad_closes):
        print("  !! Wrongly closed:", ", ".join(bad_closes.customer_name))
    if len(closed):
        print("\nAuto-closed cases and why:")
        for r in closed.itertuples():
            print(f"  {r.case_id} {r.customer_name}: {r.reason}")
    print("\nSaved data/review_decisions.csv")


if __name__ == "__main__":
    main()
