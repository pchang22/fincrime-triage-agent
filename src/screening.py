"""
Stage 3: Sanctions name screening — exact match vs. fuzzy match.

Checks every customer's name against all 39,000+ names on the OFAC list
(primary names + aliases), using three versions of increasing cleverness:

  v1  exact      names must be identical after normalizing
  v2  fuzzy      token_sort_ratio: tolerates typos and word order
  v3  fuzzy+     v2, plus token_set_ratio so a missing middle name still
                 matches, guarded so one shared surname isn't enough

Then it grades each version against the answer key from Stage 2.

Inputs:  data/screening_list.csv, data/customers.csv, data/answer_key.csv
  v4  tuned      v3, after investigating its false positives: a partial match
                 must include the person's first given name, and is capped
                 at 90 (review), never "strong"

Outputs: data/screening_results.csv   best match + scores for every customer
         data/screening_alerts.csv    customers v4 sends to review
"""

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from load_sdn import DATA_DIR, normalize_name

# Score bands (0-100). Anything >= REVIEW_AT gets an alert.
STRONG_AT = 95     # very likely the same person
REVIEW_AT = 85     # possible match: a human analyst must review

# Words that say nothing about identity, so v3's guard ignores them
# when counting how many words two names share.
FILLER = {"co", "ltd", "llc", "inc", "gmbh", "sa", "de", "la", "el", "al",
          "the", "and", "of", "company", "limited", "corp", "group", "bin", "ibn"}


def load():
    screening = pd.read_csv(DATA_DIR / "screening_list.csv")
    screening = screening.dropna(subset=["name_normalized"]).reset_index(drop=True)
    customers = pd.read_csv(DATA_DIR / "customers.csv")
    key = pd.read_csv(DATA_DIR / "answer_key.csv")
    # Customers are written "First Last" (no comma), so never flip them.
    customers["name_normalized"] = customers["name"].map(lambda n: normalize_name(n, False))
    return screening, customers, key


# ----------------------------------------------------------------------------
# The three screening versions
# ----------------------------------------------------------------------------
def screen_exact(customers, screening):
    """v1: a hit only if the normalized names are identical."""
    lookup = screening.drop_duplicates("name_normalized").set_index("name_normalized")
    rows = []
    for name in customers["name_normalized"]:
        if name in lookup.index:
            rows.append((100, lookup.index.get_loc(name)))
        else:
            rows.append((0, -1))
    # map the lookup position back to a row in the full screening table
    first_pos = screening.reset_index().drop_duplicates("name_normalized")["index"].to_numpy()
    return pd.DataFrame({"score": [s for s, _ in rows],
                         "match_idx": [first_pos[i] if i >= 0 else -1 for _, i in rows]})


def best_matches(scores):
    """For each customer (row), the best score and which list entry gave it."""
    idx = scores.argmax(axis=1)
    return pd.DataFrame({"score": scores[np.arange(len(idx)), idx].astype(int),
                         "match_idx": idx})


def screen_fuzzy(customers, screening):
    """v2: token_sort_ratio sorts the words first, so order doesn't matter,
    then measures how many character edits separate the two names."""
    scores = process.cdist(customers["name_normalized"], screening["name_normalized"],
                           scorer=fuzz.token_sort_ratio, dtype=np.uint8, workers=-1)
    return best_matches(scores), scores


def shared_words(a, b):
    return len((set(a.split()) & set(b.split())) - FILLER)


def screen_fuzzy_plus(customers, screening, sort_scores):
    """v3: also try token_set_ratio, which scores 100 when one name's words are
    all contained in the other ('andrey anikeyev' inside 'andrey anatolyevich
    anikeyev'). On its own that is far too loose: 'maria hernandez' would score
    100 against anyone named Hernandez. Guard: only trust it when the names
    share at least 2 meaningful words."""
    set_scores = process.cdist(customers["name_normalized"], screening["name_normalized"],
                               scorer=fuzz.token_set_ratio, dtype=np.uint8, workers=-1)
    names = screening["name_normalized"].to_numpy()
    out = []
    for i, cust_name in enumerate(customers["name_normalized"]):
        best_score = int(sort_scores[i].max())
        best_idx = int(sort_scores[i].argmax())
        # look at the strongest token_set candidates, best first
        for j in np.argsort(set_scores[i])[::-1][:25]:
            s = int(set_scores[i, j])
            if s <= best_score:
                break
            if shared_words(cust_name, names[j]) >= 2:
                best_score, best_idx = s, int(j)
                break
        out.append((best_score, best_idx))
    return pd.DataFrame(out, columns=["score", "match_idx"])


PARTIAL_CAP = 90   # v4: a partial (missing-words) match is never "strong"


def given_first_name(raw_name, entry_type):
    """'ABU MARZOOK, Mousa Mohammed' -> 'mousa'. None for companies etc."""
    if entry_type != "individual" or not isinstance(raw_name, str) or "," not in raw_name:
        return None
    given = normalize_name(raw_name.split(",", 1)[1], False)
    return given.split()[0] if given else None


def screen_v4(customers, screening, sort_scores):
    """v4 = v3 with two fixes found by investigating v3's false positives:
    1. A partial match must contain the listed person's first given name.
       ('James Wilson' matched 'JAMES WILSON, Alejandro Antonio' because
       the customer's first name matched half of the listed SURNAME.)
    2. A partial match is capped at 90: missing words mean less certainty,
       so it goes to human review rather than being called a strong match."""
    set_scores = process.cdist(customers["name_normalized"], screening["name_normalized"],
                               scorer=fuzz.token_set_ratio, dtype=np.uint8, workers=-1)
    names = screening["name_normalized"].to_numpy()
    firsts = [given_first_name(n, t) for n, t in zip(screening["name"], screening["type"])]
    out = []
    for i, cust_name in enumerate(customers["name_normalized"]):
        cust_words = set(cust_name.split())
        best_score = int(sort_scores[i].max())
        best_idx = int(sort_scores[i].argmax())
        for j in np.argsort(set_scores[i])[::-1][:25]:
            s = min(int(set_scores[i, j]), PARTIAL_CAP)
            if s <= best_score:
                break
            if shared_words(cust_name, names[j]) < 2:
                continue
            if firsts[j] is not None and firsts[j] not in cust_words:
                continue
            best_score, best_idx = s, int(j)
            break
        out.append((best_score, best_idx))
    return pd.DataFrame(out, columns=["score", "match_idx"])


# ----------------------------------------------------------------------------
# Grading against the answer key
# ----------------------------------------------------------------------------
def grade(result, key, threshold=REVIEW_AT):
    flagged = result["score"] >= threshold
    is_hit = key["category"] == "sanctions"
    return {
        "caught": int((flagged & is_hit).sum()),
        "missed": int((~flagged & is_hit).sum()),
        "false_positives": int((flagged & ~is_hit).sum()),
        "traps_flagged": int((flagged & (key["category"] == "trap")).sum()),
    }


def band(score):
    if score >= STRONG_AT:
        return "strong_match"
    if score >= REVIEW_AT:
        return "possible_match"
    return "clear"


def main():
    screening, customers, key = load()
    key = key.set_index("customer_id").loc[customers["customer_id"]].reset_index()
    print(f"Screening {len(customers):,} customers against {len(screening):,} list names...\n")

    v1 = screen_exact(customers, screening)
    v2, sort_scores = screen_fuzzy(customers, screening)
    v3 = screen_fuzzy_plus(customers, screening, sort_scores)
    v4 = screen_v4(customers, screening, sort_scores)
    versions = {"v1 exact": v1, "v2 fuzzy": v2, "v3 fuzzy+": v3, "v4 tuned": v4}

    # --- Scorecard ---
    n_hits = int((key["category"] == "sanctions").sum())
    print(f"Scorecard (alert if score >= {REVIEW_AT}; {n_hits} planted hits, "
          f"{(key['category'] == 'trap').sum()} traps):")
    print(f"  {'version':<11}{'caught':>8}{'missed':>8}{'false pos':>11}{'traps hit':>11}")
    for name, res in versions.items():
        g = grade(res, key)
        print(f"  {name:<11}{g['caught']:>5}/{n_hits:<2}{g['missed']:>8}"
              f"{g['false_positives']:>11}{g['traps_flagged']:>11}")

    # --- Which planted hits each version missed, and why that matters ---
    print("\nPlanted hits — best score per version:")
    hits = key["category"] == "sanctions"
    for i in key.index[hits]:
        scores = "  ".join(f"{versions[v].loc[i, 'score']:>3}" for v in versions)
        print(f"  {key.loc[i, 'reason']:<26} {customers.loc[i, 'name'][:32]:<33} {scores}")
    print(f"  {'':<26} {'':<33} {'v1':>3}  {'v2':>3}  {'v3':>3}  {'v4':>3}")

    # --- Threshold sweep for v4: the core tradeoff ---
    print("\nv4 threshold sweep (lower threshold = catch more, but more false alarms):")
    print(f"  {'threshold':>9}{'caught':>8}{'false pos':>11}")
    for t in [70, 75, 80, 85, 90, 95, 100]:
        g = grade(v4, key, t)
        print(f"  {t:>9}{g['caught']:>5}/{n_hits:<2}{g['false_positives']:>11}")

    # --- Save results (v4 is the version we use going forward) ---
    m = screening.loc[v4["match_idx"]].reset_index(drop=True)
    results = pd.DataFrame({
        "customer_id": customers["customer_id"],
        "customer_name": customers["name"],
        "score_v1": v1["score"], "score_v2": v2["score"], "score_v3": v3["score"],
        "score_v4": v4["score"],
        "decision": v4["score"].map(band),
        "matched_list_name": m["name"],
        "matched_ent_num": m["ent_num"],
        "matched_program": m["program"],
        "matched_is_alias": m["is_alias"],
        # secondary identifiers, used by the Stage 5 reviewer (not for scoring)
        "matched_dob": m.get("dob"),
        "matched_dob_years": m.get("dob_years"),
        "matched_nationality": m.get("nationality"),
    })
    results.to_csv(DATA_DIR / "screening_results.csv", index=False, encoding="utf-8-sig")
    alerts = results[results["decision"] != "clear"].sort_values("score_v4", ascending=False)
    alerts.to_csv(DATA_DIR / "screening_alerts.csv", index=False, encoding="utf-8-sig")

    print(f"\nv4 decisions: {results['decision'].value_counts().to_dict()}")
    print("Saved data/screening_results.csv and data/screening_alerts.csv")


if __name__ == "__main__":
    main()
