"""
Stage 2: Generate a fake bank — customers, transactions, and an answer key.

Inputs:  data/screening_list.csv   (made by Stage 1: src/load_sdn.py)
Outputs: data/customers.csv        who banks with us
         data/transactions.csv     what their money did (Jul–Sep 2026)
         data/answer_key.csv       who SHOULD be caught, and why

The answer key is never shown to the screening or monitoring code.
It exists only so we can measure how well those later stages work.
"""

import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from faker import Faker

SEED = 42                 # same seed -> same fake data every run
N_CUSTOMERS = 500
DATA_DIR = Path("data")
START = datetime(2026, 7, 1)
END = datetime(2026, 9, 30, 23, 59)

# Faker locale -> the country we record for customers made with it
LOCALES = {
    "en_US": "United States", "en_CA": "Canada", "en_GB": "United Kingdom",
    "de_DE": "Germany", "fr_FR": "France", "es_MX": "Mexico",
    "pt_BR": "Brazil", "it_IT": "Italy", "nl_NL": "Netherlands",
    "en_IN": "India",
}
LOW_RISK_COUNTRIES = list(LOCALES.values()) + [
    "Singapore", "Australia", "Japan", "Switzerland",
]
# Illustrative only: FATF "call for action" jurisdictions plus Cuba (US
# embargo). Real banks maintain this list carefully and update it often.
HIGH_RISK_COUNTRIES = ["Iran", "North Korea", "Myanmar", "Cuba"]

random.seed(SEED)
Faker.seed(SEED)
FAKERS = {loc: Faker(loc) for loc in LOCALES}
COMPANY_FAKER = Faker("en_US")


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def random_time(start=START, end=END):
    seconds = int((end - start).total_seconds())
    return start + timedelta(seconds=random.randint(0, seconds))


def not_round(amount):
    """Normal customers shouldn't accidentally send perfectly round amounts."""
    amount = round(amount, 2)
    if amount >= 100 and amount % 100 == 0:
        amount += 0.37
    return amount


def misspell(word):
    """Introduce one realistic typo: swap, drop, double, or swap a vowel."""
    vowels = "aeiou"
    swaps = {"a": "e", "e": "i", "i": "e", "o": "u", "u": "o"}
    for _ in range(20):
        w = list(word)
        op = random.choice(["swap", "drop", "double", "vowel"])
        i = random.randint(1, len(w) - 2)
        if op == "swap":
            w[i], w[i + 1] = w[i + 1], w[i]
        elif op == "drop" and w[i].lower() in vowels:
            del w[i]
        elif op == "double" and w[i].lower() not in vowels:
            w.insert(i, w[i])
        elif op == "vowel" and w[i].lower() in swaps:
            w[i] = swaps[w[i].lower()]
        new = "".join(w)
        if new.lower() != word.lower():
            return new
    return word + "h"  # fallback: still a one-letter difference


def split_ofac_name(raw):
    """'ABU MARZOOK, Mousa Mohammed' -> ('Mousa Mohammed', 'Abu Marzook')."""
    last, first = raw.split(",", 1)
    return first.strip().title(), last.strip().title()


# ----------------------------------------------------------------------------
# 1. Customers
# ----------------------------------------------------------------------------
def make_normal_customer():
    loc = random.choice(list(LOCALES))
    fk = FAKERS[loc]
    is_business = random.random() < 0.15
    return {
        "name": fk.company() if is_business else f"{fk.first_name()} {fk.last_name()}",
        "customer_type": "business" if is_business else "individual",
        "country": LOCALES[loc],
        "date_of_birth": None if is_business else
            fk.date_between(start_date=datetime(1950, 1, 1), end_date=datetime(2005, 12, 31)),
        "account_opened": fk.date_between(start_date=datetime(2018, 1, 1),
                                          end_date=datetime(2026, 6, 30)),
    }


def eligible_sdn_individuals(screening):
    """Real SDN individuals whose names are simple enough to plant cleanly."""
    ind = screening[(screening["type"] == "individual")].copy()
    ind = ind[ind["name"].str.count(",") == 1]
    ind = ind[ind["name_normalized"].str.fullmatch(r"[a-z ]+", na=False)]
    ind["n_tokens"] = ind["name_normalized"].str.split().str.len()
    return ind[ind["n_tokens"].between(2, 4)]


def plant_sanctions_hits(screening):
    """Customers whose names match (or nearly match) real SDN entries."""
    ind = eligible_sdn_individuals(screening)
    primary = ind[~ind["is_alias"]]
    aliases = ind[ind["is_alias"]]
    used = set()

    def pick(pool, need_middle=False):
        pool = pool[~pool["ent_num"].isin(used)]
        if need_middle:
            pool = pool[pool["name"].map(lambda n: len(split_ofac_name(n)[0].split()) >= 2)]
        row = pool.sample(1, random_state=random.randint(0, 10**6)).iloc[0]
        used.add(row["ent_num"])
        return row

    variants = ["exact", "exact", "misspelling", "misspelling", "misspelling",
                "reordered", "reordered", "missing_middle", "missing_middle", "alias"]
    planted = []
    for v in variants:
        if v == "alias":
            row = pick(aliases)
        else:
            row = pick(primary, need_middle=(v == "missing_middle"))
        first, last = split_ofac_name(row["name"])

        if v in ("exact", "alias"):
            name = f"{first} {last}"
        elif v == "misspelling":
            tokens = f"{first} {last}".split()
            longest = max(range(len(tokens)), key=lambda i: len(tokens[i]))
            tokens[longest] = misspell(tokens[longest])
            name = " ".join(tokens)
        elif v == "reordered":
            name = f"{last} {first}"           # surname first, no comma
        elif v == "missing_middle":
            name = f"{first.split()[0]} {last}"  # drop middle name(s)

        cust = make_normal_customer()
        cust.update(name=name, customer_type="individual")
        planted.append((cust, {
            "should_flag": True, "category": "sanctions", "reason": f"sanctions_{v}",
            "matched_ent_num": int(row["ent_num"]),
            "notes": f"Planted from SDN name: {row['name']}",
        }))
    return planted, used


def plant_false_positive_traps(screening, used, n=5):
    """Innocent customers who share a surname with someone on the list."""
    ind = eligible_sdn_individuals(screening)
    ind = ind[(~ind["is_alias"]) & (~ind["ent_num"].isin(used))]
    surnames = ind["name"].map(lambda n: split_ofac_name(n)[1])
    surnames = surnames[surnames.str.fullmatch(r"[A-Za-z]{4,}")]
    picks = surnames.sample(n, random_state=SEED)

    traps = []
    for ent_num, surname in zip(ind.loc[picks.index, "ent_num"], picks):
        cust = make_normal_customer()
        first = FAKERS[random.choice(list(LOCALES))].first_name()
        cust.update(name=f"{first} {surname}", customer_type="individual")
        traps.append((cust, {
            "should_flag": False, "category": "trap", "reason": "false_positive_trap",
            "matched_ent_num": None,
            "notes": f"Shares surname '{surname}' with SDN entry {ent_num}; innocent",
        }))
    return traps


# ----------------------------------------------------------------------------
# 2. Transactions
# ----------------------------------------------------------------------------
def txn(cid, ts, amount, direction, channel, country, counterparty=None):
    return {
        "customer_id": cid, "timestamp": ts, "amount": round(amount, 2),
        "direction": direction, "channel": channel,
        "counterparty_name": counterparty or COMPANY_FAKER.company(),
        "counterparty_country": country,
    }


def normal_activity(cust, n=None):
    """Everyday banking: card spend, bill payments, payroll, some cash."""
    mult = 4 if cust["customer_type"] == "business" else 1
    n = n if n is not None else random.randint(4, 14)
    out = []
    for _ in range(n):
        channel = random.choices(["card", "ach", "wire", "cash"], [50, 30, 12, 8])[0]
        mu = {"card": 4.0, "ach": 6.5, "wire": 7.6, "cash": 5.5}[channel]
        amount = random.lognormvariate(mu, 0.6) * mult
        if channel == "cash":
            amount = min(amount, 3000)           # keep normal cash well under $10k
        country = cust["country"] if random.random() < 0.8 else random.choice(LOW_RISK_COUNTRIES)
        direction = "in" if random.random() < 0.45 else "out"
        out.append(txn(cust["customer_id"], random_time(), not_round(amount),
                       direction, channel, country))
    return out


def pattern_structuring(cust):
    """Several cash deposits just under the $10,000 reporting threshold."""
    start = random_time(START, END - timedelta(days=7))
    return [txn(cust["customer_id"], start + timedelta(days=d, hours=random.randint(9, 17)),
                random.uniform(8000, 9950), "in", "cash", cust["country"], "Cash deposit")
            for d in sorted(random.sample(range(7), 4))]


def pattern_rapid_movement(cust):
    """Big wire comes in, ~95% leaves again within two days (pass-through)."""
    t0 = random_time(START, END - timedelta(days=3))
    amount_in = random.uniform(25000, 60000)
    out_total = amount_in * random.uniform(0.92, 0.98)
    split = random.uniform(0.4, 0.6)
    dest = random.choice(LOW_RISK_COUNTRIES)
    return [
        txn(cust["customer_id"], t0, amount_in, "in", "wire", random.choice(LOW_RISK_COUNTRIES)),
        txn(cust["customer_id"], t0 + timedelta(hours=random.randint(4, 20)),
            out_total * split, "out", "wire", dest),
        txn(cust["customer_id"], t0 + timedelta(hours=random.randint(21, 44)),
            out_total * (1 - split), "out", "wire", dest),
    ]


def pattern_high_risk_geo(cust):
    """Wires to or from high-risk / sanctioned jurisdictions."""
    return [txn(cust["customer_id"], random_time(), random.uniform(2000, 15000),
                random.choice(["in", "out"]), "wire", random.choice(HIGH_RISK_COUNTRIES))
            for _ in range(random.randint(2, 3))]


def pattern_velocity_spike(cust):
    """Quiet for months, then a sudden burst of activity in one week."""
    out = []
    week = START
    while week < END - timedelta(days=14):     # ~1 small payment a week
        out.append(txn(cust["customer_id"], week + timedelta(days=random.randint(0, 6)),
                       not_round(random.uniform(200, 800)), "out", "ach", cust["country"]))
        week += timedelta(days=7)
    burst_start = END - timedelta(days=10)
    for _ in range(random.randint(10, 12)):     # then a burst
        out.append(txn(cust["customer_id"], random_time(burst_start, burst_start + timedelta(days=6)),
                       not_round(random.uniform(1500, 4000)), "out",
                       random.choice(["ach", "wire"]), random.choice(LOW_RISK_COUNTRIES)))
    return out


def pattern_round_amounts(cust):
    """Repeated transfers of exactly $5,000 or $10,000."""
    return [txn(cust["customer_id"], random_time(), random.choice([5000, 10000]),
                "out", "wire", random.choice(LOW_RISK_COUNTRIES))
            for _ in range(random.randint(4, 6))]


PATTERNS = {
    "structuring": pattern_structuring,
    "rapid_movement": pattern_rapid_movement,
    "high_risk_geography": pattern_high_risk_geo,
    "velocity_spike": pattern_velocity_spike,
    "round_amounts": pattern_round_amounts,
}


# ----------------------------------------------------------------------------
# 3. Put it all together
# ----------------------------------------------------------------------------
def main():
    screening_path = DATA_DIR / "screening_list.csv"
    if not screening_path.exists():
        raise SystemExit("data/screening_list.csv not found. Run `python src/load_sdn.py` first.")
    screening = pd.read_csv(screening_path)

    # --- Build the customer list (with answer-key info attached) ---
    planted, used = plant_sanctions_hits(screening)
    traps = plant_false_positive_traps(screening, used)

    behaviour = []
    for pattern in PATTERNS:
        for _ in range(3):
            behaviour.append((make_normal_customer(), {
                "should_flag": True, "category": "behaviour", "reason": pattern,
                "matched_ent_num": None, "notes": f"Planted {pattern} pattern",
            }))

    n_normal = N_CUSTOMERS - len(planted) - len(traps) - len(behaviour)
    normal = [(make_normal_customer(), {
        "should_flag": False, "category": "normal", "reason": "normal",
        "matched_ent_num": None, "notes": "",
    }) for _ in range(n_normal)]

    everyone = planted + traps + behaviour + normal
    random.shuffle(everyone)                     # don't leave planted ones at the top
    for i, (cust, _) in enumerate(everyone, start=1):
        cust["customer_id"] = f"C{i:04d}"

    # --- Transactions: everyone gets normal activity, planted ones get extra ---
    transactions = []
    for cust, key in everyone:
        if key["reason"] == "velocity_spike":
            transactions += pattern_velocity_spike(cust)   # replaces normal activity
            continue
        transactions += normal_activity(cust)
        if key["reason"] in PATTERNS:
            transactions += PATTERNS[key["reason"]](cust)

    # --- Save ---
    customers = pd.DataFrame([c for c, _ in everyone])
    customers = customers[["customer_id", "name", "customer_type", "country",
                           "date_of_birth", "account_opened"]]

    answer_key = pd.DataFrame([{"customer_id": c["customer_id"], "name": c["name"], **k}
                               for c, k in everyone])
    answer_key["matched_ent_num"] = answer_key["matched_ent_num"].astype("Int64")

    txns = pd.DataFrame(transactions).sort_values("timestamp").reset_index(drop=True)
    txns.insert(0, "txn_id", [f"T{i:06d}" for i in range(1, len(txns) + 1)])

    customers.to_csv(DATA_DIR / "customers.csv", index=False)
    txns.to_csv(DATA_DIR / "transactions.csv", index=False)
    answer_key.to_csv(DATA_DIR / "answer_key.csv", index=False)

    # --- Summary ---
    print(f"Customers: {len(customers):,}  "
          f"({(customers.customer_type == 'business').sum()} businesses)")
    print(f"Transactions: {len(txns):,}  ({txns.timestamp.min():%b %d} – {txns.timestamp.max():%b %d, %Y})")
    print("\nAnswer key — what later stages should find:")
    print(answer_key.groupby(["category", "reason"]).size().to_string())
    print("\nPlanted sanctions hits (name in our bank vs. name on the OFAC list):")
    hits = answer_key[answer_key.category == "sanctions"]
    for _, r in hits.iterrows():
        print(f"  {r['reason']:<26} {r['name']:<35} <- {r['notes'].split(': ', 1)[1]}")
    print("\nSaved data/customers.csv, data/transactions.csv, data/answer_key.csv")


if __name__ == "__main__":
    main()
