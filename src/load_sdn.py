"""
Stage 1: Load and clean the OFAC SDN list.

Reads SDN.CSV (main list) and ALT.CSV (aliases) from the data/ folder,
cleans them, and produces one table of every name to screen against:
data/screening_list.csv
"""

import re
import unicodedata
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")

# The OFAC CSVs have no header row, so we supply the column names ourselves.
SDN_COLS = [
    "ent_num", "name", "type", "program", "title", "call_sign",
    "vess_type", "tonnage", "grt", "vess_flag", "vess_owner", "remarks",
]
ALT_COLS = ["ent_num", "alt_num", "alt_type", "alt_name", "alt_remarks"]


def read_ofac_csv(path, columns):
    """Read an OFAC CSV, trying UTF-8 first and falling back to Latin-1."""
    for encoding in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(path, header=None, names=columns,
                             encoding=encoding, dtype=str)
            break
        except UnicodeDecodeError:
            continue
    # Strip stray spaces around every text value FIRST. OFAC often writes
    # blanks as "-0- " (with a trailing space), which wouldn't match "-0-".
    df = df.apply(lambda col: col.str.strip())
    # OFAC writes empty values as "-0-". Turn them into real blanks.
    df = df.replace("-0-", pd.NA)
    # Drop junk rows (e.g. an end-of-file marker) where the ID isn't a number.
    df = df[pd.to_numeric(df["ent_num"], errors="coerce").notna()].copy()
    df["ent_num"] = df["ent_num"].astype(int)
    return df


def normalize_name(name, is_individual):
    """Turn 'SMITH, John A.' into 'john a smith' so names compare fairly."""
    if pd.isna(name):
        return pd.NA
    # Remove accents: 'José' -> 'Jose'
    name = unicodedata.normalize("NFKD", name)
    name = "".join(ch for ch in name if not unicodedata.combining(ch))
    name = name.lower()
    # Individuals are written 'LAST, First'. Flip them to 'first last'.
    # (Only for individuals: company names like 'ACME CO., LTD.' contain
    # commas too, and flipping those would scramble them.)
    if is_individual and "," in name:
        last, first = name.split(",", 1)
        name = f"{first} {last}"
    # Drop apostrophes entirely ("Sa'id" -> "said"), then turn other
    # punctuation into spaces ("ABU-MARZUQ" -> "abu marzuq").
    name = re.sub(r"['’`]", "", name)
    name = re.sub(r"[^\w\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def parse_identifiers(remarks):
    """Pull date(s) of birth and nationality out of OFAC's free-text remarks.

    Remarks look like: 'DOB 09 Feb 1951; alt. DOB 1952; POB Gaza; nationality
    Iran; Passport 1234 (Iran)'. A person can have several DOBs, a year only,
    'circa 1960', or a range '1955 to 1958'. We keep the raw text for humans
    and a list of every possible birth year for code to compare against.
    """
    if not isinstance(remarks, str):
        return pd.NA, pd.NA, pd.NA
    dob_texts = [d.strip() for d in re.findall(r"DOB\s+([^;]+)", remarks)]
    years = set()
    for d in dob_texts:
        found = [int(y) for y in re.findall(r"\b(1[89]\d\d|20\d\d)\b", d)]
        if " to " in d and len(found) == 2:           # a range: '1955 to 1958'
            years.update(range(min(found), max(found) + 1))
        else:
            years.update(found)
    nationalities = [n.strip() for n in re.findall(r"nationality\s+([^;.]+)", remarks, re.I)]
    return (" / ".join(dob_texts) or pd.NA,
            "|".join(str(y) for y in sorted(years)) or pd.NA,
            " / ".join(dict.fromkeys(nationalities)) or pd.NA)


def main():
    sdn = read_ofac_csv(DATA_DIR / "SDN.CSV", SDN_COLS)
    alt = read_ofac_csv(DATA_DIR / "ALT.CSV", ALT_COLS)

    # A blank type means the entry is a company/organization.
    sdn["type"] = sdn["type"].fillna("entity")

    # Secondary identifiers (DOB, nationality) from the remarks column.
    # Analysts use these to tell two people with similar names apart.
    sdn[["dob", "dob_years", "nationality"]] = pd.DataFrame(
        [parse_identifiers(r) for r in sdn["remarks"]], index=sdn.index)
    ids = ["dob", "dob_years", "nationality"]

    # --- Primary names ---
    primary = sdn[["ent_num", "name", "type", "program", *ids]].copy()
    primary["is_alias"] = False

    # --- Aliases: attach type/program/identifiers from the main list via ent_num ---
    aliases = alt.merge(sdn[["ent_num", "type", "program", *ids]],
                        on="ent_num", how="left")
    aliases = aliases.rename(columns={"alt_name": "name"})
    aliases = aliases[["ent_num", "name", "type", "program", "alt_type", *ids]]
    aliases["is_alias"] = True

    # --- Combine into one screening list ---
    screening = pd.concat([primary, aliases], ignore_index=True)
    screening = screening.dropna(subset=["name"])
    screening["name_normalized"] = [
        normalize_name(n, t == "individual")
        for n, t in zip(screening["name"], screening["type"])
    ]
    screening = screening.drop_duplicates(subset=["ent_num", "name_normalized"])

    out = DATA_DIR / "screening_list.csv"
    screening.to_csv(out, index=False)

    # --- Summary: proof that it worked ---
    print(f"Main list entries: {len(sdn):,}")
    print(sdn["type"].value_counts().to_string())
    print(f"\nAliases: {len(alt):,}")
    print(f"Total names to screen against: {len(screening):,}")
    print("\nTop 5 sanctions programs:")
    print(sdn["program"].value_counts().head(5).to_string())
    ind = sdn[sdn["type"] == "individual"]
    print(f"\nIndividuals with a date of birth listed: "
          f"{ind['dob'].notna().sum():,} of {len(ind):,}")
    print("\nExample normalized names:")
    print(screening[["name", "name_normalized"]].head(5).to_string(index=False))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
