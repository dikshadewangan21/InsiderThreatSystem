from pathlib import Path
import pandas as pd

ldap_dir = Path("data/raw/LDAP")

files = sorted(ldap_dir.glob("*.csv"))

dfs = []

for f in files:
    print("Reading", f.name)
    dfs.append(pd.read_csv(f))

ldap = pd.concat(dfs, ignore_index=True)

ldap.to_csv("data/raw/LDAP.csv", index=False)

print("Saved LDAP.csv")