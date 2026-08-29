"""
data_explore.py  -  Step 2a: understand the raw CASAS data before building features.

Run:  python data_explore.py

It reads data/hh101.csv, works out its structure, and prints a summary.
Nothing is changed or saved - this is just to look at what we have.
"""
import pandas as pd
from pathlib import Path
from collections import Counter

DATA_FILE = Path("data/hh101.csv")

# ---------------------------------------------------------------------------
# 1. Read the file.
#    Rows have 4 OR 5 fields (the 5th is an optional activity label), so we
#    can't assume a fixed shape - we read line by line to stay safe.
#    Each line looks like:  date, time, sensor, value [, label]
# ---------------------------------------------------------------------------
rows = []
with open(DATA_FILE, "r") as f:
    for line in f:
        parts = line.rstrip("\n").split(",")
        if len(parts) < 4:
            continue  # skip blank / malformed lines
        date, time, sensor, value = parts[0], parts[1], parts[2], parts[3]
        label = parts[4] if len(parts) >= 5 else ""
        rows.append((date, time, sensor, value, label))

df = pd.DataFrame(rows, columns=["date", "time", "sensor", "value", "label"])
print(f"Total sensor events : {len(df):,}")
print(f"Date range          : {df['date'].iloc[0]}  ->  {df['date'].iloc[-1]}")

# ---------------------------------------------------------------------------
# 2. What sensors are in this home, and what type is each?
#    The leading letters of a sensor name tell you its kind.
# ---------------------------------------------------------------------------
def sensor_type(name):
    if name.startswith("MA"):   return "Motion-area (MA)"
    if name.startswith("M"):    return "Motion (M)"
    if name.startswith("LS"):   return "Light-sensor (LS)"
    if name.startswith("L"):    return "Light (L)"
    if name.startswith("T"):    return "Temperature (T)"
    if name.startswith("D"):    return "Door (D)"
    if name.startswith("BA"):   return "Battery (BA)"
    if "Door" in name:          return "Door (named)"
    return "Other"

df["stype"] = df["sensor"].apply(sensor_type)

print(f"\nUnique sensors      : {df['sensor'].nunique()}")
print("Events by sensor type:")
for t, c in df["stype"].value_counts().items():
    print(f"   {t:18s} {c:,}")

print("\nTop 15 busiest sensors:")
for name, count in df["sensor"].value_counts().head(15).items():
    print(f"   {name:14s} {count:,}")

# ---------------------------------------------------------------------------
# 3. What activities are labelled? Fill each begin->end span.
#    A line 'X="begin"' starts activity X; 'X="end"' finishes it.
#    Everything in between belongs to X. Unlabelled stretches are "Other".
# ---------------------------------------------------------------------------
def read_marker(lbl):
    if not lbl:
        return (None, None)
    name = lbl.split("=")[0].strip()
    low = lbl.lower()
    if "begin" in low: return (name, "begin")
    if "end" in low:   return (name, "end")
    return (name, "point")

current = "Other"
activity_col = []
begin_counts = Counter()
for lbl in df["label"]:
    name, boundary = read_marker(lbl)
    if boundary == "begin":
        current = name
        begin_counts[name] += 1
        activity_col.append(current)
    elif boundary == "end":
        activity_col.append(current)   # the 'end' line still belongs to X
        current = "Other"
    else:
        activity_col.append(current)
df["activity"] = activity_col

print(f"\nDistinct activities labelled: {len(begin_counts)}")
print("Activities (times each one begins):")
for name, c in begin_counts.most_common():
    print(f"   {name:24s} {c}")

# ---------------------------------------------------------------------------
# 4. How much of the data actually sits inside a labelled activity?
# ---------------------------------------------------------------------------
labelled = (df["activity"] != "Other").sum()
pct = 100 * labelled / len(df)
print(f"\nEvents inside a labelled activity : {labelled:,} ({pct:.1f}%)")
print(f"Events with no activity ('Other') : {len(df)-labelled:,} ({100-pct:.1f}%)")

print("\nEvents per activity (after filling spans):")
for name, c in df["activity"].value_counts().items():
    print(f"   {name:24s} {c:,}")
