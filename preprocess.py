"""
preprocess.py  -  Step 2b: turn raw CASAS events into model-ready features.

For every home file  data/hh*.csv  this script:
  1. reads the raw events (date, time, room, value [, label]),
  2. fills each begin->end activity span so every event has an activity,
  3. maps the 34 fine-grained activities into your 12 chosen classes,
  4. slides a fixed-size window over the events and turns each window into
     one feature row + one label (the majority activity in that window),
  5. saves the result to  data/processed/<home>.npz  (one file per home =
     one client for the Federated Learning stage).

Run:  python preprocess.py
"""
# python libraries
import json #saves metadata
import numpy as np #numerical arrays
import pandas as pd #handles csv data
from pathlib import Path #handles file/folder paths
from collections import Counter #counts activities

# look for raw data in data/
RAW_DIR = Path("data")
# store processed data in data/processed
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# program looks at 30 sensor events at a time (window)
WINDOW = 30      # number of events in each window
STEP   = 15      # how far the window slides each time (15 = 50% overlap)
# atleast 50% of the window must be one activity
PURITY = 0.5     # a window is kept only if >=50% of it is one activity
# e.g cook cook cook cook ...

# --- Your 12-class grouping: raw activity  ->  consolidated class -----------
CLASS_MAP = {
    "Watch_TV": "Watch_TV",
    "Personal_Hygiene": "Hygiene", "Groom": "Hygiene", "Bathe": "Hygiene",
    "Cook_Breakfast": "Cook", "Cook_Lunch": "Cook", "Cook_Dinner": "Cook", "Cook": "Cook",
    "Sleep": "Sleep", "Go_To_Sleep": "Sleep", "Wake_Up": "Sleep", "Sleep_Out_Of_Bed": "Sleep",
    "Toilet": "Toilet", "Bed_Toilet_Transition": "Toilet",
    "Dress": "Dress",
    "Relax": "Relax", "Read": "Relax", "Phone": "Relax",
    "Wash_Dishes": "Wash_Dishes", "Wash_Breakfast_Dishes": "Wash_Dishes",
    "Wash_Lunch_Dishes": "Wash_Dishes", "Wash_Dinner_Dishes": "Wash_Dishes",
    "Eat_Breakfast": "Eat", "Eat_Lunch": "Eat", "Eat_Dinner": "Eat", "Eat": "Eat", "Drink": "Eat",
    "Morning_Meds": "Take_Meds", "Evening_Meds": "Take_Meds",
    "Leave_Home": "Leave_Home", "Step_Out": "Leave_Home",
    "Enter_Home": "Enter_Home",
}

# Fixed order => the integer for each class is identical across every home.
# (Federated Learning needs all clients to share the same label space.)
CLASS_LIST = ["Watch_TV", "Hygiene", "Cook", "Sleep", "Toilet", "Dress",
              "Relax", "Wash_Dishes", "Eat", "Take_Meds", "Leave_Home", "Enter_Home"]
# converts activites into numbers
CLASS_TO_INT = {c: i for i, c in enumerate(CLASS_LIST)}

# function processes one home file (hh1.csv, hh2.csv, etc)
def load_home(path):
    """Read one home file into a DataFrame with a datetime, room, value and
    a consolidated class (or NaN for unlabelled/dropped events)."""
    rows = []
    # opens the csv file and reads line by line
    with open(path, "r") as f:
        for line in f:
            # 2026-01-01,07:10,Bedroom,ON,Sleep=begin 
            # splits the line into date, time, room, value and label
            # 2026-01-01 07:10 Bedroom ON Sleep=begin
            p = line.rstrip("\n").split(",")
            if len(p) < 4:
                continue
            rows.append((p[0], p[1], p[2], p[3], p[4] if len(p) >= 5 else ""))
    df = pd.DataFrame(rows, columns=["date", "time", "room", "value", "label"])
    df["dt"] = pd.to_datetime(df["date"] + " " + df["time"], errors="coerce")
    df = df.dropna(subset=["dt"]).reset_index(drop=True)

    # fill begin->end spans so every event carries its activity
    current, acts = "Other", []
    for lbl in df["label"]:
        if not lbl:
            acts.append(current); continue
        name, low = lbl.split("=")[0].strip(), lbl.lower()
        if "begin" in low:
            current = name; acts.append(current)
        elif "end" in low:
            acts.append(current); current = "Other"
        else:
            acts.append(current)
    df["activity"] = acts
    df["cls"] = df["activity"].map(CLASS_MAP)   # unmapped -> NaN (dropped later)
    return df

def make_windows(df, rooms):
    """Slide a window over the events; return (X features, y labels)."""
    room_idx = {r: i for i, r in enumerate(rooms)}
    rooms_arr = df["room"].to_numpy()
    vals_arr  = df["value"].to_numpy()
    cls_arr   = df["cls"].to_numpy()
    dt_arr    = df["dt"].to_numpy()

    X, y = [], []
    for s in range(0, len(df) - WINDOW + 1, STEP):
        e = s + WINDOW
        w_rooms, w_vals, w_cls = rooms_arr[s:e], vals_arr[s:e], cls_arr[s:e]
        w_dt = df["dt"].iloc[s:e]

        # label = majority consolidated class, if it is pure enough
        labels = [c for c in w_cls if isinstance(c, str)]
        if not labels:
            continue
        top, cnt = Counter(labels).most_common(1)[0]
        if cnt < PURITY * WINDOW:
            continue

        # ---- features ----
        feat = []
        rc = np.zeros(len(rooms))                       # room occupancy fractions
        for r in w_rooms:
            if r in room_idx:
                rc[room_idx[r]] += 1
        feat.extend((rc / WINDOW).tolist())
        feat.append(len(set(w_rooms)))                  # how many rooms used
        feat.append(sum(a != b for a, b in zip(w_rooms[:-1], w_rooms[1:])))  # transitions
        hour = w_dt.iloc[0].hour + w_dt.iloc[0].minute / 60.0
        feat.append(np.sin(2 * np.pi * hour / 24))      # time-of-day (cyclic)
        feat.append(np.cos(2 * np.pi * hour / 24))
        dur = (w_dt.iloc[-1] - w_dt.iloc[0]).total_seconds()
        feat.append(dur)                                # window duration (s)
        feat.append(WINDOW / (dur / 60) if dur > 0 else 0.0)   # events per minute
        feat.append(float(np.mean(w_vals == "ON")))     # fraction of 'ON' events

        X.append(feat)
        y.append(CLASS_TO_INT[top])
    return np.array(X, dtype=float), np.array(y, dtype=int)


def feature_names(rooms):
    return ([f"room_{r}" for r in rooms] +
            ["n_rooms", "transitions", "hour_sin", "hour_cos",
             "duration_s", "events_per_min", "on_fraction"])


def main():
    files = sorted(RAW_DIR.glob("hh*.csv"))
    if not files:
        print("No data/hh*.csv files found. Put your home files in data/ first.")
        return

    print(f"Found {len(files)} home file(s): {[f.name for f in files]}\n")
    homes = {f.stem: load_home(f) for f in files}

    # shared room vocabulary across ALL homes -> identical feature columns
    rooms = sorted({r for df in homes.values() for r in df["room"].unique()})
    feats = feature_names(rooms)
    print(f"Shared rooms ({len(rooms)}): {rooms}")
    print(f"Feature vector length: {len(feats)}\n")

    for name, df in homes.items():
        X, y = make_windows(df, rooms)
        np.savez(OUT_DIR / f"{name}.npz", X=X, y=y)
        dist = Counter(CLASS_LIST[i] for i in y)
        print(f"{name}: {len(X):>5} windows | classes present: {len(dist)}/12")
        for c in CLASS_LIST:
            if dist.get(c):
                print(f"      {c:12s} {dist[c]:>5}")

    # save the shared setup so later scripts use the same features/labels
    meta = {"rooms": rooms, "feature_names": feats,
            "class_list": CLASS_LIST, "window": WINDOW, "step": STEP}
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved processed clients + meta.json to {OUT_DIR}/")


if __name__ == "__main__":
    main()
