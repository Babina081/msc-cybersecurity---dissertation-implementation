# Privacy-Preserving Activity Recognition in Smart Homes

Federated Learning (FL) with Differential Privacy (DP) for recognising daily activities from smart-home sensor data (CASAS dataset).

Each home is treated as a separate client. Homes train a shared model on their own data and send only model weights to a server, which averages them (FedAvg). Raw sensor data never leaves the home. Differential privacy (DP-SGD via Opacus) adds a formal (ε, δ) privacy guarantee on top, and the project measures how much accuracy is lost at different privacy levels.

## Project structure

```
dissertation-implementation/
├── check_setup.py            # Step 1  - verify the environment
├── data_explore.py           # Step 2a - explore the raw CASAS data
├── preprocess.py             # Step 2b - build windowed features per home
├── fl_utils.py               # Shared helpers for Step 3 (model, loading, metrics)
├── baseline_centralized.py   # Step 3, Stage 1 - centralised baseline
├── fl_train.py               # Step 3, Stage 2 - federated learning (FedAvg)
├── fl_dp_experiment.py       # Step 3, Stage 3 - federated learning + DP sweep
├── plot_results.py           # Step 4 - privacy-utility trade-off figure
├── requirements.txt
├── results.csv               # Output of Stage 3
├── figures/
│   └── tradeoff_curve.png    # Output of Step 4
└── data/
    ├── hh101.csv             # Raw CASAS home files
    ├── hh102.csv
    ├── hh103.csv
    └── processed/            # Output of Step 2b
        ├── hh101.npz
        ├── hh102.npz
        ├── hh103.npz
        └── meta.json
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The raw data files (`data/hh*.csv`) come from the CASAS smart-home datasets (Washington State University). Place them in `data/` before running the pipeline.

## Pipeline: step-by-step file processing

Run the scripts in the order below from the project root. Each step reads the output of the previous one.

```
data/hh*.csv ──► preprocess.py ──► data/processed/*.npz + meta.json
                                              │
                 ┌────────────────────────────┼─────────────────────────┐
                 ▼                            ▼                         ▼
      baseline_centralized.py           fl_train.py            fl_dp_experiment.py
                                                                        │
                                                                        ▼
                                                                  results.csv
                                                                        │
                                                                        ▼
                                                                 plot_results.py
                                                                        │
                                                                        ▼
                                                        figures/tradeoff_curve.png
```

### Step 1: Check the environment

```bash
python check_setup.py
```

Confirms that `torch`, `opacus`, `scikit-learn`, `pandas`, `numpy` and `matplotlib` are installed, runs a small PyTorch computation, and creates an Opacus `PrivacyEngine`. Nothing is read or written.

### Step 2a: Explore the raw data

```bash
python data_explore.py
```

- **Input:** `data/hh101.csv`
- **Output:** printed summary only (no files saved)

Each raw line has the form `date, time, sensor, value [, label]`, where the optional label marks the start or end of an activity (e.g. `Sleep="begin"`). The script reports the number of events and the date range, groups sensors by type (motion, light, temperature, door, battery), fills every begin→end span so each event carries an activity, and shows how much of the data is labelled versus unlabelled ("Other").

### Step 2b: Preprocess into features

```bash
python preprocess.py
```

- **Input:** every `data/hh*.csv`
- **Output:** `data/processed/<home>.npz` (one per home) and `data/processed/meta.json`

For each home, the script:

1. **Reads the raw events** line by line and parses date and time into a timestamp.
2. **Fills activity spans** so every event between a `begin` and `end` marker is labelled with that activity.
3. **Maps activities into 12 classes.** The fine-grained CASAS activities are grouped into Watch_TV, Hygiene, Cook, Sleep, Toilet, Dress, Relax, Wash_Dishes, Eat, Take_Meds, Leave_Home and Enter_Home. Unmapped activities are dropped. The class order is fixed so every home shares the same label numbering, which federated learning requires.
4. **Slides a window over the events.** Windows are 30 events long and move 15 events at a time (50% overlap). A window is kept only if at least 50% of it belongs to one activity, which becomes its label.
5. **Extracts 15 features per window:** the fraction of events in each of the 8 rooms, the number of distinct rooms, room transitions, time of day (encoded as sine and cosine), window duration, events per minute, and the fraction of `ON` events.
6. **Saves the results.** Each home's feature matrix `X` and labels `y` are stored in its own `.npz` file, so each file represents one federated client. The room list, feature names, class list and window settings are written to `meta.json` so later scripts use identical settings.

### Step 3: Model training and experiments

All three stages use the same neural network (an MLP: features → 64 → 32 → 12 classes) and are evaluated with **macro-F1** (every class counts equally) and **weighted-F1** (common classes count more).

`fl_utils.py` holds the shared code. It is imported by the scripts below and is not run directly. It loads each home's `.npz` file, splits each home 80/20 into train and test sets, scales features using statistics from the pooled training data only (to avoid leakage), and provides class weights so rare activities are not ignored.

#### Stage 1: Centralised baseline

```bash
python baseline_centralized.py
```

- **Input:** `data/processed/*.npz`, `meta.json`
- **Output:** printed F1 scores and a per-class report

Pools every home's data in one place and trains a single model for 60 epochs. This is the no-privacy best case that the other stages are compared against.

#### Stage 2: Federated learning (no privacy)

```bash
python fl_train.py
```

- **Input:** `data/processed/*.npz`, `meta.json`
- **Output:** printed comparison of centralised and federated F1

Runs 40 FedAvg rounds. In each round, every home trains a copy of the global model locally for 2 epochs, and the server averages the weights in proportion to each home's data size. A centralised model is trained on the same split, so the difference between them is the cost of federation.

#### Stage 3: Federated learning with differential privacy

```bash
python fl_dp_experiment.py
```

- **Input:** `data/processed/*.npz`, `meta.json`
- **Output:** `results.csv`

Each home trains locally with DP-SGD through Opacus, which clips each sample's gradient (max norm 1.0) and adds calibrated Gaussian noise. The script tests privacy budgets ε ∈ {0.1, 0.5, 1.0, 5.0} with δ = 1e-4, plus a no-DP condition that keeps gradient clipping but switches the noise off, so the cost of the noise itself is isolated. Each condition runs for 15 rounds across 5 random seeds. The achieved ε is verified with an RDP accountant, and macro-F1, weighted-F1, achieved ε and runtime are saved for every run.

### Step 4: Plot the results

```bash
python plot_results.py
```

- **Input:** `results.csv`
- **Output:** `figures/tradeoff_curve.png` and a printed summary table

Plots macro-F1 and weighted-F1 against ε on a log scale, with error bars across seeds, the no-DP ceiling as dashed lines, and the 0.75 hypothesis target as a dotted line.

## Results

Mean scores over 5 seeds, from `results.csv`:

| Condition | Macro-F1 | Weighted-F1 |
|-----------|---------:|------------:|
| no-DP     | 0.617    | 0.743       |
| ε = 5.0   | 0.590    | 0.725       |
| ε = 1.0   | 0.478    | 0.635       |
| ε = 0.5   | 0.269    | 0.472       |
| ε = 0.1   | 0.269    | 0.444       |

For reference, Stage 2 reported a centralised macro-F1 of 0.674 and a federated (no-DP) macro-F1 of 0.639.

At ε = 5.0 the model stays close to the no-privacy ceiling. Accuracy drops noticeably at ε = 1.0 and sharply at ε ≤ 0.5, showing the trade-off between stronger privacy and model utility.