# TinyHAR

This guide shows how to **run your TinyHAR LOSO training script** (`tinyhar_train.py`) **inside** the upstream repository: **`https://github.com/mariusbock/tal_for_har`**.

- Clone the upstream repo, put the WEAR dataset into `data/`.
- Run `data_creation.py` once.
- Copy your training script into the repo and run it.

---

## 1) Clone the upstream repository

```bash
git clone https://github.com/mariusbock/tal_for_har.git
cd tal_for_har
```

Create a virtual environment (recommended) and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# If present, prefer the repo's requirements file
pip install -r requirements.txt
# If you need extras:
# pip install torch numpy pandas scikit-learn
```

---

## 2) Add the WEAR dataset under `data/`

Place the dataset in the repository’s `data` directory. A typical structure after download/unzip looks like:

```
tal_for_har/
  data/
    wear/
      raw/
        inertial/
          sbj_0
          sbj_1
          ... 
```

---

## 3) Run `data_creation.py` once for WEAR

The upstream repo expects you to **run `data_creation.py` once** to generate auxiliary files (e.g., chunked data/annotations). For **WEAR** (i.e., “all other datasets” in their docs), use:

- `create_annotations=True`
- `chunked=True`
- `chunk_size=[1, 5, 30, 60]`
- `window_size=50`
- `window_overlap=50`

> In the original repo, these are **script parameters**—they may be set inside the file (not as CLI flags). Open `data_creation.py`, verify the settings for WEAR, then run:

```bash
python data_creation.py
```

This step only has to be done once per dataset/layout.

---

## 4) Copy your training script into the repo

Copy your **`tinyhar_train.py`** into the repository. Two common layouts work:
```
tal_for_har/
  tinyhar_train.py
  inertial_baseline/
  data/
  ...
```
Run with:
```bash
python tinyhar_train.py
```

---

## 5) Expected inputs for your script

- **Raw CSVs for LOSO**: The script expects subject CSVs under
  ```
  data/wear/raw/inertial/sbj_{ID}.csv
  ```
  where the **last column is the activity label (string)** and all preceding columns are numeric features (sensor channels).

- **Labels**: Your script trains **without** the `'null'` class. It maps 18 activity classes (jogging variants, stretches, push-ups, sit-ups, burpees, lunges variants, bench-dips). Make sure label strings in your CSV exactly match your script’s dictionary.

> Note: The upstream `data_creation.py` step is primarily for generating chunked/annotation files used by their experiments. Your training script reads the **raw** per-subject CSVs path above. If your WEAR CSVs aren’t already at `data/wear/raw/inertial/`, place them there (or edit `RAW_ROOT` in your script).

---

## 6) Configure & run your LOSO training

Open `tinyhar_train.py` and adjust constants if needed:
- `SUBJECTS   = list(range(6))`, use `list(range(24))` for `sbj_0..sbj_23`
- `WINDOW_SIZE = 50`, `OVERLAP = 0.5`
- `BATCH_SIZE = 100`, `EPOCHS = 30`, `LR = 1e-4`
- `DEVICE = "cuda" if torch.cuda.is_available() else "cpu"`


**Outputs** are written to:
```
data/wear/metrics/
  loso_sbj_{ID}.json   # per-fold best metrics and confusion matrix
  loso_summary.json    # cross-subject mean summary
```


---

## Notes

- Keep the upstream repo’s **license** and **citation** rules.
- If you later want to run the upstream experiments, use their `main.py` and config files under `configs/`.
