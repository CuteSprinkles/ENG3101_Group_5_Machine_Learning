# 💰 Adult Income Prediction — Logistic Regression vs XGBoost

> ENG3101 Machine Learning Project — Group 5, Singapore Institute of Technology

A Jupyter notebook that predicts whether a person earns **more than USD $50,000 a year** using the UCI Adult (1994 US Census) dataset. It compares **Logistic Regression** and **XGBoost**, each with three feature-selection approaches and tunes all of them with `GridSearchCV` and stratified 5-fold cross-validation. `(ง •̀_•́)ง`

![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.1-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.4.1-189FDD)
![pandas](https://img.shields.io/badge/pandas-3.0.3-150458?logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-UCI_Adult-6E40C9)

> [!CAUTION]
> ⚠️ **The PDF export (`Group_5_Final_ML_Code.pdf`) may have formatting issues.** Long code lines are wrapped (marked with `␣`) and some formulas, tables or wide outputs may display incorrectly or be partly missing. Please use **`Group_5_Final_ML_Code.ipynb`** as the complete and correct version.

---

## 🌟 Project Overview

**Research question:** Can selecting fewer input features keep or improve prediction performance and does XGBoost perform better than Logistic Regression?

| Target value | Income class | Role |
|:---:|:---:|:---:|
| 0 | ≤50K | Negative class |
| 1 | >50K | Positive class |

Six experiments are compared using the same five cross-validation folds:

| No. | Model | Feature selection |
|:---:|:---|:---|
| 1 | Logistic Regression | None (control) |
| 2 | Logistic Regression | ANOVA F-statistic |
| 3 | Logistic Regression | Mutual Information |
| 4 | XGBoost | None (control) |
| 5 | XGBoost | ANOVA F-statistic |
| 6 | XGBoost | Mutual Information |

The experiment with the highest **mean validation F1-score (>50K class)** is locked in, and only that model is tested on the separate `adult.test` file.

> ⚠️ For educational use only. The data is historical US census data and should not be used for real decisions such as hiring or credit approval.

---

## ✨ What the Notebook Does

1. 📥 Reads `adult.data` and `adult.test` (downloads them from UCI if missing) and saves `adult_train.csv` and `adult_test.csv`
2. 📊 Explores the **training data only** — class balance, distributions, outliers, missing values
3. 🧹 Drops `sex`, `race` and `fnlwgt`, puts missing categories into an **`Unknown`** category, and one-hot encodes the categorical columns
4. 🔧 Runs all 6 experiments through a leak-free `Pipeline` inside `GridSearchCV`
5. 🏆 Picks the winner by mean validation F1 and tests it once on `adult.test`
6. 🔍 Shows the confusion matrix, ROC curve, retained features and feature importance
7. ⚖️ Runs a descriptive TPR/FPR audit by sex and race (these columns are **not** used as inputs)

**Pipeline order:**
`Impute + one-hot encode` → `Select features (ANOVA / MI / none)` → `Remove constant columns` → `Scale (Logistic Regression only)` → `Model`

---

## 🏆 Results (submitted run)

| Experiment | Mean CV F1 | CV F1 SD | Best settings |
|:---|:---:|:---:|:---|
| **XGBoost + ANOVA** ✅ | **0.7140** | 0.0096 | top 75%, `max_depth=6`, `n_estimators=150` |
| XGBoost + Mutual Information | 0.7138 | 0.0095 | top 75%, `max_depth=6`, `n_estimators=150` |
| XGBoost + None | 0.7138 | 0.0097 | `max_depth=6`, `n_estimators=300` |
| Logistic Regression + None | 0.6599 | 0.0087 | `C=10` |
| Logistic Regression + MI | 0.6591 | 0.0076 | top 75%, `C=10` |
| Logistic Regression + ANOVA | 0.6591 | 0.0080 | top 75%, `C=100` |

**Final test on `adult.test` (XGBoost + ANOVA, threshold 0.50):**

| F1 | Precision | Recall | Specificity | ROC-AUC | Accuracy |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.7111 | 0.7745 | 0.6573 | 0.9408 | 0.9277 | 0.8738 |

**Takeaways**
- XGBoost beat Logistic Regression by about 0.054 mean CV F1.
- ANOVA and MI kept **75 of 100** encoded columns with almost no change in F1. The differences between the three XGBoost experiments are much smaller than their fold-to-fold SD, so feature selection mainly helped by **reducing inputs**, not by clearly raising F1.

---

## 🧰 Requirements

Tested with:

| Package | Version |
|:---|:---|
| Python | 3.14.6 |
| pandas | 3.0.3 |
| numpy | 2.4.6 |
| scikit-learn | **1.9.1** |
| XGBoost | **3.4.1** |

Plus `matplotlib`, `seaborn` and JupyterLab. Section 1.1 installs everything:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn==1.9.1 xgboost==3.4.1
```

If the cell installs or updates anything, use **Restart Kernel & Run All**.

---

## 📂 Files

```
.
├── Group_5_Final_ML_Code.ipynb   # Main notebook
├── Group_5_Final_ML_Code.pdf     # PDF export with all outputs
├── README.md
├── adult.data / adult.test / adult.names   # UCI files (auto-downloaded if missing)
└── adult_train.csv / adult_test.csv        # Created by the notebook (Section 1.4)
```

---

## 🚀 How to Run

1. Put the notebook in a folder (optionally with `adult.data`, `adult.test` and `adult.names`).
2. Open it in JupyterLab.
3. Click **Run → Restart Kernel and Run All Cells**.

The slow part is **Section 4.4**: 56 candidate settings × 5 folds = **280 CV fits + 6 refits = 286 model fits**. Depending on the computer, this can take several minutes or longer. If it is too slow or appears stuck, see the next section.

---

## ⚙️ Tweaking `n_jobs` (speed vs stability)

Two separate `n_jobs` settings control how many CPU cores are used:

| Where | Code | Submitted value | What it controls |
|:---|:---|:---:|:---|
| Section 4.1 | `XGBClassifier(..., n_jobs = 2)` | `2` | CPU **threads per XGBoost model** |
| Section 4.4 | `GridSearchCV(..., n_jobs = -2)` | `-2` | How many **CV fits run at the same time** (parallel workers) |

### 🧮 What the values mean

| `n_jobs` value | Meaning |
|:---:|:---|
| `1` | No parallelism — one at a time (slowest, most stable) |
| `2`, `4`, … | Use exactly that many workers/threads |
| `-1` | Use **all** CPU cores |
| `-2` | Use all CPU cores **except one** (at least one worker), so the computer stays responsive |

### ⚠️ The two settings multiply

During the XGBoost experiments, the total load is roughly:

```
total CPU threads ≈ (GridSearchCV workers) × (XGBoost n_jobs)
```

Example on an 8-core laptop with the submitted settings: `-2` → 7 workers × 2 threads = **14 threads on 8 cores**. That is oversubscribed, and it can make the run slower, the laptop hot, or Jupyter look frozen.

**Rule of thumb:** keep `workers × XGBoost threads` close to your number of CPU cores.

Check your core count in a notebook cell:

```python
import os
print(os.cpu_count())
```

### 🎛️ Recommended presets

| Situation | `GridSearchCV(n_jobs=…)` | `XGBClassifier(n_jobs=…)` |
|:---|:---:|:---:|
| As submitted (most computers) | `-2` | `2` |
| Many cores (8+), avoid oversubscription | `-2` | `1` |
| Weak laptop (2–4 cores) or running hot | `2` | `1` |
| Cell stuck / crashes / need clearer errors | `1` | `-1` |
| Sharing the computer with other heavy apps | `2` | `1` |

The **`1` / `-1` preset** is the safest fallback. GridSearchCV runs one fit at a time, which makes errors easier to read and avoids worker processes hanging in Jupyter, while XGBoost still uses all cores. The Logistic Regression experiments will be slower here, because the `saga` solver runs single-threaded.

### 🔄 After changing `n_jobs`

1. Edit the value in **Section 4.1** and/or **Section 4.4**.
2. **Restart the kernel and run all cells.** `xgboost_model` is defined in Section 4.1, so re-running only Section 4.4 will not pick up a new XGBoost value.
3. Compare run times using the `Search seconds` column the notebook already records:

```python
display(cv_results[["Experiment", "Search seconds"]].round(1))
```

> 💡 `n_jobs` only changes **speed**, not the method. With `RANDOM_STATE = 48` and the same folds, the scores should stay the same. XGBoost may show tiny last-decimal differences when its thread count changes, which is normal for floating-point maths.

### 🛟 If the cell still hangs

- Interrupt the kernel (■), set `GridSearchCV(n_jobs = 1)`, then restart and run all.
- Close other heavy apps (browsers with many tabs, games, other notebooks).
- On Windows or macOS, plug in the charger. Battery-saver modes can throttle the CPU a lot.
- Use `print(..., flush = True)` progress messages (already in Section 4.4) to see which experiment is running.

---

## 🧪 Other Settings

| Setting | Location | Default | Notes |
|:---|:---|:---:|:---|
| `RANDOM_STATE` | Section 1.2 | `48` | Fixed seed for folds and models |
| `N_FOLDS` | Section 1.2 | `5` | Number of stratified CV folds |
| `THRESHOLD` | Section 1.2 | `0.50` | Probability cutoff for class 1 (>50K) |
| `logistic_grid` | Section 4.2 | `C = 0.1, 1, 10, 100` | Logistic Regression grid |
| `xgboost_grid` | Section 4.2 | `max_depth = 6, 8`; `n_estimators = 150, 300` | XGBoost grid |
| `percentile` | Section 4.4 | `25, 50, 75` | % of encoded features kept by ANOVA/MI |

Adding grid values increases run time **multiplicatively**. For example, a third `max_depth` value adds 50% more XGBoost fits.

---

## ⚠️ Limitations

- Small hyperparameter grid; the winner is the best **among the settings tested**.
- The same folds chose both the hyperparameters and the winning experiment, so CV scores may be slightly optimistic.
- No SMOTE or class weights were used, by design, so that the comparison stays focused on models and feature selection.
- `education` and `education-num` overlap, so their importance may be split between them.
- 23 test rows are identical to training rows, and 5 rows repeat within the test set (no unique IDs to check).
- Removing `sex` and `race` does **not** guarantee fairness. Features such as `relationship` can act as proxies.

---

## 👥 Team
Proudly done by **Group 5 members**: <br>
Royce · Naufal · Hui Shin · Yan Jun · Zhen Xiang

Dataset: B. Becker and R. Kohavi, "Adult," *UCI Machine Learning Repository*, 1996, doi: [10.24432/C5XW20](https://archive.ics.uci.edu/dataset/2/adult).
