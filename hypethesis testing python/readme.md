
## Overview
This project applies statistical hypothesis testing to analyze student performance data.

---

## Dataset
Students Performance dataset containing:
- math score
- reading score
- writing score
- test preparation course
- gender

---

## Methods Used

### 1. One-sample t-test
- Compares average math score against a benchmark (65)

### 2. Independent t-test
- Compares math scores between students with/without test preparation

### 3. Paired t-test
- Compares reading vs writing scores (same students)

### 4. Proportion z-test
- Tests gender differences in high performance (math > 80)

---

## Statistical Significance
- Alpha level: 0.05
- Decisions based on p-values

---

## Tech Stack
- Python
- SciPy
- Statsmodels
- Pandas

---

## How to run

```bash
python stats_analysis.py