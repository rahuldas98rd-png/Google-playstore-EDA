# Google Play Store Dataset

## Overview

This dataset contains information scraped from the Google Play Store, covering over **10,000 Android applications** across 34 categories. It is intended for exploratory data analysis, feature engineering, and building insights into the Android app ecosystem.

---

## File Information

| Property | Value |
|---|---|
| Filename | `googleplaystore.csv` |
| Format | CSV (Comma-Separated Values) |
| Total Records | 10,841 rows |
| Total Features | 14 columns (13 after dropping index) |
| File Size | ~1.27 MB |

---

## Feature Descriptions

| Column | Data Type (Raw) | Description |
|---|---|---|
| `Unnamed: 0` | int | Auto-generated row index — drop before analysis |
| `App` | string | Name of the application |
| `Category` | string | Primary category the app belongs to (e.g., `FAMILY`, `GAME`, `TOOLS`) |
| `Rating` | float | Average user rating on the Play Store (scale: 1.0 – 5.0) |
| `Reviews` | string* | Total number of user reviews — stored as string, needs casting to int |
| `Size` | string* | App size in mixed formats: `19M`, `3.1M`, `899k`, or `Varies with device` — needs normalization |
| `Installs` | string* | Install count bracket (e.g., `1,000,000+`) — stored as string, needs cleaning |
| `Type` | string | Whether the app is `Free` or `Paid` |
| `Price` | string* | Price of the app (e.g., `0`, `$4.99`) — stored as string, needs `$` removed and casting to float |
| `Content Rating` | string | Target audience group: `Everyone`, `Teen`, `Mature 17+`, `Everyone 10+`, `Adults only 18+`, `Unrated` |
| `Genres` | string | More specific genre classification (can differ from `Category`) |
| `Last Updated` | string* | Date the app was last updated — stored as string (e.g., `7-Jan-18`), needs parsing to datetime |
| `Current Ver` | string | Current version of the application |
| `Android Ver` | string | Minimum Android version required to run the app (e.g., `4.1 and up`) |

> \* These columns require data type conversion and cleaning before analysis.

---

## Data Quality Notes

### Missing Values

| Column | Missing Count | % Missing |
|---|---|---|
| `Rating` | 1,474 | 13.6% |
| `Type` | 1 | ~0% |
| `Content Rating` | 1 | ~0% |
| `Current Ver` | 8 | ~0% |
| `Android Ver` | 3 | ~0% |

`Rating` has the most significant missingness and should be handled via imputation or exclusion depending on the analysis goal.

### Duplicates

- **1,181 duplicate app entries** exist (based on `App` name).
- After removing duplicates, the dataset reduces to **9,659 unique apps**.
- Duplicates should be dropped to avoid skewing aggregation-based analyses.

### Data Type Issues

Several columns that appear numeric are stored as strings in raw form:

- **`Reviews`** — mostly numeric strings; one corrupted row (`3.0M` instead of an integer) must be removed.
- **`Size`** — mixed units (`M` for megabytes, `k` for kilobytes, `Varies with device`). Recommend converting all to a uniform scale (e.g., kilobytes) and replacing `Varies with device` with `NaN`.
- **`Installs`** — contains commas and `+` characters (e.g., `1,000,000+`). Strip these and cast to integer.
- **`Price`** — contains `$` prefix for paid apps. Strip and cast to float.
- **`Last Updated`** — non-standard date format (`%d-%b-%y`). Parse with `pandas.to_datetime()`.

---

## Value Distributions (Key Columns)

### Category (Top 10)
| Category | Count |
|---|---|
| FAMILY | 1,972 |
| GAME | 1,144 |
| TOOLS | 843 |
| MEDICAL | 463 |
| BUSINESS | 460 |
| PRODUCTIVITY | 424 |
| PERSONALIZATION | 392 |
| COMMUNICATION | 387 |
| SPORTS | 384 |
| LIFESTYLE | 382 |

### Type
| Type | Count |
|---|---|
| Free | 10,039 |
| Paid | 800 |

### Content Rating
| Rating | Count |
|---|---|
| Everyone | 8,714 |
| Teen | 1,208 |
| Mature 17+ | 499 |
| Everyone 10+ | 414 |
| Adults only 18+ | 3 |
| Unrated | 2 |

### App Rating (Numeric)
| Statistic | Value |
|---|---|
| Mean | 4.19 |
| Median | 4.30 |
| Std Dev | 0.54 |
| Min | 1.00 |
| Max | 19.00* |

> \* A rating of 19.0 is clearly erroneous — this is a known data quality issue in this dataset and should be treated as an outlier.

---

## Recommended Preprocessing Steps

1. **Drop** the `Unnamed: 0` index column.
2. **Remove** the one corrupted row where `Reviews = "3.0M"`.
3. **Convert** `Reviews` to `int64`.
4. **Normalize** `Size`: convert `M`/`k` values to a common unit, set `Varies with device` → `NaN`.
5. **Clean** `Installs`: strip commas and `+`, convert to `int64`.
6. **Clean** `Price`: strip `$`, convert to `float64`.
7. **Parse** `Last Updated` using `pd.to_datetime(df['Last Updated'], format='%d-%b-%y')`.
8. **Drop duplicate** apps (keep first or last occurrence based on use case).
9. **Handle missing** `Rating` values — consider median imputation or removal.
10. **Clip or remove** any `Rating` values above 5.0 as invalid.

---

## Source

This dataset is a widely used open-source dataset originally scraped from the Google Play Store and made available for educational and analytical purposes.

> **Note:** The data reflects a historical snapshot and does not represent current Play Store listings.
