# EMG Gesture Analysis: Combining vs Averaging Data

## Overview

The `run_umap.py` tool now supports two new analysis modes:

1. **test_folder** - Analyze individual files from a gesture folder separately
2. **compare_gesture** - Compare two complete gestures by combining all their files

This guide explains the data analysis approach and when to use each method.

---

## New Commands

### Test Individual Files in a Folder

Analyze all files in a gesture folder, treating each file as a separate entity:

```bash
python run_umap.py test_folder closed-hand
python run_umap.py test_folder opened-hand
python run_umap.py test_folder hang-loose
python run_umap.py test_folder peace
python run_umap.py test_folder spider-man
```

**Output**: Each file (ECH01-0228.csv, ECH02-0228.csv, etc.) is treated as a separate label/cluster in the UMAP visualization, showing how individual trials cluster.

### Compare Two Complete Gestures

Combine all files from two gesture folders and compare them:

```bash
# Compare spider-man vs peace
python run_umap.py compare_gesture spider-man peace

# Compare closed-hand vs opened-hand
python run_umap.py compare_gesture closed-hand opened-hand

# Compare hang-loose vs peace
python run_umap.py compare_gesture hang-loose peace
```

**Output**: All 5 spider-man files + all 5 peace files are combined into two groups (labeled "Spider-Man" and "Peace"), showing if the gestures are distinguishable regardless of trial variation.

---

## Data Analysis Methodology

### Why We Combine Files Instead of Averaging

#### **UMAP/PCA Work on Individual Data Points**

Both UMAP and PCA operate at the sample level:
- **UMAP**: Non-linear dimensionality reduction that preserves local structure. It works best with many individual data points to identify natural clusters.
- **PCA**: Linear dimensionality reduction that finds principal components explaining variance. More data points improve component reliability.

Example with our EMG data:
- **File**: One CSV with ~1000 EMG samples
- **Combined approach**: 5 files × 1000 samples = 5000 data points
- **Averaged approach**: 5 files → 1 averaged file = 1000 data points (75% information loss)

#### **Benefits of Combining Files**

| Aspect | Combining | Averaging |
|--------|-----------|-----------|
| **Data Points** | 5000+ samples | ~1000 samples |
| **Variability Info** | Preserved | Lost |
| **Cluster Quality** | Better separation | Blurred boundaries |
| **Outlier Detection** | Visible | Hidden |
| **Gesture Consistency** | Observable | Unknown |

#### **What We Learn from Combining**

1. **Gesture Consistency**: Do all trials of "spider-man" cluster together?
2. **Intra-gesture Variation**: How much do samples vary within the same gesture?
3. **Inter-gesture Separation**: How well separated are different gestures?
4. **Natural Clusters**: UMAP finds true patterns without imposed averaging

#### **What We Lose with Averaging**

1. **Trial Variability**: Can't see if some trials are harder to recognize
2. **Statistical Power**: 75% fewer samples for analysis
3. **Outliers**: Bad EMG recordings are hidden
4. **Real-World Performance**: Classifiers receive individual samples, not averages

### When to Average (and when we don't)

**Average if:**
- You only care about the "ideal" EMG signature of each gesture
- You want to reduce computational load (rarely needed with UMAP)
- You're publishing summary figures for a paper

**Combine if** (recommended for analysis):
- You want to understand real-world gesture recognition performance
- You need to assess intra-gesture reliability
- You're developing classification algorithms
- You want to identify problematic trials

---

## Analysis Workflow

### Step 1: Explore Individual Files
```bash
# See how each trial of a gesture behaves
python run_umap.py test_folder spider-man
```
- Look for: Tight clusters (consistent gesture) or spread clusters (variable signal)

### Step 2: Compare Gestures
```bash
# See if gestures are separable
python run_umap.py compare_gesture spider-man peace
```
- Look for: Clear separation (good gestures for classification) or overlap (confusing gestures)

### Step 3: Multi-Gesture Comparison
```bash
# Add all gestures to preset configuration in gesture_configs.json
# Then run:
python run_umap.py all_gestures_combined
```
- Shows overall gesture space and inter-gesture relationships

---

## Expected Results

### Example 1: Clear Gesture Separation
```
UMAP Plot:
- Red cluster (spider-man): All 5 trials tightly grouped
- Blue cluster (peace): All 5 trials tightly grouped
- Clear separation between clusters

Interpretation: These gestures are distinct and easy to classify
```

### Example 2: Overlapping Gestures
```
UMAP Plot:
- Purple cluster (closed-hand): Spreading across the space
- Orange cluster (opened-hand): Overlaps with some closed-hand regions

Interpretation: These gestures share similar EMG patterns in some trials
(possibly neutral hand positions)
```

### Example 3: Individual File Variability
```
UMAP Plot (test_folder spider-man):
- ESP01: Cluster in one region
- ESP02: Cluster slightly shifted
- ESP03: Cluster far from others (poor signal/calibration?)
- ESP04: Similar to ESP01
- ESP05: Similar to ESP02

Interpretation: Most trials consistent, ESP03 might need recalibration
```

---

## Implementation Details

### test_folder Method

```python
def test_folder_individual(self, folder_name):
    """Test all files in a folder individually (each file as separate label)"""
    files = self._get_folder_files(folder_name)
    # Each file gets its own label (basename without extension)
    labels = [Path(f).stem for f in files]
    # Run UMAP with all files + all labels
    cmd = ['python', 'umap_test.py', '--files'] + files + ['--labels'] + labels
    subprocess.run(cmd, check=True)
```

**Files**: ESP01-0228.csv, ESP02-0228.csv, ESP03-0228.csv, ESP04-0228.csv, ESP05-0228.csv
**Labels**: ESP01, ESP02, ESP03, ESP04, ESP05

### compare_gesture Method

```python
def compare_gestures(self, gesture1, gesture2):
    """Compare two gestures by combining all files in each folder"""
    files1 = self._get_folder_files(gesture1)
    files2 = self._get_folder_files(gesture2)
    
    # All gesture1 files get label "Gesture1" (formatted)
    # All gesture2 files get label "Gesture2" (formatted)
    all_labels = [gesture1.title()] * len(files1) + [gesture2.title()] * len(files2)
```

**Files**: All files from both folders (10 total)
**Labels**: 5 "Spider-Man" + 5 "Peace"
**Effect**: UMAP groups by gesture, showing if they're separable

---

## Gesture Folder Structure

```
CSV-Files/
├── closed-hand/          (5 files: ECH01-05-0228.csv)
├── opened-hand/          (5 files: EOH01-05-0228.csv)
├── hang-loose/           (5 files: EHL01-05-0228.csv)
├── peace/                (5 files: EPE01-05-0228.csv)
└── spider-man/           (5 files: ESP01-05-0228.csv)
```

Each folder contains exactly 5 trials of the same gesture from different sessions/attempts.

---

## Quick Reference

| Command | Purpose | Data Points | Best For |
|---------|---------|-------------|----------|
| `test_folder peace` | See individual trial consistency | ~5000 | Understanding intra-gesture variation |
| `compare_gesture spider-man peace` | Compare two complete gestures | ~10000 | Assessing gesture separability |
| `all_gestures` | Compare all gestures together | ~25000 | Overall gesture space analysis |

---

## Performance Notes

### Sample Counts
- **Single file**: ~1000 samples
- **Folder (combine)**: ~5000 samples
- **Two folders**: ~10000 samples
- **All 5 folders**: ~25000 samples

### Computation Time
- **UMAP**: O(n log n) ~  2-5 seconds for 10k samples
- **PCA**: O(n) ~ <1 second for any size

UMAP scales well; computation time increases slowly with data size.

### Memory Usage
- **Per file**: ~10-20 MB (CSV in memory)
- **5 files combined**: ~50-100 MB
- **All data**: ~250-300 MB

---

## Recommendations

1. **Start with test_folder** to understand gesture quality
2. **Then compare_gesture** for pairs of interest
3. **Finally compare_gesture** between all pairs to build a confusion matrix
4. **Use presets** for production analysis

```bash
# Recommended analysis sequence
python run_umap.py test_folder spider-man
python run_umap.py test_folder peace
python run_umap.py compare_gesture spider-man peace
python run_umap.py compare_gesture spider-man closed-hand
python run_umap.py compare_gesture peace closed-hand
# ... and so on for all pairs
```

This gives you a complete understanding of:
- Individual gesture consistency (test_folder)
- Pairwise separability (compare_gesture for each pair)
- Overall gesture classification potential

---

## Troubleshooting

### "Error: Folder 'xyz' not found in CSV-Files/"
- Check spelling: `closed-hand` not `closedhand`
- Ensure you're in the correct directory
- Verify folder exists: `ls CSV-Files/closed-hand/`

### Gestures overlap in UMAP
- This is normal! Some gestures share EMG patterns
- Try comparing different gesture pairs
- Consider gesture selection for classification tasks

### One trial is far from others (test_folder)
- Possible calibration issue
- Check signal quality during that recording
- May need to recalibrate or rerecord

---

For more information on UMAP parameters or customization, see `UMAP_USAGE_GUIDE.md`.
