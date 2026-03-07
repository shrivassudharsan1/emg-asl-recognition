# Quick Start: New Gesture Analysis Commands

## Two New Analysis Modes

### 1️⃣ Test Individual Files (`test_folder`)

**Analyze each trial of a gesture separately to see consistency:**

```bash
python run_umap.py test_folder <gesture-name>
```

**Examples:**
```bash
python run_umap.py test_folder spider-man
python run_umap.py test_folder peace
python run_umap.py test_folder closed-hand
python run_umap.py test_folder opened-hand
python run_umap.py test_folder hang-loose
```

**What you'll see:**
- 5 separate clusters (one per trial file)
- Each trial labeled as: ESP01, ESP02, ESP03, ESP04, ESP05
- Indicates whether the gesture is performed consistently
- Shows if any trials had poor signal quality

**Data:** ~5,000 samples (5 files × ~1000 samples each)

---

### 2️⃣ Compare Two Gestures (`compare_gesture`)

**Compare two complete gestures by combining all their files:**

```bash
python run_umap.py compare_gesture <gesture1> <gesture2>
```

**Examples:**
```bash
# Basic comparisons
python run_umap.py compare_gesture spider-man peace
python run_umap.py compare_gesture closed-hand opened-hand
python run_umap.py compare_gesture hang-loose peace

# More combinations
python run_umap.py compare_gesture spider-man closed-hand
python run_umap.py compare_gesture peace opened-hand
python run_umap.py compare_gesture closed-hand hang-loose
```

**What you'll see:**
- 2 large clusters (one per gesture)
- All 5 spider-man files grouped under "Spider-Man" label
- All 5 peace files grouped under "Peace" label
- Whether gestures are distinguishable by EMG patterns

**Data:** ~10,000 samples (10 files × ~1000 samples each)

---

## Quick Analysis Workflow

### Complete Gesture Analysis (Recommended Order)

```bash
# Step 1: Check individual gesture consistency
python run_umap.py test_folder spider-man
python run_umap.py test_folder peace

# Step 2: Test if they're separable
python run_umap.py compare_gesture spider-man peace

# Step 3: Build a gesture confusion matrix (test all pairs)
python run_umap.py compare_gesture spider-man closed-hand
python run_umap.py compare_gesture spider-man opened-hand
python run_umap.py compare_gesture spider-man hang-loose
python run_umap.py compare_gesture peace closed-hand
python run_umap.py compare_gesture peace opened-hand
python run_umap.py compare_gesture peace hang-loose
python run_umap.py compare_gesture closed-hand opened-hand
python run_umap.py compare_gesture closed-hand hang-loose
python run_umap.py compare_gesture opened-hand hang-loose
```

---

## Available Gestures

All gestures have **5 trials each**:

| Gesture | Command | Files | Labels |
|---------|---------|-------|--------|
| Spider-Man | `test_folder spider-man` | ESP01-05-0228.csv | ESP01, ESP02, ESP03, ESP04, ESP05 |
| Peace | `test_folder peace` | EPE01-05-0228.csv | EPE01, EPE02, EPE03, EPE04, EPE05 |
| Closed Hand | `test_folder closed-hand` | ECH01-05-0228.csv | ECH01, ECH02, ECH03, ECH04, ECH05 |
| Opened Hand | `test_folder opened-hand` | EOH01-05-0228.csv | EOH01, EOH02, EOH03, EOH04, EOH05 |
| Hang Loose | `test_folder hang-loose` | EHL01-05-0228.csv | EHL01, EHL02, EHL03, EHL04, EHL05 |

---

## Why We Combine Instead of Average

### Combining Files (What We Do) ✅

**Pros:**
- Preserves all data variability (~5000 samples per gesture)
- Shows real-world gesture recognition challenge
- Reveals which trials are problematic
- Better for ML training (more data = better models)
- Natural cluster formation

**Example:**
```
UMAP: Spider-Man (all 5 trials combined)
- 5000 EMG samples all labeled "Spider-Man"
- Natural clustering shows gesture signature
- Outliers visible = quality issues caught
```

### Averaging Files (What We Don't Do) ❌

**Cons:**
- Loses 75% of data (only 1000 averaged samples)
- Hides intra-gesture variation
- Masks calibration problems
- "Perfect" but unrealistic EMG pattern
- Reduces classification potential

**Example:**
```
Averaged: Spider-Man (5 trials averaged)
- 1000 averaged samples labeled "Spider-Man"
- No variation visible
- Perfect but artificial - real data isn't this clean
```

---

## Interpreting Results

### Good Gesture (test_folder)
```
UMAP Plot: Tight clusters
✅ All 5 trials cluster together
✅ Consistent EMG pattern
✅ Good signal quality
```

### Poor Gesture (test_folder)
```
UMAP Plot: Scattered points
❌ Files spread across space
❌ Inconsistent pattern
❌ Possible calibration issues
```

### Separable Gestures (compare_gesture)
```
UMAP Plot: Two distinct clusters
✅ Clear separation between Spider-Man and Peace
✅ Easy to classify
✅ Little overlap
```

### Confusing Gestures (compare_gesture)
```
UMAP Plot: Overlapping clusters
⚠️  Some Spider-Man samples look like Peace
⚠️  Difficult to classify
⚠️  May need better gesture distinction
```

---

## Data Information

### File Structure
```
/Users/Ricardo/Documents/GitHub/TritonNeuroTech/CSV-Files/
├── spider-man/      (5 files - Gesture: Spider-Man hand sign)
├── peace/           (5 files - Gesture: Peace hand sign)
├── closed-hand/     (5 files - Gesture: Closed fist)
├── opened-hand/     (5 files - Gesture: Open hand)
└── hang-loose/      (5 files - Gesture: Hang loose sign)
```

### Data Per File
- **Samples**: ~1000 EMG measurements
- **Channels**: 8 EMG electrodes
- **Format**: Tab-separated CSV
- **File size**: ~1-2 MB

### Total Dataset Size
- **5 gestures × 5 trials**: 25 files
- **Total samples**: ~25,000 EMG measurements
- **Gestures**: Human hand signs/positions
- **Source**: EMG forearm sensors

---

## Performance Tips

### Faster Analysis
1. **Smaller comparisons**: Use `test_folder` for single gesture
2. **Focus on pairs**: Compare 2 gestures at a time
3. **Sequential runs**: Do one analysis, then the next

### Better Visualizations
1. **Wait for output**: UMAP/PCA take 2-5 seconds
2. **Look for clusters**: Red/blue/green = different gestures
3. **Check scattering**: Tight = consistent, Scattered = variable
4. **Note overlaps**: Where clustering happens tells you about gesture similarity

---

## Error Messages & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| "Folder 'xyz' not found" | Typo in gesture name | Use: `spider-man` not `spiderman` |
| "No such file or directory" | Wrong working directory | Run from `/TritonNeuroTech/` |
| Command not found | Python not activated | Run: `source .venv/bin/activate` |
| UMAP overlay error | Old matplotlib | Update: `pip install --upgrade matplotlib` |

---

## Next Steps

See `GESTURE_ANALYSIS_GUIDE.md` for:
- Detailed methodology explanation
- Statistical analysis guidance
- Custom configuration examples
- Advanced UMAP tuning

See `UMAP_USAGE_GUIDE.md` for:
- Parameter customization
- Batch testing
- Data export examples
- Troubleshooting

---

**Q&A:**

**Q: Should I average the files?**
A: No! Combining gives you 5x more data and shows real variation. Averaging hides important patterns.

**Q: How long does analysis take?**
A: 2-10 seconds depending on data size. UMAP is the major component.

**Q: What if gestures overlap in the plot?**
A: That's information! It shows those gestures have similar EMG signatures and may be hard to distinguish.

**Q: Can I combine more than 2 gestures?**
A: Yes! Create a custom configuration in `gesture_configs.json` or edit `umap_test.py` directly.

---
