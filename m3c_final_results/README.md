# M3c HDC Transfer Optimization — Final Results

## Key Insight: Metric Correction

**Original metric (Transfer Efficiency)** was unstable — it depended on random initialization of the student model. The "before" accuracy varied between runs (46-53%), making the improvement calculation unreliable.

**Corrected metric (Student Final Accuracy)** is stable — regardless of initialization, the student converges to approximately the same final accuracy. This is the true measure of HDC transfer quality.

## Results Summary (by Student Accuracy)

| Experiment | Configuration | Student Accuracy | vs Baseline |
|------------|---------------|------------------|-------------|
| **M3c″** | Baseline (500 ex, hard) | 73.4% | — |
| **M3c⁴** | Soft labels | 73.2% | -0.2% |
| **M3c⁵** | 2000 examples | 76.8-77.0% | +3.4-3.6% |
| **M3c⁶** | 10000d HDC | 74.0% | +0.6% |
| **M3c⁷** | Soft + 2000 ex | 77.6% | +4.2% |
| **M3c⁸** | 5000 examples | 77.4% | +4.0% |
| **M3c⁹** | **Soft + 5000 ex** | **78.0%** | **+4.6%** 🏆 |

## Key Findings

### What Works ✅

1. **More training examples**
   - 500 → 2000: +3.5% accuracy
   - 2000 → 5000: +0.5% accuracy (diminishing returns)

2. **Soft labels + More data (combined)**
   - Best result: 78.0% accuracy
   - Small but consistent improvement over hard labels

### What Doesn't Help ❌

1. **Larger HDC dimension (10000d)**
   - No improvement, more parameters to train
   - 4096d is sufficient

2. **Soft labels alone (without more data)**
   - No improvement over baseline

## Practical Recommendations

### Simple & Effective: M3c⁵
```
Configuration:
- 2000 examples
- Hard labels (CrossEntropyLoss)
- 4096d HDC

Result: ~77% accuracy
Complexity: Low
```

### Maximum Performance: M3c⁹
```
Configuration:
- 5000 examples  
- Soft labels (KLDivLoss)
- 4096d HDC

Result: 78% accuracy
Complexity: Medium
```

**Recommendation:** Use M3c⁵ for most cases. The +1% from M3c⁹ rarely justifies the added complexity.

## Teacher-Student Gap

```
Teacher (DistilBERT): 87.4%
Student (MLP 2.2M):   78.0%
Gap:                   9.4%

Student achieves 89% of teacher performance
while being 30× smaller and requiring only HDC vectors!
```

## Directory Structure

```
m3c_optimization/
├── README.md
├── baseline/           # M3c″ (73.4%)
├── soft_labels/        # M3c⁴ (73.2%)
├── more_examples_2000/ # M3c⁵ (77.0%) ← Simple & Effective
├── larger_hdc/         # M3c⁶ (74.0%)
├── combined_2000/      # M3c⁷ (77.6%)
├── more_examples_5000/ # M3c⁸ (77.4%)
└── all_combined_5000/  # M3c⁹ (78.0%) ← Best
```

## Lessons Learned

1. **Use stable metrics**
   - Transfer efficiency depends on random init
   - Final accuracy is reproducible

2. **Simple improvements work**
   - More data is the most reliable improvement
   - Complex combinations don't always help

3. **Diminishing returns exist**
   - 500 → 2000 examples: big gain
   - 2000 → 5000 examples: small gain

4. **HDC dimension has a sweet spot**
   - 4096d works well
   - 10000d doesn't help without more data

## Technical Details

- **Teacher:** DistilBERT-base-uncased (66M params)
- **Student:** 3-layer MLP (2.2M params)
- **HDC Encoder:** Random projection 384d → 4096d, ternary quantization
- **Task:** SST-2 sentiment classification
- **Test set:** 500 examples from validation split
