# M4a Results: HDC Transfer Across Tasks

## Executive Summary

**Key Finding:** HDC transfer has a fundamental limitation for sentence-pair tasks.

| Task | Input Type | Teacher | Student | Ratio |
|------|------------|---------|---------|-------|
| SST-2 | Single sentence | 88% | 77% | **87.5%** ✅ |
| AG News | Single sentence | 93.2% | 86.6% | **92.9%** ✅ |
| MNLI (weak) | Sentence pairs | 66.6% | 45.0% | 67.6% ⚠️ |
| MNLI (strong) | Sentence pairs | 91.4% | 43.0% | **47.0%** ❌ |

## Critical Discovery

### The Bottleneck is NOT in Teacher Quality

**Hypothesis tested:** "Smarter teacher = better transfer efficiency"

**Result:** REJECTED

```
Weak Teacher (66.6%) → Student 45.0% → Ratio 67.6%
Strong Teacher (91.4%) → Student 43.0% → Ratio 47.0%
                                              ↓
                               WORSE with better teacher!
```

### The Bottleneck IS in HDC Encoding for Pairs

Current encoding method:
```python
premise_emb = encode("A man is sleeping")      # 384d
hypothesis_emb = encode("A person is resting") # 384d
combined = concatenate([premise_emb, hypothesis_emb])  # 768d
hdc_vec = project_to_hdc(combined)             # 4096d
```

**Problem:** Simple concatenation loses relational information!

- The meaning of "entailment" is not in either sentence alone
- It's in the RELATIONSHIP between them
- Concatenation treats them as independent features

## What Works vs What Doesn't

### ✅ Single Sentence Tasks (HDC Works Well)
- Sentiment analysis (SST-2): 87.5% efficiency
- Topic classification (AG News): 92.9% efficiency
- Binary and multi-class
- Short and long texts (IMDB)

### ❌ Sentence Pair Tasks (HDC Bottleneck)
- Natural Language Inference (MNLI): ~45% ceiling
- Teacher quality doesn't help
- Encoding method is the limiting factor

## Implications for Resonance Protocol

1. **Current HDC is sufficient for:**
   - Single-document classification
   - Sentiment analysis
   - Topic routing
   - Content filtering

2. **Needs new approach for:**
   - Semantic similarity
   - Question-answering
   - Entailment/contradiction detection
   - Any task requiring cross-document reasoning

## Next Steps: Investigating the Bottleneck

Need to explore alternative encoding methods for sentence pairs:

1. **Difference vectors:** `hdc(premise) - hdc(hypothesis)`
2. **Element-wise operations:** `hdc(premise) * hdc(hypothesis)`
3. **Cross-attention inspired:** Learned combination
4. **Separate projections:** Different HDC spaces for premise/hypothesis

## Files

### Notebooks
- `M4a1_IMDB_Teacher.ipynb` / `M4a1_IMDB_Student.ipynb`
- `M4a2_AGNews_Teacher.ipynb` / `M4a2_AGNews_Student.ipynb`
- `M4a3_MNLI_Teacher.ipynb` / `M4a3_MNLI_Student.ipynb`
- `M4a3b_MNLI_StrongTeacher.ipynb` / `M4a3b_MNLI_StrongStudent.ipynb`

### Results
- `m4a_imdb_*.json` - IMDB results (too easy, 100%)
- `m4a_agnews_*.json` - AG News results (86.6%)
- `m4a_mnli_*.json` - MNLI weak teacher (45%)
- `m4a_mnli_strong_*.json` - MNLI strong teacher (43%)

## Conclusion

We found a clear limitation boundary:
- **Within boundary:** HDC achieves 85-93% of teacher performance
- **Outside boundary:** HDC hits ~45% ceiling regardless of teacher

This is valuable because it tells us exactly what to fix.
