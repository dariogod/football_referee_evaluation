# PrTreid Output Evaluation Summary

## Overview
- **Total predictions evaluated**: 53,024
- **Number of SNGS directories**: 49
- **Classes found**: player, goalkeeper, referee, other, ball
- **Focus on main classes**: player, goalkeeper, referee

## Class Distribution

### True Labels:
- **Player**: 46,454 (87.6%)
- **Referee**: 4,492 (8.5%)  
- **Goalkeeper**: 1,946 (3.7%)
- **Other**: 132 (0.2%)
- **Ball**: 0 (0.0%)

### Predicted Labels:
- **Player**: 48,001 (90.5%)
- **Referee**: 3,428 (6.5%)
- **Goalkeeper**: 1,329 (2.5%)
- **Other**: 117 (0.2%)
- **Ball**: 149 (0.3%)

## Detailed Metrics for Main Classes

| Class      | TP    | FP   | FN   | TN    | Accuracy | Precision | Recall | F1     |
|------------|-------|------|------|-------|----------|-----------|--------|--------|
| **Player** | 46,093| 1,908| 361  | 4,662 | 0.9572   | 0.9603    | 0.9922 | 0.9760 |
| **Referee**| 3,301 | 127  | 1,191| 48,405| 0.9751   | 0.9630    | 0.7349 | 0.8336 |
| **Goalkeeper**| 1,069| 260  | 877  | 50,818| 0.9786   | 0.8044    | 0.5493 | 0.6528 |

## Key Performance Insights

### Player Classification (Best Performance)
- **Excellent recall** (99.22%): Very few players are missed
- **High precision** (96.03%): Low false positive rate
- **Outstanding F1-score** (97.60%): Best overall balance

### Referee Classification (Good Performance)
- **Very high precision** (96.30%): When predicted as referee, it's usually correct
- **Moderate recall** (73.49%): Some referees are misclassified as other roles
- **Good F1-score** (83.36%): Solid overall performance

### Goalkeeper Classification (Challenging)
- **Moderate precision** (80.44%): Some false positives from other classes
- **Lower recall** (54.93%): Many goalkeepers are misclassified
- **Lowest F1-score** (65.28%): Most challenging class to identify

## Confusion Matrix Analysis

```
                Predicted
              GK   Other  Player  Referee
True    GK   1069    10     746      92
       Other   4    78      49       1  
      Player 218     2   46093      34
     Referee  38    27    1113    3301
```

### Common Misclassifications:
1. **Goalkeepers → Players** (746 cases): Largest source of goalkeeper errors
2. **Referees → Players** (1,113 cases): Main source of referee false negatives
3. **Players → Referees** (34 cases): Minimal confusion in this direction

## Overall Performance
- **Overall Accuracy**: 95.32%
- **Macro-averaged Precision**: 94.26% (excluding ball class)
- **Macro-averaged Recall**: 75.88% (excluding ball class)
- **Macro-averaged F1**: 83.75% (excluding ball class)

## Recommendations
1. **Goalkeeper Detection**: Focus on improving goalkeeper recall - consider additional training data or feature engineering
2. **Referee-Player Confusion**: Address the referee→player misclassification pattern
3. **Class Imbalance**: The large player class (87.6%) may be affecting minority class performance

## Files Generated
- `confusion_matrix.py`: Complete evaluation script
- `confusion_matrix.png`: Visual confusion matrix
- `evaluation_summary.md`: This summary report 