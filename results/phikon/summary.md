# Phikon (frozen) + linear probe: first results (3 seeds, mean ± SD)

## Contribution 1: leakage (HMU, same images, only the split differs)

| setting     | auc           | bal_acc       | ece           |
|:------------|:--------------|:--------------|:--------------|
| hmu-grouped | 0.878 ± 0.013 | 0.791 ± 0.010 | 0.048 ± 0.012 |
| hmu-random  | 0.923 ± 0.004 | 0.843 ± 0.003 | 0.042 ± 0.013 |

Leakage gap (random − grouped AUC), per seed: [0.033, 0.062, 0.042]; mean 0.046

## Contribution 2: cross-dataset generalisation

|                                    | auc           | bal_acc       | ece           |
|:-----------------------------------|:--------------|:--------------|:--------------|
| ('gashis-random', 'cross-dataset') | 0.768 ± 0.008 | 0.703 ± 0.005 | 0.210 ± 0.029 |
| ('gashis-random', 'in-domain')     | 1.000 ± 0.000 | 0.995 ± 0.001 | 0.005 ± 0.002 |
| ('hmu-grouped', 'cross-dataset')   | 0.862 ± 0.034 | 0.691 ± 0.067 | 0.072 ± 0.042 |
| ('hmu-grouped', 'in-domain')       | 0.878 ± 0.013 | 0.791 ± 0.010 | 0.048 ± 0.012 |
| ('hmu-random', 'cross-dataset')    | 0.811 ± 0.019 | 0.677 ± 0.039 | 0.082 ± 0.061 |
| ('hmu-random', 'in-domain')        | 0.923 ± 0.004 | 0.843 ± 0.003 | 0.042 ± 0.013 |

- gashis-random: generalisation gap (in-domain − cross AUC) = 0.232 ± 0.008
- hmu-grouped: generalisation gap (in-domain − cross AUC) = 0.016 ± 0.043
- hmu-random: generalisation gap (in-domain − cross AUC) = 0.113 ± 0.023

## Contribution 3: uncertainty on the UNSEEN dataset (entropy of probe ensemble)

| setting       | err_detect_auc   | balacc@100    | balacc@90     | balacc@80     | balacc@70     |
|:--------------|:-----------------|:--------------|:--------------|:--------------|:--------------|
| gashis-random | 0.657 ± 0.006    | 0.708 ± 0.004 | 0.727 ± 0.003 | 0.746 ± 0.004 | 0.768 ± 0.004 |
| hmu-grouped   | 0.747 ± 0.033    | 0.688 ± 0.064 | 0.685 ± 0.084 | 0.682 ± 0.103 | 0.680 ± 0.117 |
| hmu-random    | 0.728 ± 0.009    | 0.672 ± 0.061 | 0.681 ± 0.075 | 0.688 ± 0.090 | 0.694 ± 0.107 |

err_detect_auc = how well uncertainty separates wrong from right predictions (0.5 = useless, 1 = perfect).
balacc@80 = balanced accuracy on the 80% most confident patches (the other 20% referred to a pathologist).