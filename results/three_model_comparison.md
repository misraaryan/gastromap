# Three-model comparison (3 seeds, mean ± SD)
Source HMU = TUM vs NOR; GasHisSDB = 160 px Abnormal vs Normal. Cross = the other dataset, no retraining.

## In-domain AUC
| model    | gashis-random   | hmu-grouped   | hmu-random    |
|:---------|:----------------|:--------------|:--------------|
| ResNet50 | nan             | 0.891 ± 0.010 | 0.916 ± 0.001 |
| Phikon   | 1.000 ± 0.000   | 0.878 ± 0.013 | 0.923 ± 0.004 |
| UNI      | 1.000 ± 0.000   | 0.888 ± 0.020 | 0.924 ± 0.001 |

## Cross-dataset AUC
| model    | gashis-random   | hmu-grouped   | hmu-random    |
|:---------|:----------------|:--------------|:--------------|
| ResNet50 | nan             | 0.692 ± 0.035 | 0.702 ± 0.029 |
| Phikon   | 0.768 ± 0.008   | 0.862 ± 0.034 | 0.811 ± 0.019 |
| UNI      | 0.777 ± 0.020   | 0.920 ± 0.004 | 0.869 ± 0.016 |

## Cross-dataset balanced accuracy
| model    | gashis-random   | hmu-grouped   | hmu-random    |
|:---------|:----------------|:--------------|:--------------|
| ResNet50 | nan             | 0.541 ± 0.047 | 0.522 ± 0.008 |
| Phikon   | 0.703 ± 0.005   | 0.691 ± 0.067 | 0.677 ± 0.039 |
| UNI      | 0.678 ± 0.025   | 0.654 ± 0.075 | 0.669 ± 0.051 |

## Gaps
| model    | leakage gap (random−grouped, in-domain AUC)   | gen. gap hmu-grouped   | gen. gap hmu-random   | gen. gap gashis-random   |
|:---------|:----------------------------------------------|:-----------------------|:----------------------|:-------------------------|
| ResNet50 | 0.025 ± 0.010                                 | 0.199 ± 0.025          | 0.214 ± 0.029         | –                        |
| Phikon   | 0.046 ± 0.015                                 | 0.016 ± 0.043          | 0.113 ± 0.023         | 0.232 ± 0.008            |
| UNI      | 0.036 ± 0.019                                 | -0.032 ± 0.020         | 0.055 ± 0.016         | 0.223 ± 0.020            |

## Label-free adaptation: cross-dataset balanced accuracy
|                             | none          | target_std    | coral         |
|:----------------------------|:--------------|:--------------|:--------------|
| ('Phikon', 'gashis-random') | 0.703 ± 0.005 | 0.700 ± 0.004 | 0.651 ± 0.003 |
| ('Phikon', 'hmu-grouped')   | 0.691 ± 0.067 | 0.780 ± 0.041 | 0.712 ± 0.017 |
| ('Phikon', 'hmu-random')    | 0.677 ± 0.039 | 0.746 ± 0.017 | 0.703 ± 0.007 |
| ('UNI', 'gashis-random')    | 0.678 ± 0.025 | 0.709 ± 0.012 | 0.652 ± 0.003 |
| ('UNI', 'hmu-grouped')      | 0.654 ± 0.075 | 0.841 ± 0.006 | 0.729 ± 0.005 |
| ('UNI', 'hmu-random')       | 0.669 ± 0.051 | 0.797 ± 0.009 | 0.716 ± 0.006 |

## Adaptation + referral: balanced accuracy on 80% most confident cross-dataset patches
|                             | none          | target_std    | coral         |
|:----------------------------|:--------------|:--------------|:--------------|
| ('Phikon', 'gashis-random') | 0.746 ± 0.004 | 0.743 ± 0.003 | 0.684 ± 0.006 |
| ('Phikon', 'hmu-grouped')   | 0.682 ± 0.103 | 0.841 ± 0.036 | 0.757 ± 0.017 |
| ('Phikon', 'hmu-random')    | 0.688 ± 0.090 | 0.799 ± 0.018 | 0.744 ± 0.011 |
| ('UNI', 'gashis-random')    | 0.720 ± 0.020 | 0.754 ± 0.009 | 0.684 ± 0.008 |
| ('UNI', 'hmu-grouped')      | 0.588 ± 0.075 | 0.899 ± 0.011 | 0.773 ± 0.006 |
| ('UNI', 'hmu-random')       | 0.635 ± 0.078 | 0.860 ± 0.022 | 0.760 ± 0.008 |

## Uncertainty error-detection AUROC (cross-dataset)
|                             | none          | target_std    |
|:----------------------------|:--------------|:--------------|
| ('Phikon', 'gashis-random') | 0.657 ± 0.006 | 0.659 ± 0.009 |
| ('Phikon', 'hmu-grouped')   | 0.747 ± 0.033 | 0.760 ± 0.022 |
| ('Phikon', 'hmu-random')    | 0.728 ± 0.009 | 0.700 ± 0.028 |
| ('ResNet50', 'hmu-grouped') | 0.656 ± 0.006 | nan           |
| ('ResNet50', 'hmu-random')  | 0.677 ± 0.026 | nan           |
| ('UNI', 'gashis-random')    | 0.663 ± 0.008 | 0.666 ± 0.005 |
| ('UNI', 'hmu-grouped')      | 0.810 ± 0.035 | 0.798 ± 0.009 |
| ('UNI', 'hmu-random')       | 0.747 ± 0.036 | 0.752 ± 0.017 |
