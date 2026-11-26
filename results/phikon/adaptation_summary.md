# Label-free adaptation at the new hospital (Phikon probe, cross-dataset only, 3 seeds)

## Balanced accuracy (threshold 0.5)
| setting       | none          | target_std    | coral         |
|:--------------|:--------------|:--------------|:--------------|
| gashis-random | 0.703 ± 0.005 | 0.700 ± 0.004 | 0.651 ± 0.003 |
| hmu-grouped   | 0.691 ± 0.067 | 0.780 ± 0.041 | 0.712 ± 0.017 |
| hmu-random    | 0.677 ± 0.039 | 0.746 ± 0.017 | 0.703 ± 0.007 |

## AUC
| setting       | none          | target_std    | coral         |
|:--------------|:--------------|:--------------|:--------------|
| gashis-random | 0.768 ± 0.008 | 0.765 ± 0.009 | 0.715 ± 0.001 |
| hmu-grouped   | 0.862 ± 0.034 | 0.869 ± 0.040 | 0.797 ± 0.021 |
| hmu-random    | 0.811 ± 0.019 | 0.824 ± 0.022 | 0.787 ± 0.013 |

## Calibration error (ECE, lower is better)
| setting       | none          | target_std    | coral         |
|:--------------|:--------------|:--------------|:--------------|
| gashis-random | 0.210 ± 0.029 | 0.205 ± 0.031 | 0.274 ± 0.015 |
| hmu-grouped   | 0.072 ± 0.042 | 0.067 ± 0.019 | 0.058 ± 0.050 |
| hmu-random    | 0.082 ± 0.061 | 0.071 ± 0.029 | 0.083 ± 0.044 |

## Balanced accuracy on the 80% most confident patches
| setting       | none          | target_std    | coral         |
|:--------------|:--------------|:--------------|:--------------|
| gashis-random | 0.746 ± 0.004 | 0.743 ± 0.003 | 0.684 ± 0.006 |
| hmu-grouped   | 0.682 ± 0.103 | 0.841 ± 0.036 | 0.757 ± 0.017 |
| hmu-random    | 0.688 ± 0.090 | 0.799 ± 0.018 | 0.744 ± 0.011 |

## Uncertainty error-detection AUROC
| setting       | none          | target_std    | coral         |
|:--------------|:--------------|:--------------|:--------------|
| gashis-random | 0.657 ± 0.006 | 0.659 ± 0.009 | 0.637 ± 0.002 |
| hmu-grouped   | 0.747 ± 0.033 | 0.760 ± 0.022 | 0.705 ± 0.011 |
| hmu-random    | 0.728 ± 0.009 | 0.700 ± 0.028 | 0.695 ± 0.017 |
