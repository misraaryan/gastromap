# ResNet50 (fine-tuned CNN) vs Phikon (frozen foundation model + linear probe)
Source: HMU-GC-HE-30K (TUM vs NOR). Cross-dataset target: all 33,284 GasHisSDB 160 px patches. 3 seeds, mean ± SD.

| Model | HMU split | In-domain AUC | Cross AUC | In-domain bal. acc | Cross bal. acc | Cross ECE |
|---|---|---|---|---|---|---|
| ResNet50 | grouped | 0.891 ± 0.010 | 0.692 ± 0.035 | 0.805 ± 0.017 | 0.541 ± 0.047 | 0.230 ± 0.091 |
| ResNet50 | random  | 0.916 ± 0.001 | 0.702 ± 0.029 | 0.831 ± 0.008 | 0.522 ± 0.008 | 0.243 ± 0.044 |
| Phikon   | grouped | 0.878 ± 0.013 | 0.862 ± 0.034 | 0.791 ± 0.010 | 0.691 ± 0.067 | 0.072 ± 0.042 |
| Phikon   | random  | 0.923 ± 0.004 | 0.811 ± 0.019 | 0.843 ± 0.003 | 0.677 ± 0.039 | 0.082 ± 0.061 |

Leakage gap (random − grouped, in-domain AUC): ResNet50 0.025 (0.013, 0.030, 0.031); Phikon 0.046 (0.033, 0.062, 0.042).
Generalisation gap (in-domain − cross AUC): ResNet50 0.199 ± 0.025 (grouped), 0.214 ± 0.029 (random); Phikon 0.016 ± 0.043 (grouped), 0.113 ± 0.023 (random).

Uncertainty on the unseen dataset (entropy): error-detection AUROC ResNet50 0.66–0.68 vs Phikon 0.73–0.75; referring the
20% least confident patches does not raise ResNet50 balanced accuracy (0.54 -> 0.53; 0.52 -> 0.50).

Reference: GasHisSDB-trained Phikon: in-domain AUC 1.000, cross (HMU) 0.768 ± 0.008 (see results/phikon/summary.md).
