# Signal-feature augmentation — mlp

Test set; click/vocal are the pair the analysis flagged. `vocal→click` = true vocals misclassified as click (and vice-versa).

| arm | val acc | test acc | click F1 | vocal F1 | vocal→click | click→vocal |
|---|---|---|---|---|---|---|
| embedding | 0.720 | 0.845 | 0.951 | 0.857 | 2 | 0 |
| features | 0.598 | 0.721 | 0.924 | 0.767 | 2 | 2 |
| embedding+features | 0.719 | 0.817 | 0.943 | 0.834 | 2 | 1 |
