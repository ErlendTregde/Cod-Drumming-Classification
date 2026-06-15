# Signal-feature augmentation — logistic

Test set; click/vocal are the pair the analysis flagged. `vocal→click` = true vocals misclassified as click (and vice-versa).

| arm | val acc | test acc | click F1 | vocal F1 | vocal→click | click→vocal |
|---|---|---|---|---|---|---|
| embedding | 0.707 | 0.838 | 0.941 | 0.857 | 3 | 0 |
| features | 0.600 | 0.711 | 0.924 | 0.771 | 1 | 9 |
| embedding+features | 0.708 | 0.834 | 0.947 | 0.853 | 2 | 0 |
