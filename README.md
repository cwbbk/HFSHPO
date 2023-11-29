# Hierarchical ontology terms selection-based knowledge discovery framework (HTSKDD)

This is a Python implementation of the hierarchical ontology terms selection-based knowledge discovery framework reported in

C. Wan and C. Barton (2024) A novel hierarchy-based knowledge discovery framework for elucidating human aging-related phenotypic abnormalities. Accepted by the 39th ACM/SIGAPP Symposium On Applied Computing (SAC 2024).

---------------------------------------------------------------
# Requirements

- Python 3.6 
- Numpy 
- Scikit-learn

---------------------------------------------------------------
# Running 

* Step 1. Download the Gene Ontology feature set, Gene Ontology ancestors/descendents sets, and the Human Phenotype Ontology label set files from ./data.

* Step 2. Execute the Python code ./src/HTSKDD.py to obtain the predictive accuracy of EL-HIP+ and the corresponding selected Gene Ontology terms for predicting different Human Phenotype Ontology terms.

