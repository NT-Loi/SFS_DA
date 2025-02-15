# Statistical Inference for Feature Selection after Optimal Transport-based Domain Adaptation (AISTATS 2025)

This package implements a statistical inference for feature selection (FS) after optimal transport-based domain adaptation (OT-based DA). The main idea is to  leverages the SI framework and employs a divide-and conquer approach to efficiently compute the $p$-value. By providing valid $p$-values for the selected features, our proposed method not only controls the false positive rate (FPR) in FS under DA but also maximizes the true positive rate (TPR), i.e., reducing the false negative rate (FNR).
We believe this study represents a significant step toward controllable machine learning in the context of DA. 

See the paper <https://arxiv.org/abs/2410.15022> for more details.

## Installation & Requirements

This package has the following requirements:
- [numpy](http://numpy.org)
- [mpmath](http://mpmath.org/)
- [matplotlib](https://matplotlib.org/)
- [scipy](https://scipy.org/)
- [scikit-learn](http://scikit-learn.org)
- [statsmodels](https://www.statsmodels.org/)

We recommend to install or update anaconda to the latest version and use Python 3 (We used Python 3.12.3).

**NOTE**: We use scipy package (version 1.13.1) to solve the linear program (simplex method). However, the default package does not return the set of basic variables. Therefore, we slightly modified the package so that it can return the set of basic variables by replacing the two files '_linprog.py' and '_linprog_simplex.py' in scipy.optimize module with our modified files in the folder 'files_to_replace'.

## Reproducibility

- Example for computing $p$-value for Lasso after DA
```
>> ex0_p_value_lasso_DA.ipynb
```

- Example for computing $p$-value for Elastic Net after DA
```
>> ex1_p_value_elasticnet_DA.ipynb
```

- Check the uniformity of the pivot
```
>> ex2_validity_of_p_value.ipynb
```
