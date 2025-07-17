# Advance_Econometrics_2: Prediction and Financial Investment

This repository contains the solutions and analysis for a take-home exam in Advanced Econometrics (January 2025), divided into two distinct problems: a prediction task using various machine learning models and an optional problem on financial investment using causal inference techniques.

### Project Contents

* `Advance_Econometrics_Take_Home_Exam_2.ipynb`: The main Jupyter Notebook containing all the code, analysis, results, and comments for both Problem 1 and Problem 2.
* `calif_penn_2011_2.csv`: The primary dataset used for the prediction task in Problem 1.
* `calif_penn_2011.txt`: A text file containing descriptions for the columns in the `calif_penn_2011_2.csv` dataset.
* `pension.csv`: The dataset used for the financial investment problem (Problem 2).

### Problem 1: Prediction

This problem focuses on predicting a target variable using a real-world dataset. The main steps and models explored include:

1.  **Data Elicitation and Preprocessing:**
    * **Database Selection and Cleaning:** Loading the `calif_penn_2011_2.csv` dataset, which contains census tract information for California and Pennsylvania. Variables related to inner encoding systems (e.g., IDs, codes) were dropped due to their perceived lack of predictive power beyond the features they represent (e.g., location codes' impact is through location, not inherent meaning). Variables that could lead to perfect or high collinearity, such as percentage variables that sum to 100, or highly correlated measures like mean household income with median household income, were also removed.
    * **Missing Data Handling:** Rows where the dependent variable (`Median_house_value`) is null were removed, as introducing values might generate noise during model training. For other null values in independent variables, the median of the column was used for imputation, as the median is more robust to outliers commonly found in housing data.
    * **Data Splitting:** The cleaned dataset was divided into training (80%) and testing (20%) sets.

2.  **Parametric Approach (Linear Regression Models):**
    * **Ordinary Least Squares (OLS):**
        * **Up to Degree 1:** A basic linear regression model was fitted, yielding an MAE of 81680.32 and an RMSE of 108423.86.
        * **Up to Degree 2 + All Possible Interactions:** The OLS model was extended with polynomial features of degree 2 and all interaction terms. This significantly improved performance, resulting in an MAE of 66476.79 and an RMSE of 89874.35.
    * **LASSO Regression (L1 Regularization):**
        * **Up to Degree 1:** LASSO regression was implemented, with the optimal `alpha` hyperparameter (21.3) determined using 10-Fold Cross-Validation, resulting in an RMSE of 103645.67. The coefficients for this model are less interpretable and not directly comparable to OLS due to standardization and penalties.
        * **Up to Degree 2 + All Possible Interactions:** LASSO was applied with polynomial features of degree 2, achieving an MAE of 66313.08 and an RMSE of 90112.77. LASSO's design allows some coefficients to be zero, which helps with model overcomplexity, unlike OLS.
    * **Ridge Regression (L2 Regularization):**
        * **Up to Degree 2 + All Possible Interactions:** Ridge regression was implemented with polynomial features (degree 2), with the optimal `alpha` (1.7) found via 10-Fold Cross-Validation, leading to an MAE of 66310.28 and an RMSE of 89875.80.

3.  **Non-Parametric Approach (Tree-Based Models):**
    * **Random Tree Model (Decision Tree Regressor):**
        * **Hyperparameter Tuning:** Optimal `ccp_alpha` (0.0) and `min_samples_leaf` (2) were determined using 10-Fold Cross-Validation.
        * **Model Evaluation:** The model achieved an MAE of 49305.43 and an RMSE of 92889.00.
        * **Feature Importance:** `Median_household_income`, `LONGITUDE`, and `LATITUDE` were identified as the most important variables.
        * **Tree Visualization:** Attempting to plot the generated decision tree (noted to be very large).
    * **Random Forest Model:**
        * **Hyperparameter Tuning:** Optimal `min_samples_leaf` (1) and `max_features` (0.5) were identified using 10-Fold Cross-Validation.
        * **Model Evaluation:** This model achieved the lowest errors, with an MAE of 41829.03 and an RMSE of 71377.20.
        * **Feature Importance:** Similar to the Random Tree, `Median_household_income`, `LONGITUDE`, and `LATITUDE` were the top three most important features.
    * **General Comments on Tree Models:** Both Random Tree and Random Forest models favored high complexity (few minimum samples per leaf and no pruning), which is generally acceptable if hyperparameters are tuned via cross-validation. Direct comparison of RT and RF is noted as less meaningful given RF's nature as a refinement of RT, and the inherent 'black-box' nature of ML methods reduces interpretability. Multicollinearity is considered less problematic for these algorithms as their focus is on minimizing a loss function rather than interpretability.

4.  **Model Selection:**
    * Based on both MAE and RMSE metrics, the **Random Forest model** was selected as the best predictor.

### Problem 2: Financial Investment (Optional)

This optional problem aims to estimate the intention-to-treat (ITT) effect of participation in 401(k) pension plans (`p401`) on employees’ net financial assets (`net_tfa`), using eligibility for 401(k) plans (`e401`) as an instrumental variable to address endogeneity.

1.  **Data Loading and Inspection:**
    * The `pension.csv` dataset was loaded, containing `net_tfa`, `p401`, `e401`, and individual characteristics such as age, income, education, family size, marital status, and other pension/homeownership details.

2.  **OLS Approach:**
    * **Initial Model:** An OLS regression was performed with `net_tfa` as the dependent variable and `e401` as the key independent variable, along with various control variables including age, income, family size, marital status, homeownership, IRA, defined benefit pension, and education level dummies. The ITT effect of `e401` was estimated at 6440.91, but some variables (male, fsize, hown, and education dummies) were not statistically significant.
    * **Refined Model:** The model was adjusted by removing non-significant variables and replacing categorical education with a continuous `educ` variable, and `hown` with `hequity`. In this refined model, `educ` became statistically significant, while `marr` did not.
    * **Quadratic Education Test:** A quadratic term for `educ` was added, but it was not statistically significant, suggesting a linear effect for education.
    * **Final OLS Model:** The final OLS regression indicated an ITT effect of 6711.42 for `e401` on `net_tfa`.

3.  **LASSO Approach:**
    * LASSO regression was applied using the covariates from the final OLS model. It was noted that LASSO coefficients are generally less interpretable and not directly comparable to OLS due to standardization and penalty.

4.  **Double-Selection LASSO:**
    * **Concept:** This method involves a two-step process: first, regressing the outcome variable on covariates and the treatment variable on covariates; second, refitting an OLS model including only the covariates that were significant predictors in either of the first-step regressions.
    * **Implementation with Refined Covariates:** Using the refined set of covariates, the ITT effect of `e401` was estimated at 6727.05.
    * **Implementation with All Covariates:** When all variables were included, the ITT effect of `e401` became statistically insignificant (-38.23).
    * **Implementation with "Less Picky" Covariates:** A more curated selection of covariates resulted in an ITT effect of 8624.06 for `e401`.

5.  **Double Machine Learning (DML) with Random Forest:**
    * **Concept:** This approach uses flexible machine learners (like Random Forest) for the nuisance functions (g(X) and m(X)) in the partialling-out framework, allowing for non-linear relationships.
    * **Implementation with Refined Covariates:** Using Random Forest as the learner, the ITT effect of `e401` was estimated at 10507.17.
    * **Implementation with "Less Picky" Covariates:** With the "less picky" covariate set, the DML with Random Forest estimated the ITT effect of `e401` to be 7887.46.

6.  **Summary Table:**
    * A summary table compares the estimated ITT effects from OLS (6711.42), Double-Selection LASSO (-38.23 and 8624.06 depending on covariate selection), Double ML (10507.17), and Double ML (aug.) (7887.46). It is noted that choosing the "best" model for causal effect estimation can be challenging without further theoretical or empirical justification.

### How to Run the Code

To explore the analysis and replicate the results:

1.  **Google Colab/Jupyter Environment:** Open the `Advance_Econometrics_Take_Home_Exam_2.ipynb` notebook in Google Colab or any local Jupyter environment.
2.  **Upload Data Files:** Ensure `calif_penn_2011_2.csv`, `calif_penn_2011.txt`, and `pension.csv` are in the same directory as the notebook if running locally, or upload them to your Colab environment.
3.  **Install Libraries:** The notebook includes a `!pip install -U DoubleML` command. Run all other necessary `import` statements at the beginning of each problem section.
4.  **Run Cells:** Execute the cells sequentially. The notebook is designed to be run from top to bottom, with outputs and comments appearing in line.

---

**Note:** This notebook uses the `DoubleML` library for Double Machine Learning, which might require installation if not already present in your environment.

---
