# Bayesian Multivariate Markov Switching Model for Macroeconomic Analysis

This repository contains the implementation and estimation of a Multivariate Dynamic Factor Model with Markov Regime Switching (MS-DFM) using Bayesian techniques. This project was developed as part of the Macroeconometrics course and serves as a valuable tool for exploring the application of this novel methodology for macroeconomic data analysis. The model's utility stems from its ability to intelligently group multiple variables through a common factor.

## Project Objective

The primary objectives of this work are:
* To **implement a Multivariate MS-DFM** capable of modeling a set of monthly and quarterly macroeconomic variables.
* To **estimate the model parameters using Bayesian inference**, specifically through a Gibbs sampling scheme.
* To **derive and estimate the probability of "low activity" (recession)**, a key indicator for economic decision-making.

## Model Specification

While the full model specification is detailed in the accompanying Colab notebook, a key feature is the behavior of the common factor (denoted as $f_t$) and its associated augmentation variable ($x_t$), which are governed by a Markov Switching process:

$$f_t = (1-s_t)\mu_0 + s_t\mu_1 + s_tx_t + \varepsilon_{f,t} \sim N(0,\sigma^2_f)$$
$$x_t = s_tx_{t-1} + (1-s_t)v_t \sim N(0,\sigma^2_v)$$

Where $s_t$ represents the unobserved regime, which switches according to a Markov chain. This allows the model to capture different phases of the economic cycle.

## Estimation Methodology (Bayesian)

The model is estimated using **Gibbs Sampling**, a Markov Chain Monte Carlo (MCMC) algorithm that allows for drawing parameters from the joint posterior distribution. The process is decomposed into the following iterative steps:

1.  **Compute State Vector ($h_t$)**: This involves using the Kalman Filter for state prediction and covariance, and the Carter-Kohn algorithm to draw a sample of the unobserved states, conditional on current parameters and observations.
2.  **Compute Regime State Vector ($S_t$)**: Based on the estimated factors, the Hamilton Filter is applied to obtain regime probabilities, followed by the Kim & Nelson algorithm to sample the sequence of regimes.
3.  **Compute Unobservable Augmentation Vector ($x_t$)**: A sample of $x_t$ is generated based on its dynamics and the drawn regime states ($S_t$) and relevant variances.
4.  **Estimate Parameters**: Model parameters are updated by drawing from their conditional posterior distributions, utilizing conjugate priors (Normal-Inverse Gamma for coefficients and variances, and Dirichlet for transition probabilities).

## Project Contents

This repository contains the following files:

* `Final_Project_.ipynb`: The main Colab Notebook containing the implementation of the Bayesian Multivariate Markov Switching Dynamic Factor Model, including code, detailed analysis, and results. This is the primary notebook for exploring the methodology.
* `Results.ipynb`: A Colab Notebook containing the code that reproduce visualizations of the results.
* `Results.pdf`: A PDF with the visualization of the results.
* `Data_ME.ipynb`: A Colab Notebook dedicated to data elicitation, preprocessing, and cleaning for the macroeconomic variables used in the project.
* `USA_Normalize_Data.csv`: The primary dataset used for the estimation and analysis in the project.
* `Real-Time Weakness of the Global Economy.pdf`: A PDF document with the in-working-process paper we have replicated.


---

This project serves as a practical demonstration and exploration of how Bayesian Markov switching techniques can offer deeper insights into economic dynamics and business cycles through a dynamic factor framework.
