# Advance_Econometrics_1: MLE, Indirect Inference, and Metropolis-Hastings Applications

This repository contains the solutions and analysis for a take-home exam in Advanced Econometrics (March 2025), exploring key concepts in time series modeling and Bayesian inference. The project is divided into two main problems, focusing on parameter estimation and simulation techniques.

## Project Contents

* `Advance_Econometrics_Take_Home_Exame_1.pdf`: A comprehensive report detailing the problem statements, methodologies, results, and discussion for both exercises.
* `P1_Indirect_Inference.m`: MATLAB script for Problem 1, implementing MA(1) MLE and indirect inference.
* `P2_MH.m`: MATLAB script for Problem 2, implementing the Metropolis-Hastings algorithm for an AR(2) model.

## Problem 1: MA(1) MLE and Indirect Inference

This section focuses on the estimation of the parameter $\theta$ for a true data generating process (DGP) defined as an MA(1) model: $x_t = u_t + \theta u_{t-1}$, where $u_t \sim i.i.d. \mathcal{N}(0,1)$. The exercise explores different estimation approaches through a Monte Carlo simulation ($S=10,000$ runs for various true $\theta$ values):

* **Direct MLE via MA(1) Log-likelihood:** Estimation of $\hat{\theta}_{MA(1)}$ directly via the true model's log-likelihood function.
* **Direct MLE via AR(1) Log-likelihood (Misspecified):** Estimation of $\hat{\theta}_{AR(1)}$ directly via an AR(1) model's log-likelihood function.
* **Indirect Inference via Implied AR(1) Estimate:** Estimation of $\tilde{\theta}_{MA(1)}$ indirectly via the implied estimate, which is computed 'inside' an AR(1) model specification but is robust to it.

**Key Findings:**
Results suggest that an AR(1) misspecification for an MA(1) process can lead to huge failure. This mismatch persists even asymptotically, especially for true values of $\theta$ near the unit root. In contrast, the implied estimator exhibits greater robustness, as this problem disappears asymptotically due to its misspecification robustness. The report includes detailed tables and plots showing the performance of each estimator across a grid of $\theta$ values and different sample sizes ($T=100$, $T=1,000$, $T=10,000$).

## Problem 2: Metropolis-Hastings & an AR(2) Model

This section delves into Bayesian inference using the Metropolis-Hastings (MH) algorithm for an AR(2) model: $y_{t}=\alpha_{1}y_{t-1}+\alpha_{2}y_{t-2}+\epsilon_{t}$, with $\epsilon_{t}\sim i.i.d. \mathcal{N}(0,1+\alpha^{2})$.

**Exploration Stages:**
1.  **AR(2) Data Simulation:** Data from this AR(2) process is simulated for $\alpha=0.45$ with different initial values ($y_0=y_1=0$, $y_0=y_1=25$, $y_0=y_1=100$) to observe how, independently of the initial value, these three series converge to the same stationary process.
2.  **Metropolis-Hastings Design:** Metropolis-Hastings belongs to the family of MSMC (Markov Chain Monte Carlo) methods and is a way of generating samples from a known distribution. It's ideal for producing samples from Posterior Distributions, especially when their properties are not always straightforward. The probability of retaining a certain draw is computed as a ratio between the target distribution evaluated at the new parameter versus the old parameter.
3.  **MH Implementation with Various Priors:** The MH algorithm is implemented and analyzed under four different prior specifications for the parameter $\alpha$:
    * **No informative Prior:** Here, the Posterior equals the (Conditional) Likelihood Function.
    * **Non-Stationary Restriction:** An indicative prior $p_s(\alpha) \sim 1(1-\alpha z-\alpha z^2; |z|>1 \forall z)$ is imposed to give probability zero to non-stationary candidates, rejecting any draw that violates stationarity.
    * **Normal Distributed Prior:** A very restricted prior $p_N(\alpha) \sim \mathcal{N}(0, 0.01^2)$, which basically means a strong belief that the parameter is zero.
    * **Beta Distributed Prior:** A prior $p_\beta(\alpha) \sim \beta(10, 10)$, which accounts for the belief that the parameter is between 0 and 0.5.

**Analysis and Discussion:**
For each prior, the generated MH chains and their corresponding non-parametric kernel density estimations (representing the Posterior Distribution) are analyzed across different chain sizes (100, 1,000, 10,000). The analysis highlights how strong beliefs can affect results, as shown by the very restricted Normal Prior example where even strong data signals might not fully center the distribution around the true value without proper "burn-out". The choice of the variance of the candidate generator Random Walk ($\Sigma$) is discussed, aiming for an acceptance rate between 30%-60%.

## How to Run the Code

To explore the analysis and replicate the results:

1.  **MATLAB Installation:** Ensure you have MATLAB installed on your system.
2.  **Navigate to Project Directory:** Open MATLAB and navigate to the directory containing `P1_Indirect_Inference.m` and `P2_MH.m`.
3.  **Run Scripts:**
    * For Problem 1: Run `P1_Indirect_Inference.m`.
    * For Problem 2: Run `P2_MH.m`.

    The scripts will generate plots and may output results to the MATLAB command window, corresponding to the figures and tables discussed in the `Advance_Econometrics_Take_Home_Exame_1.pdf` report.
