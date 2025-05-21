# Stochastic Processes Code Reference

This document provides a reference guide for the various stochastic process implementations found in `Code.txt`. Each section corresponds to a specific topic or technique in stochastic processes.

## Table of Contents

1. [Steady State Distributions](#steady-state-distributions)
2. [Poisson Distribution](#poisson-distribution)
3. [Weather Markov Chain Example](#weather-markov-chain-example)
4. [N-Step Transition](#n-step-transition)
5. [Alternative Steady State Calculation](#alternative-steady-state-calculation)
6. [Mean Recurrence Time](#mean-recurrence-time)
7. [Absorbing States](#absorbing-states)
8. [Transition Between States](#transition-between-states)
9. [Absorption Probabilities](#absorption-probabilities)
10. [Hidden Markov Model - Forward Algorithm](#hidden-markov-model---forward-algorithm)
11. [Queueing Theory](#queueing-theory)
12. [Poisson Process Functions](#poisson-process-functions)
13. [Waiting Time Predictions](#waiting-time-predictions)
14. [Gaussian Process Regression](#gaussian-process-regression)
15. [Mean First Passage Time](#mean-first-passage-time)

---

## Steady State Distributions

This section calculates the steady state distribution for a 2-state Markov chain using eigenvalue decomposition.

```r
transition_matrix <- matrix(c(0.6, 0.4, 0.3, 0.7), byrow=TRUE, nrow=2)
eigen_result <- eigen(t(transition_matrix))
steady_state <- eigen_result$vectors[,1] / sum(eigen_result$vectors[,1])
print(steady_state)
```

## Poisson Distribution

Implementation of a basic Poisson process simulation.

```r
lambda <- 4 # Average rate (events per unit time)
time_intervals <- 10
events <- rpois(time_intervals, lambda)
print(events)
```

## Weather Markov Chain Example

A 3-state Markov chain representing weather states with transition probabilities.

```r
weather_states <- c("Sunny", "Cloudy", "Rainy")

# Transition probability matrix P
P_weather <- matrix(c(
  0.7, 0.2, 0.1,  # From Sunny: P(S->S)=0.7, P(S->C)=0.2, P(S->R)=0.1
  0.3, 0.4, 0.3,  # From Cloudy: P(C->S)=0.3, P(C->C)=0.4, P(C->R)=0.3
  0.2, 0.4, 0.4   # From Rainy: P(R->S)=0.2, P(R->C)=0.4, P(R->R)=0.4
), nrow = 3, byrow = TRUE)

# Verify that each row sums to 1 (stochastic matrix property)
cat("\nVerify rows sum to 1:\n")
print(rowSums(P_weather))
```

## N-Step Transition

Calculation of n-step transition probabilities for a 4-state Markov chain.

```r
P_four_state <- matrix(c(
  0.5, 0.3, 0.2, 0.0,
  0.0, 0.6, 0.3, 0.1,
  0.2, 0.0, 0.5, 0.3,
  0.1, 0.1, 0.0, 0.8
), nrow = 4, byrow = TRUE)

n_step_transition <- function(P, n) {
  result <- P
  for (i in 2:n) {
    result <- result %*% P
  }
  return(result)
}

P_10step <- n_step_transition(P_four_state, 10)
print(round(P_10step, 4))  # Round to 4 decimal places for readability
```

## Alternative Steady State Calculation

Another method to compute the steady state distribution using the system of linear equations approach.

```r
P_simple <- matrix(c(
  0.3, 0.6, 0.1,
  0.4, 0.2, 0.4,
  0.1, 0.5, 0.4
), nrow = 3, byrow = TRUE)

A <- t(diag(3) - P_simple)
A[3,] <- rep(1, 3)
b <- c(0, 0, 1)
pi_simple <- solve(A, b)
print(round(pi_simple, 4))
```

## Mean Recurrence Time

Calculation of mean recurrence times from steady state probabilities.

```r
recurrence_times <- 1 / pi_simple
print(round(recurrence_times, 2))
```

## Absorbing States

Identification of absorbing states in a Markov chain.

```r
states <- c("$0", "$1", "$2", "$3")
is_absorbing <- diag(P_simple) == 1
absorbing_states <- states[is_absorbing]
cat("Absorbing states:", absorbing_states, "\n")
```

## Transition Between States

Analysis of transitions between specific states.

```r
#state 2 and 3 transition to state 2 and 3
Q = P_simple[c(2,3), c(2,3), drop=FALSE]
```

## Absorption Probabilities

Calculation of absorption probabilities.

```r
N <- solve(diag(nrow(Q)) - Q)
rowSums(N) # Absorbing Steps
B <- N %*% R # Absorbing Prob
```

## Hidden Markov Model - Forward Algorithm

Implementation of the forward algorithm for Hidden Markov Models.

```r
forward_algorithm <- function(A, B, pi, observations) {
  n_states <- nrow(A)
  n_steps <- length(observations)

  # Initialize alpha matrix (forward probabilities)
  alpha <- matrix(0, nrow = n_states, ncol = n_steps)

  # Compute alpha for t=1 (initialization)
  for (i in 1:n_states) {
    alpha[i, 1] <- pi[i] * B[i, observations[1]]
  }

  # Compute alpha for t=2,...,T (induction)
  for (t in 2:n_steps) {
    for (j in 1:n_states) {
      # Sum over all possible previous states
      sum_val <- 0
      for (i in 1:n_states) {
        sum_val <- sum_val + alpha[i, t-1] * A[i, j]
      }
      alpha[j, t] <- sum_val * B[j, observations[t]]
    }
  }

  # Total probability of the observation sequence
  prob_observations <- sum(alpha[, n_steps])

  return(list(
    alpha = alpha,
    prob_observations = prob_observations
  ))
}
```

## Queueing Theory

Basic queueing theory calculations for an M/M/1 queue.

```r
arrival_rate <- 2    # Customers arrive at rate of 2 per hour
service_rate <- 3    # Server can handle 3 customers per hour
time_horizon <- 8    # Simulate for 8 hours

# Theoretical results from queueing theory
rho <- arrival_rate / service_rate  # Traffic intensity
L <- rho / (1 - rho)                # Expected number of customers in system
W <- 1 / (service_rate - arrival_rate) # Expected time in system
Lq <- L - rho                       # Expected queue length
Wq <- W - 1/service_rate            # Expected waiting time
Ls <- L - Lq                        # Expected number of customers in service
Ws <- W - Wq                        # Expected time in service
```

## Poisson Process Functions

Various functions for working with Poisson processes and calculating probabilities.

```r
# Function to calculate probability of exactly n events in time period t with rate lambda
poisson_probability <- function(n, lambda, t) {
  expected_events <- lambda * t
  prob <- dpois(n, lambda = expected_events)
  return(prob)
}

# Function to calculate probability of at most n events in time period t
poisson_probability_at_most <- function(n, lambda, t) {
  expected_events <- lambda * t
  prob <- ppois(n, lambda = expected_events)
  return(prob)
}

# Function to calculate probability of at least n events in time period t
poisson_probability_at_least <- function(n, lambda, t) {
  expected_events <- lambda * t
  # P(X ≥ n) = 1 - P(X < n) = 1 - P(X ≤ n-1)
  prob <- 1 - ppois(n - 1, lambda = expected_events)
  return(prob)
}

# Function to calculate probability of between n1 and n2 events (inclusive)
poisson_probability_between <- function(n1, n2, lambda, t) {
  expected_events <- lambda * t
  # P(n1 ≤ X ≤ n2) = P(X ≤ n2) - P(X ≤ n1-1)
  prob <- ppois(n2, lambda = expected_events) - ppois(n1 - 1, lambda = expected_events)
  return(prob)
}
```

## Waiting Time Predictions

Functions for predicting waiting times in Poisson processes.

```r
# Function to predict next arrival time given the last arrival time
predict_next_arrival <- function(last_arrival, lambda) {
  inter_arrival_time <- rexp(1, rate = lambda)
  next_arrival <- last_arrival + inter_arrival_time
  return(next_arrival)
}

# Function to predict the distribution of waiting time until the next n events
predict_waiting_time <- function(n, lambda) {
  mean_waiting_time <- n / lambda
  variance <- n / (lambda^2)
  std_dev <- sqrt(variance)

  return(list(
    distribution = "Gamma",
    shape = n,
    rate = lambda,
    mean = mean_waiting_time,
    standard_deviation = std_dev
  ))
}

# Function to calculate probability of waiting less than t time units for the next event
prob_waiting_less_than <- function(t, lambda) {
  # P(T < t) = 1 - e^(-lambda*t)
  prob <- 1 - exp(-lambda * t)
  return(prob)
}

# Function to calculate probability of waiting more than t time units for the next event
prob_waiting_more_than <- function(t, lambda) {
  # P(T > t) = e^(-lambda*t)
  prob <- exp(-lambda * t)
  return(prob)
}
```

## Gaussian Process Regression

Implementation of Gaussian Process Regression using only built-in R functions.

```r
# Define the RBF kernel function
rbf_kernel <- function(x1, x2, l = 1, sigma_f = 1) {
  outer(x1, x2, function(a, b) sigma_f^2 * exp(- (a - b)^2 / (2 * l^2)))
}

# Simulate training data
set.seed(42)
x_train <- seq(-3, 3, length.out = 10)
y_train <- sin(x_train) + rnorm(length(x_train), sd = 0.1)

# Define test points
x_test <- seq(-3, 3, length.out = 100)

# Set hyperparameters
l <- 1          # length scale
sigma_f <- 1    # signal variance
sigma_n <- 0.1  # noise standard deviation

# Compute kernel matrices
K <- rbf_kernel(x_train, x_train, l, sigma_f)
K_s <- rbf_kernel(x_train, x_test, l, sigma_f)
K_ss <- rbf_kernel(x_test, x_test, l, sigma_f)

# Add noise to the diagonal of the training kernel matrix
K_noise <- K + sigma_n^2 * diag(length(x_train))

# Compute the posterior mean and covariance
K_inv <- solve(K_noise)
mu_s <- t(K_s) %% K_inv %% y_train  # Posterior mean
cov_s <- K_ss - t(K_s) %% K_inv %% K_s  # Posterior covariance

# Extract standard deviation for confidence interval
std_dev <- sqrt(diag(cov_s))
lower_bound <- mu_s - 1.96 * std_dev
upper_bound <- mu_s + 1.96 * std_dev
```

## Mean First Passage Time

Function to calculate mean first passage times in a Markov chain using the fundamental matrix approach.

```r
calculate_mfpt_fundamental <- function(P) {
  n <- nrow(P)

  # Calculate steady state distribution
  A <- t(diag(n) - P)
  A[n, ] <- rep(1, n)
  b <- c(rep(0, n-1), 1)
  pi <- solve(A, b)

  # Create matrix W with rows equal to π
  W <- matrix(pi, nrow = n, ncol = n, byrow = TRUE)

  # Calculate fundamental matrix Z = (I - P + W)^(-1)
  Z <- solve(diag(n) - P + W)

  # Calculate mean first passage times
  M <- matrix(0, nrow = n, ncol = n)
  for (i in 1:n) {
    for (j in 1:n) {
      if (i != j) {
        M[i, j] <- (Z[j, j] - Z[i, j]) / pi[j]
      } else {
        M[i, i] <- 1 / pi[i]
      }
    }
  }

  return(list(M = M, pi = pi, Z = Z))
}
```

---

Note: This README provides documentation for the code snippets in `Code.txt`. For execution, copy the relevant sections to an R environment. Some code snippets may have dependencies on previous sections or require additional input.