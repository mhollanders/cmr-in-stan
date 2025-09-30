if (!require(expm)) install.packages("expm")
simulate_cmr <- function(N_super = 100,
                         J = 8,
                         K = 1,
                         S = 1,
                         ME = 0,
                         JS = 0,
                         I_aug = 500,
                         grainsize = 0,
                         mu_gamma_prior = c(1, 1),
                         h_gamma_prior = c(1, 3),
                         p_beta_prior = c(1, 1), 
                         q_gamma_prior = c(1, 3), 
                         delta_beta_prior = c(1, 1),
                         eta_dirichlet_prior = 1) {
  
  # metadata, parameters, and detection histories
  Jm1 <- J - 1
  tau <- rlnorm(Jm1)
  tau_scl <- tau / mean(tau)
  h <- rgamma(S, h_gamma_prior[1], h_gamma_prior[2])
  p <- matrix(rbeta(S * J, p_beta_prior[1], p_beta_prior[2]), S, J)
  if (S == 1) {
    phi_tau <- exp(-h * tau)
  } else {
    Sm1 <- S - 1 ; Sp1 <- S + 1
    q <- rgamma(S * Sm1, q_gamma_prior[1], q_gamma_prior[2])
    H <- array(0, c(Jm1, Sp1, Sp1))
    for (j in 1:Jm1) {
      H[j, , ] <- expm::expm(rate_matrix(h, q) * tau_scl[j])
    }
    if (ME) {
      delta <- rbeta(Sm1, delta_beta_prior[1], delta_beta_prior[2])
      E <- triangular_bidiagonal_stochastic_matrix(delta)
    }
    if (ME | JS) {
      eta <- matrix(0, S, J)
      if (length(eta_dirichlet_prior) == 1) {
        alpha <- rep(eta_dirichlet_prior, S)
      } else {
        alpha <- eta_dirichlet_prior
      }
      for (j in 1:J) {
        eta[, j] <- rdirch(1, alpha)
      }
    }
  }
  z <- matrix(0, N_super, J)
  y <- array(0, c(N_super, J, K))
  
  # simulate
  if (JS) {
    mu <- rgamma(1, mu_gamma_prior[1], mu_gamma_prior[2])
    beta <- rdirch(1, c(1, mu * tau_scl))
    b <- sort(rcat(N_super, beta))
    if (S == 1) {
      B <- D <- numeric(J)
      for (i in 1:N_super) {
        z[i, b[i]] <- 1
        B[b[i]] <- B[b[i]] + 1
        if (b[i] < J) {
          for (j in (b[i] + 1):J) {
            jm1 <- j - 1
            z[i, j] <- rbinom(1, 1, z[i, jm1] * phi_tau[jm1])
            if (z[i, jm1] == 1 & z[i, j] == 0) {
              D[j] <- D[j] + 1
            }
          }
        }
        for (j in b[i]:J) {
          y[i, j, ] <- rbinom(K, 1, z[i, j] * p[j])
        }
      }
    } else {
      B <- D <- matrix(0, S, J)
      for (i in 1:N_super) {
        z[i, b[i]] <- rcat(1, eta[, b[i]])
        B[z[i, b[i]], b[i]] <- B[z[i, b[i]], b[i]] + 1
        if (b[i] < J) {
          for (j in (b[i] + 1):J) {
            jm1 <- j - 1
            z[i, j] <- rcat(1, H[jm1, z[i, jm1], ])
            if (z[i, jm1] <= S & z[i, j] == Sp1) {
              D[z[i, jm1], j] <- D[z[i, jm1], j] + 1
            }
          }
        }
        for (j in b[i]:J) {
          if (z[i, j] <= S) {
            y[i, j, ] <- rbinom(K, 1, p[z[i, j], j]) * z[i, j]
            if (ME) {
              for (k in 1:K) {
                if (y[i, j, k]) {
                  y[i, j, k] <- rcat(1, E[z[i, j], ])
                }
              }
            }
          }
        }
      }
    }
    obs <- which(rowSums(y) > 0)
    I <- length(obs)
    y <- y[obs, , 1:K]
  } else {
    I <- N_super
    f <- sort(sample(1:ifelse(K > 1 | ME, J, Jm1), I, replace = T))
    if (S == 1) {
      for (i in 1:I) {
        f_k <- sample(1:K, 1)
        z[i, f[i]] <- y[i, f[i], f_k] <- 1
        for (k in setdiff(1:K, f_k)) {
          y[i, f[i], k] <- rbinom(1, 1, p[f[i]])
        }
        if (f[i] < J) {
          for (j in (f[i] + 1):J) {
            jm1 <- j - 1
            z[i, j] <- rbinom(1, 1, z[i, jm1] * phi_tau[jm1])
            y[i, j, ] <- rbinom(K, 1, z[i, j] * p[j])
          }
        }
      }
    } else {
      for (i in 1:I) {
        f_k <- sample(1:K, 1)
        if (ME) {
          z[i, f[i]] <- rcat(1, eta[, f[i]])
          y[i, f[i], f_k] <- rcat(1, E[z[i, f[i]], ])
        } else {
          z[i, f[i]] <- y[i, f[i], f_k] <- sample(1:S, 1)
        }
        for (k in setdiff(1:K, f_k)) {
          y[i, f[i], k] <- rbinom(1, 1, p[z[i, f[i]], f[i]]) * z[i, f[i]]
          if (ME) {
            if (y[i, f[i], k]) {
              y[i, f[i], k] <- rcat(1, E[z[i, f[i]], ])
            }
          }
        }
        if (f[i] < J) {
          for (j in (f[i] + 1):J) {
            jm1 <- j - 1
            z[i, j] <- rcat(1, H[jm1, z[i, jm1], ])
            if (z[i, j] <= S) {
              y[i, j, ] <- rbinom(K, 1, p[z[i, j], j]) * z[i, j]
              if (ME) {
                for (k in 1:K) {
                  if (y[i, j, k]) {
                    y[i, j, k] <- rcat(1, E[z[i, j], ])
                  }
                }
              }
            }
          }
        }
      }
    }
    y <- y[, , 1:K]
  }
  
  # output
  if (JS | K > 1) {
    p <- p[1:S, ]
  } else {
    p <- p[1:S, -1]
  }
  variables <- list(h = h, p = p)
  generated <- list(I = I, J = J, tau = tau, y = y, grainsize = grainsize)
  if (K > 1) {
    generated <- append(generated,
                        list(K_max = K, K = rep(K, J)),
                        after = 2)
  }
  if (S > 1) {
    generated$S <- S
    variables <- append(variables, list(q = q))
    if (ME & !JS) {
      variables <- append(variables, list(eta = eta, delta = delta))
    }
  }
  if (JS) {
    generated$I_aug <- I_aug
    variables <- append(variables,
                        list(mu = mu,
                             beta = beta,
                             N_super = N_super,
                             B = B,
                             D = D,
                             N = apply(z, 2, \(j) 
                                       sapply(1:S, \(s) sum(j == s)))))
    if (S > 1) {
      variables <- append(variables, list(eta = eta))
    }
  }
  list(variables = variables, generated = generated)
}
