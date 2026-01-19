functions {
  #include util.stanfunctions
  #include js.stanfunctions
  #include js-rng.stanfunctions
}

data {
  int<lower=1> I, J;  // number of individuals and surveys
  int<lower=2> S;  // number of alive states
  vector<lower=0>[J - 1] tau;  // survey intervals
  array[I, J] int<lower=0, upper=S> y;  // detection history
  int<lower=1> I_aug;  // number of augmented individuals
  int<lower=0, upper=1> dirichlet;  // logistic-normal (0) or Dirichlet (1) entry
}

transformed data {
  int Jm1 = J - 1, Sm1 = S - 1, Sp1 = S + 1, I_all = I + I_aug;
  array[I, 2] int f_l = first_last(y);
  vector[Jm1] tau_scl = tau / exp(mean(log(tau))), log_tau_scl = log(tau_scl);
}

parameters {
  vector<lower=0>[S] h;  // mortality hazard rates
  row_vector<lower=0>[S * Sm1] q;  // transition rates
  matrix<lower=0, upper=1>[S, J] p;  // detection probabilities
  vector<lower=0, upper=1>[Sm1] delta;  // event probabilities
  real<lower=0> mu;  // Dirichlet concentration or logistic-normal scale
  real gamma;  // first entry offset
  sum_to_zero_vector[J] u;  // unconstrained entries
  column_stochastic_matrix[S, J] eta;  // entry state probabilities
  real<lower=0, upper=1> psi;  // inclusion probability
}

transformed parameters {
  vector[J] log_alpha = append_row(gamma, log_tau_scl), 
            log_beta;
  if (dirichlet) {
    log_alpha += log(mu);
    log_beta = sum_to_zero_log_simplex_jacobian(u);
  } else {
    log_alpha += mu * u;
    log_beta = log_softmax(log_alpha);
  }
  real lprior = gamma_lpdf(h | 1, 3) + gamma_lpdf(q | 1, 3)
                + beta_lpdf(to_vector(p) | 1, 1)
                + gamma_lpdf(mu | 1, 1) + std_normal_lpdf(gamma);
}

model {
  matrix[S, S] log_E = log(triangular_bidiagonal_stochastic_matrix(delta));
  matrix[S, J] log_eta = log(eta);
  matrix[Sp1, Sp1] Q = rate_matrix(h, q);
  array[Jm1] matrix[Sp1, Sp1] log_H;
  for (j in 1:Jm1) {
    log_H[j, :S] = log(matrix_exp(Q * tau_scl[j])[:S]);
    log_H[j, Sp1] = append_col(rep_row_vector(negative_infinity(), S), 0);
  }
  matrix[S, J] logit_p = logit(p);
  tuple(vector[I], vector[2], matrix[J, I], vector[J], array[I] matrix[S, J],
        matrix[S, J]) lp =
    js_me(y, f_l, log_H, logit_p, log_E, log_beta, log_eta, psi);
  target += sum(lp.1) + I_aug * log_sum_exp(lp.2);
  /* Code change for individual effects
  array[Jm1] matrix[Sp1, Sp1] log_H_j;
  for (j in 1:Jm1) {
    log_H_j[j, :S] = log(matrix_exp(Q * tau_scl[j])[:S]);
    log_H_j[j, Sp1] = append_col(rep_row_vector(negative_infinity(), S), 0);
  }
  array[I_all, Jm1] matrix[Sp1, Sp1] log_H = rep_array(log_H_j, I_all);
  array[I_all] matrix[S, J] logit_p = rep_array(logit(p), I_all);
  tuple(vector[I], matrix[2, I_aug], matrix[J, I], matrix[J, I_aug], 
        array[I] matrix[S, J], array[I_aug] matrix[S, J]) lp = 
    js_me(y, f_l, log_H, logit_p, log_E, log_beta, log_eta, psi);
  target += sum(lp.1);
  for (i in 1:I_aug) {
    target += log_sum_exp(lp.2[:, i]);
  } // */
  target += lprior;
  target += dirichlet ?
            dirichlet_lupdf(exp(log_beta) | exp(log_alpha)) 
            : std_normal_lupdf(u);
}

generated quantities {
  vector[I] log_lik;
  array[S, J] int N, B, D;
  int N_super;
  {
    matrix[S, S] log_E = log(triangular_bidiagonal_stochastic_matrix(delta));
    matrix[S, J] log_eta = log(eta);
    matrix[Sp1, Sp1] Q = rate_matrix(h, q);
    array[Jm1] matrix[Sp1, Sp1] log_H;
    for (j in 1:Jm1) {
      log_H[j, :S] = log(matrix_exp(Q * tau_scl[j])[:S]);
      log_H[j, Sp1] = append_col(rep_row_vector(negative_infinity(), S), 0);
    }
    matrix[S, J] logit_p = logit(p);
    tuple(vector[I], vector[2], matrix[J, I], vector[J], array[I] matrix[S, J],
          matrix[S, J]) lp =
      js_me(y, f_l, log_H, logit_p, log_E, log_beta, log_eta, psi);
    /* Code change for individual effects
    array[Jm1] matrix[Sp1, Sp1] log_H_j;
    for (j in 1:Jm1) {
      log_H_j[j, :S] = log(matrix_exp(Q * tau_scl[j])[:S]);
      log_H_j[j, Sp1] = append_col(rep_row_vector(negative_infinity(), S), 0);
    }
    array[I_all, Jm1] matrix[Sp1, Sp1] log_H = rep_array(log_H_j, I_all);
    array[I_all] matrix[S, J] logit_p = rep_array(logit(p), I_all);
    tuple(vector[I], matrix[2, I_aug], matrix[J, I], matrix[J, I_aug], 
          array[I] matrix[S, J], array[I_aug] matrix[S, J]) lp = 
      js_me(y, f_l, log_H, logit_p, log_E, log_beta, log_eta, psi); // */
    log_lik = lp.1;
    tuple(array[S, J] int, array[S, J] int, array[S, J] int, int) latent =
      js_me_rng(lp, y, f_l, log_H, logit_p, log_E, I_aug);
    N = latent.1;
    B = latent.2;
    D = latent.3;
    N_super = latent.4;
  }
}
