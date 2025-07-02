functions {
  #include util.stanfunctions
  #include js.stanfunctions
  #include js-rng.stanfunctions
}

data {
  int<lower=1> I, J;  // number of individuals and surveys
  int<lower=2> S;  // number of alive states
  vector<lower=0>[J - 1] tau;  // survey intervals
  array[I, J] int<lower=0, upper=S + 1> y;  // detection history
  int<lower=1> I_aug;  // number of augmented individuals
}

transformed data {
  int I_all = I + I_aug, Jm1 = J - 1, Sm1 = S - 1, Sp1 = S + 1;
  array[I, 2] int f_l = first_last(y);
  vector[Jm1] log_tau_scl = log(tau / sum(tau) * Jm1);
}

parameters {
  vector<lower=0>[S] h;  // mortality hazard rates
  row_vector<lower=0>[S * Sm1] q;  // transition rates
  matrix<lower=0, upper=1>[S, J] p;  // detection probabilities
  real<lower=0> mu;  // baseline entry rate
  simplex[J] beta;  // entry probabilities
  column_stochastic_matrix[S, J] eta;  // entry state probabilities
  real<lower=0, upper=1> psi;  // inclusion probability
}

transformed parameters {
  vector[J] log_alpha = zeros_vector(J);
  log_alpha[2:] += log(mu) + log_tau_scl;
  real lprior = gamma_lpdf(h | 1, 1) + gamma_lpdf(q | 1, 1)
                + beta_lpdf(to_vector(p) | 1, 1)
                + gamma_lpdf(mu | 1, 1) + dirichlet_lpdf(beta | exp(log_alpha));
}

model {
  vector[J] log_beta = log(beta);
  matrix[S, J] log_eta = log(eta);
  array[Jm1] matrix[S, S] log_H;
  for (j in 1:Jm1) {
    log_H[j] = log(matrix_exp(rate_matrix(h, q) * tau[j]));
  }
  matrix[S, J] logit_p = logit(p);
  tuple(vector[I], vector[2], matrix[J, I], vector[J], array[I] matrix[S, J],
        matrix[S, J]) lp =
    js_ms(y, f_l, log_H, logit_p, log_beta, log_eta, psi);
  target += sum(lp.1) + I_aug * log_sum_exp(lp.2);
  /* Code change for individual effects
  array[Jm1] matrix[S, S] log_H_j;
  for (j in 1:Jm1) {
    log_H_j[j] = log(matrix_exp(rate_matrix(h, q) * tau[j]));
  }
  array[I_all, Jm1] matrix[S, S] log_H = rep_array(log_H_j, I_all);
  array[I_all] matrix[S, J] logit_p = rep_array(logit(p), I_all);
  tuple(vector[I], matrix[2, I_aug], matrix[J, I], matrix[J, I_aug], 
        array[I] matrix[S, J], array[I_aug] matrix[S, J]) lp = 
    js_ms(y, f_l, log_H, logit_p, log_beta, log_eta, psi);
  target += sum(lp.1);
  for (i in 1:I_aug) {
    target += log_sum_exp(lp.2[:, i]);
  } // */
  target += lprior;
}

generated quantities {
  vector[I] log_lik;
  array[S, J] int N, B, D;
  int N_super;
  {
    vector[J] log_beta = log(beta);
    matrix[S, J] log_eta = log(eta);
    matrix[Sp1, Sp1] Q = rep_matrix(0, Sp1, Sp1);
    Q[:S] = append_col(rate_matrix(h, q), h);
    array[Jm1] matrix[Sp1, Sp1] log_H;
    for (j in 1:Jm1) {
      log_H[j, :S] = log(matrix_exp(Q * tau[j])[:S]);
      log_H[j, Sp1] = append_col(rep_row_vector(negative_infinity(), S), 0);
    }
    matrix[S, J] logit_p = logit(p);
    tuple(vector[I], vector[2], matrix[J, I], vector[J], array[I] matrix[S, J],
          matrix[S, J]) lp =
      js_ms(y, f_l, log_H[:, :S, :S], logit_p, log_beta, log_eta, psi);
    /* Code change for individual effects
    array[Jm1] matrix[Sp1, Sp1] log_H_j;
    for (j in 1:Jm1) {
      log_H_j[j, :S] = log(matrix_exp(Q * tau[j])[:S]);
      log_H_j[j, Sp1] = append_col(rep_row_vector(negative_infinity(), S), 0);
    }
    array[I_all, Jm1] matrix[Sp1, Sp1] log_H = rep_array(log_H_j, I_all);
    array[I_all] matrix[S, J] logit_p = rep_array(logit(p), I_all);
    tuple(vector[I], matrix[2, I_aug], matrix[J, I], matrix[J, I_aug], 
          array[I] matrix[S, J], array[I_aug] matrix[S, J]) lp = 
      js_ms(y, f_l, log_H[:, :, :S, :S], logit_p, log_beta, log_eta, psi); // */
    log_lik = lp.1;
    tuple(array[S, J] int, array[S, J] int, array[S, J] int, int) latent = 
      js_ms_rng(lp, y, f_l, log_H, logit_p, I_aug);
    N = latent.1;
    B = latent.2;
    D = latent.3;
    N_super = latent.4;
  }
}
