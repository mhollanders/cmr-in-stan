functions {
  #include util.stanfunctions
  #include js.stanfunctions
}

data {
  int<lower=1> I, J;  // number of individuals and surveys
  vector<lower=0>[J - 1] tau;  // survey intervals
  array[I, J] int<lower=0, upper=1> y;  // detection history
  int<lower=1> I_aug;  // number of augmented individuals
}

transformed data {
  int Jm1 = J - 1, I_all = I + I_aug;
  array[I, 2] int f_l = first_last(y);
  vector[Jm1] log_tau_scl = log(tau / sum(tau) * Jm1);
}

parameters {
  real<lower=0> h;  // mortality hazard rate
  vector<lower=0, upper=1>[J] p;  // detection probabilities
  real<lower=0, upper=1> psi;  // inclusion probability
  real<lower=0> mu;  // baseline entry rate
  vector<lower=0>[J] u;  // unnormalised entry rates
}

transformed parameters {
  vector[J] beta = u / sum(u),
            log_alpha = zeros_vector(J);
  log_alpha[2:J] += log(mu) + log_tau_scl;
}

model {
  vector[J] log_beta = log(beta);
  vector[Jm1] log_phi = -h * tau;
  vector[J] logit_p = logit(p);
  tuple(vector[I], vector[2], matrix[J, I], vector[J]) lp =
    js(y, f_l, psi, log_beta, log_phi, logit_p);
  target += sum(lp.1) + I_aug * log_sum_exp(lp.2);
  /* Code change for individual effects
  matrix[Jm1, I_all] log_phi = rep_matrix(-h * tau, I_all);
  matrix[J, I_all] logit_p = rep_matrix(logit(p), I_all);
  tuple(vector[I], matrix[2, I_aug], matrix[J, I], matrix[J, I_aug]) lp =
    js(y, f_l, psi, log_beta, log_phi, logit_p);
  target += sum(lp.1);
  for (i in 1:I_aug) {
    target += log_sum_exp(lp.2[:, i]);
  } // */
  target += gamma_lupdf(h | 1, 1) + beta_lupdf(p | 1, 1)
            + beta_lupdf(psi | 1, 1) + gamma_lupdf(mu | 1, 1)
            + gamma_lupdf(u | exp(log_alpha), 1);
}

generated quantities {
  vector[I] log_lik;
  real lprior = gamma_lpdf(h | 1, 1) + beta_lpdf(p | 1, 1)
                + beta_lpdf(psi | 1, 1) + gamma_lpdf(mu | 1, 1)
                + gamma_lpdf(u | exp(log_alpha), 1);
  array[J] int N, B, D;
  int N_super;
  {
    vector[J] log_beta = log(beta);
    vector[Jm1] log_phi = -h * tau;
    vector[J] logit_p = logit(p);
    tuple(vector[I], vector[2], matrix[J, I], vector[J]) lp;
    lp = js(y, f_l, psi, log_beta, log_phi, logit_p);
    log_lik = lp.1;
    tuple(array[J] int, array[J] int, array[J] int, int) latent =
      js_rng(lp, f_l, I_aug, log_phi, logit_p);
    /* Code change for individual effects
    matrix[Jm1, I_all] log_phi = rep_matrix(-h * tau, I_all);
    matrix[J, I_all] logit_p = rep_matrix(logit(p), I_all);
    tuple(vector[I], matrix[2, I_aug], matrix[J, I], matrix[J, I_aug]) lp =
      js(y, f_l, psi, log_beta, log_phi, logit_p);
    log_lik = lp.1;
    tuple(array[J] int, array[J] int, array[J] int, int) latent = 
      js_rng(lp, f_l, log_phi, logit_p); // */
    N = latent.1;
    B = latent.2;
    D = latent.3;
    N_super = latent.4;
  }
}
