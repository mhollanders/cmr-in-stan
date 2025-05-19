functions {
  #include util.stanfunctions
  #include js.stanfunctions
}

data {
  int<lower=1> I,  // number of individuals
               J,  // number of surveys
               K_max;  // maximum number of secondaries
  vector<lower=0>[J - 1] tau;  // survey intervals
  array[J] int<lower=0, upper=K_max> K;  // number of secondaries
  array[I, J, K_max] int<lower=0, upper=1> y;  // detection history
  int<lower=1> I_aug;  // number of augmented individuals
}

transformed data {
  int Jm1 = J - 1, I_all = I + I_aug;
  array[I, 2] int f_l = first_last(y);
  vector[Jm1] log_tau_scl = log(tau / sum(tau) * Jm1);
}

parameters {
  real<lower=0> h;  // mortality hazard rate
  row_vector<lower=0, upper=1>[J] p;  // detection probabilities
  real<lower=0, upper=1> psi;  // inclusion probability
  real<lower=0> mu;  // baseline entry rate
  vector<lower=0>[J] u;  // unnormalised entry rates
}

transformed parameters {
  vector[J] log_alpha = zeros_vector(J),  // log concentration vector
            beta = u / sum(u);  // entry probabilities
  log_alpha[2:J] += log(mu) + log_tau_scl;
}

model {
  vector[J] log_beta = log(beta);
  vector[Jm1] log_phi = -h * tau;
  matrix[K_max, J] logit_p = rep_matrix(logit(p), K_max);
  tuple(vector[I], vector[2], matrix[J, I], vector[J]) lp;
  lp = js_rd(y, f_l, K, psi, log_beta, log_phi, logit_p);
  target += sum(lp.1) + I_aug * log_sum_exp(lp.2);
  /* Code change for individual effects
  matrix[Jm1, I_all] log_phi = rep_matrix(-h * tau, I_all);
  array[I_all] matrix[K_max, J] logit_p =
    rep_array(rep_matrix(logit(p), K_max), I_all);
  tuple(vector[I], matrix[2, I_aug], matrix[J, I], matrix[J, I_aug]) lp;
  lp = js_rd(y, f_l, K, psi, log_beta, log_phi, logit_p);
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
  array[J] int N, B, D;
  int N_super;
  {
    vector[J] log_beta = log(beta);
    tuple(array[J] int, array[J] int, array[J] int, int) latent;
    vector[Jm1] log_phi = -h * tau;
    matrix[K_max, J] logit_p = rep_matrix(logit(p), K_max);
    tuple(vector[I], vector[2], matrix[J, I], vector[J]) lp;
    lp = js_rd(y, f_l, K, psi, log_beta, log_phi, logit_p);
    latent = js_rd_rng(lp, f_l, K, log_phi, logit_p, I_aug);
  /* Code change for individual effects
    matrix[Jm1, I_all] log_phi = rep_matrix(-h * tau, I_all);
    array[I_all] matrix[K_max, J] logit_p =
      rep_array(rep_matrix(logit(p), K_max), I_all);
    tuple(vector[I], matrix[2, I_aug], matrix[J, I], matrix[J, I_aug]) lp;
    lp = js_rd(y, f_l, K, psi, log_beta, log_phi, logit_p);
    latent = js_rd_rng(lp, f_l, K, log_phi, logit_p); // */
    log_lik = lp.1;
    N = latent.1;
    B = latent.2;
    D = latent.3;
    N_super = latent.4;
  }
}
