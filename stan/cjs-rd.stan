functions {
  #include util.stanfunctions
  #include cjs.stanfunctions
}

data {
  int<lower=1> I,  // number of individuals
               J,  // number of surveys
               K_max;  // maximum number of secondaries
  array[J] int<lower=1, upper=K_max> K;  // number of secondaries
  vector<lower=0>[J - 1] tau;  // survey intervals
  array[I, J, K_max] int<lower=0, upper=1> y;  // detection history
}

transformed data {
  int Jm1 = J - 1;
  array[I, 2] int f_l = first_last(y);
  array[I] int g = first_sec(y, f_l[:, 1]);
}

parameters {
  real<lower=0> h;  // mortality hazard rate
  row_vector<lower=0, upper=1>[J] p;  // detection probabilities
}

transformed parameters {
  real lprior = gamma_lpdf(h | 1, 1) + beta_lpdf(p | 1, 1);
}

model {
  vector[Jm1] log_phi = -h * tau;
  matrix[K_max, J] logit_p = rep_matrix(logit(p), K_max);
  /* Code change for individual effects
  matrix[Jm1, I] log_phi = rep_matrix(-h * tau, I);
  array[I] matrix[K_max, J] logit_p =
    rep_array(rep_matrix(logit(p), K_max), I); // */
  target += sum(cjs_rd(y, f_l, K, g, log_phi, logit_p));
  target += lprior;
}

generated quantities {
  vector[I] log_lik;
  {
    vector[Jm1] log_phi = -h * tau;
    matrix[K_max, J] logit_p = rep_matrix(logit(p), K_max);
    /* Code change for individual effects
    matrix[Jm1, I] log_phi = rep_matrix(-h * tau, I);
    array[I] matrix[K_max, J] logit_p =
      rep_array(rep_matrix(logit(p), K_max), I); // */
    log_lik = cjs_rd(y, f_l, K, g, log_phi, logit_p);
  }
}
