functions {
  #include util.stanfunctions
  #include cjs.stanfunctions
}

data {
  int<lower=1> I, J;  // number of individuals and surveys
  vector<lower=0>[J - 1] tau;  // survey intervals
  array[I, J] int<lower=0, upper=1> y;  // detection history
  int<lower=0> grainsize;  // threading
}

transformed data {
  int Jm1 = J - 1;
  array[I, 2] int f_l = first_last(y);
  array[I] int seq = linspaced_int_array(I, 1, I);
}

parameters {
  real<lower=0> h;  // mortality hazard rate
  vector<lower=0, upper=1>[Jm1] p;  // detection probabilities
}

transformed parameters {
  real lprior = gamma_lpdf(h | 1, 3) + beta_lpdf(p | 1, 1);
}

model {
  vector[Jm1] log_phi = -h * tau, logit_p = logit(p);
  target += cjs(y, f_l, log_phi, logit_p);
  /* Code change for individual effects
  matrix[Jm1, I] log_phi = rep_matrix(-h * tau, I),
                 logit_p = rep_matrix(logit(p), I);
  target += grainsize ?
            reduce_sum(partial_cjs, seq, grainsize, y, f_l, log_phi, logit_p)
            : sum(cjs(y, f_l, log_phi, logit_p)); // */
  target += lprior;
}

generated quantities {
  vector[I] log_lik;
  {
    vector[Jm1] log_phi = -h * tau, logit_p = logit(p);
    /* Code change for individual effects
    matrix[Jm1, I] log_phi = rep_matrix(-h * tau, I),
                   logit_p = rep_matrix(logit(p), I); // */
    log_lik = cjs(y, f_l, log_phi, logit_p);
  }
}
