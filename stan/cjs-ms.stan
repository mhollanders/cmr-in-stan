functions {
  #include util.stanfunctions
  #include cjs.stanfunctions
}

data {
  int<lower=1> I, J;  // number of individuals and surveys
  int<lower=2> S;  // number of alive states
  vector<lower=0>[J - 1] tau;  // survey intervals
  array[I, J] int<lower=0, upper=S> y;  // detection history
}

transformed data {
  int Jm1 = J - 1, Sm1 = S - 1;
  array[I, 2] int f_l = first_last(y);
}

parameters {
  vector<lower=0>[S] h;  // mortality hazard rates
  row_vector<lower=0>[S * Sm1] q;  // transition rates
  matrix<lower=0, upper=1>[S, Jm1] p;  // detection probabilities
}

model {
  matrix[S, Jm1] h_mat = rep_matrix(h, Jm1),
                 logit_p = logit(p);
  matrix[Jm1, S * Sm1] q_mat = rep_matrix(q, Jm1);
  /* Code change for individual effects
  array[I] matrix[S, Jm1] h_mat = rep_array(rep_matrix(h, Jm1), I),
                          logit_p = rep_array(logit(p), I);
  array[I] matrix[Jm1, S * Sm1] q_mat = rep_array(rep_matrix(q, Jm1), I); // */
  target += sum(cjs_ms(y, f_l, tau, h_mat, q_mat, logit_p));
  target += gamma_lupdf(h | 1, 1) + gamma_lupdf(q | 1, 1)
            + beta_lupdf(to_vector(p) | 1, 1);
}

generated quantities {
  vector[I] log_lik;
  {
    matrix[S, Jm1] h_mat = rep_matrix(h, Jm1),
                   logit_p = logit(p);
    matrix[Jm1, S * Sm1] q_mat = rep_matrix(q, Jm1);
    /* Code change for individual effects
    array[I] matrix[S, Jm1] h_mat = rep_array(rep_matrix(h, Jm1), I),
                            logit_p = rep_array(logit(p), I);
    array[I] matrix[Jm1, S * Sm1] q_mat = rep_array(rep_matrix(q, Jm1), I); // */
    log_lik = cjs_ms(y, f_l, tau, h_mat, q_mat, logit_p);
  }
}
