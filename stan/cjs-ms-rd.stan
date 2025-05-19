functions {
  #include util.stanfunctions
  #include cjs.stanfunctions
}

data {
  int<lower=1> I,  // number of individuals
               J,  // number of surveys
               K_max;  // maximum number of secondaries
  array[J] int<lower=1, upper=K_max> K;  // number of secondaries
  int<lower=2> S;  // number of alive states
  vector<lower=0>[J - 1] tau;  // survey intervals
  array[I, J, K_max] int<lower=0, upper=S> y;  // detection history
}

transformed data {
  int Jm1 = J - 1, Sm1 = S - 1;
  array[I, 2] int f_l = first_last(y);
  array[I] int f_k = first_sec(y, f_l[:, 1]);
}

parameters {
  vector<lower=0>[S] h;  // mortality hazard rates
  row_vector<lower=0>[S * Sm1] q;  // transition rates
  matrix<lower=0, upper=1>[S, J] p;  // detection probabilities
}

model {
  matrix[S, Jm1] h_mat = rep_matrix(h, Jm1);
  matrix[Jm1, S * Sm1] q_mat = rep_matrix(q, Jm1);
  array[J] matrix[S, K_max] logit_p;
  for (j in 1:J) {
    logit_p[j, :, 1:K[j]] = rep_matrix(logit(p[:, j]), K[j]);
  }
  /* Code change for individual effects
  array[I] matrix[S, Jm1] h_mat = rep_array(rep_matrix(h, Jm1), I);
  array[I] matrix[Jm1, S * Sm1] q_mat = rep_array(rep_matrix(q, Jm1), I);
  array[I, J] matrix[S, K_max] logit_p; 
  for (j in 1:J) {
    vector[S] logit_p_j = logit(p[:, j]);
    for (i in 1:I) {
      logit_p[i, j, :, 1:K[j]] = rep_matrix(logit_p_j, K[j]);
    }
  } // */
  target += sum(cjs_ms_rd(y, f_l, tau, f_k, K, h_mat, q_mat, logit_p));
  target += gamma_lupdf(h | 1, 1) + gamma_lupdf(q | 1, 1)
            + beta_lupdf(to_vector(p) | 1, 1);
}

generated quantities {
  vector[I] log_lik;
  {
    matrix[S, Jm1] h_mat = rep_matrix(h, Jm1);
    matrix[Jm1, S * Sm1] q_mat = rep_matrix(q, Jm1);
    array[J] matrix[S, K_max] logit_p;
    for (j in 1:J) {
      logit_p[j, :, 1:K[j]] = rep_matrix(logit(p[:, j]), K[j]);
    }
    /* Code change for individual effects
    array[I] matrix[S, Jm1] h_mat = rep_array(rep_matrix(h, Jm1), I);
    array[I] matrix[Jm1, S * Sm1] q_mat = rep_array(rep_matrix(q, Jm1), I);
    array[I, J] matrix[S, K_max] logit_p; 
    for (j in 1:J) {
      vector[S] logit_p_j = logit(p[:, j]);
      for (i in 1:I) {
        logit_p[i, j, :, 1:K[j]] = rep_matrix(logit_p_j, K[j]);
      }
    } // */
    log_lik = cjs_ms_rd(y, f_l, tau, f_k, K, h_mat, q_mat, logit_p);
  }
}
