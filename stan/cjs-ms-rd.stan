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
  array[I] int g = first_sec(y, f_l[:, 1]);
}

parameters {
  vector<lower=0>[S] h;  // mortality hazard rates
  row_vector<lower=0>[S * Sm1] q;  // transition rates
  matrix<lower=0, upper=1>[S, J] p;  // detection probabilities
}

transformed parameters {
  real lprior = gamma_lpdf(h | 1, 1) + gamma_lpdf(q | 1, 1)
                + beta_lpdf(to_vector(p) | 1, 1);
}

model {
  array[Jm1] matrix[S, S] log_H;
  for (j in 1:Jm1) {
    log_H[j] = log(matrix_exp(rate_matrix(h, q)[:S, :S] * tau[j]));
  }
  array[J] matrix[S, K_max] logit_p;
  for (j in 1:J) {
    logit_p[j, :, :K[j]] = rep_matrix(logit(p[:, j]), K[j]);
  }
  /* Code change for individual effects
  array[Jm1] matrix[S, S] log_H_j;
  for (j in 1:Jm1) {
    log_H_j[j] = log(matrix_exp(rate_matrix(h, q)[:S, :S] * tau[j]));
  }
  array[I, Jm1] matrix[S, S] log_H = rep_array(log_H_j, I);
  array[J] matrix[S, K_max] logit_p_j;
  for (j in 1:J) {
    logit_p_j[j] = rep_matrix(logit(p[:, j]), K[j]);
  }
  array[I_all, J] matrix[S, K_max] logit_p = rep_array(logit_p_j, I_all);  // */
  target += sum(cjs_ms_rd(y, f_l, K, g, log_H, logit_p));
  target += lprior;
}

generated quantities {
  vector[I] log_lik;
  {
    array[Jm1] matrix[S, S] log_H;
    for (j in 1:Jm1) {
      log_H[j] = log(matrix_exp(rate_matrix(h, q)[:S, :S] * tau[j]));
    }
    array[J] matrix[S, K_max] logit_p;
    for (j in 1:J) {
      logit_p[j, :, :K[j]] = rep_matrix(logit(p[:, j]), K[j]);
    }
    /* Code change for individual effects
    array[Jm1] matrix[S, S] log_H_j;
    for (j in 1:Jm1) {
      log_H_j[j] = log(matrix_exp(rate_matrix(h, q)[:S, :S] * tau[j]));
    }
    array[I, Jm1] matrix[S, S] log_H = rep_array(log_H_j, I);
    array[J] matrix[S, K_max] logit_p_j;
    for (j in 1:J) {
      logit_p_j[j] = rep_matrix(logit(p[:, j]), K[j]);
    }
    array[I_all, J] matrix[S, K_max] logit_p = rep_array(logit_p_j, I_all); // */
    log_lik = cjs_ms_rd(y, f_l, K, g, log_H, logit_p);
  }
}
