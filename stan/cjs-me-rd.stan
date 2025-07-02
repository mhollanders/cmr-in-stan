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
  int Jm1 = J - 1, Sm1 = S - 1, Sp1 = S + 1;
  array[I, 2] int f_l = first_last(y);
  array[I] int g = first_sec(y, f_l[:, 1]);
}

parameters {
  vector<lower=0>[S] h;  // mortality hazard rates
  row_vector<lower=0>[S * Sm1] q;  // transition rates
  matrix<lower=0, upper=1>[S, J] p;  // detection probabilities
  column_stochastic_matrix[S, J] eta;  // initial state probabilities
  vector<lower=0, upper=1>[Sm1] delta;  // event probabilities
}

transformed parameters {
  real lprior = gamma_lpdf(h | 1, 1) + gamma_lpdf(q | 1, 1)
                + beta_lpdf(to_vector(p) | 1, 1);
}

model {
  matrix[Sp1, Sp1] Q = rep_matrix(0, Sp1, Sp1);
  Q[:S] = append_col(rate_matrix(h, q), h);
  matrix[S, S] log_E = log(triangular_bidiagonal_stochastic_matrix(delta));
  matrix[S, J] log_eta = log(eta);
  array[Jm1] matrix[Sp1, Sp1] log_H;
  for (j in 1:Jm1) {
    log_H[j, :S] = log(matrix_exp(Q * tau[j])[:S]);
    log_H[j, Sp1] = append_col(rep_row_vector(negative_infinity(), S), 0);
  }
  array[J] matrix[S, K_max] logit_p;
  for (j in 1:J) {
    logit_p[j, :, :K[j]] = rep_matrix(logit(p[:, j]), K[j]);
  }
  /* Code change for individual effects
  array[Jm1] matrix[Sp1, Sp1] log_H_j;
  for (j in 1:Jm1) {
    log_H_j[j, :S] = log(matrix_exp(Q * tau[j])[:S]);
    log_H_j[j, Sp1] = append_col(rep_row_vector(negative_infinity(), S), 0);
  }
  array[I, Jm1] matrix[Sp1, Sp1] log_H = rep_array(log_H_j, I);
  array[I, J] matrix[S, K_max] logit_p;
  for (j in 1:J) {
    vector[S] logit_p_j = logit(p[:, j]);
    for (i in 1:I) {
      logit_p[i, j, :, :K[j]] = rep_matrix(logit_p_j, K[j]);
    }
  } // */
  target += sum(cjs_me_rd(y, f_l, K, g, log_H, logit_p, log_E, log_eta));
  target += lprior;
}

generated quantities {
  vector[I] log_lik;
  {
    matrix[Sp1, Sp1] Q = rep_matrix(0, Sp1, Sp1);
    Q[:S] = append_col(rate_matrix(h, q), h);
    matrix[S, S] log_E = log(triangular_bidiagonal_stochastic_matrix(delta));
    matrix[S, J] log_eta = log(eta);
    array[Jm1] matrix[Sp1, Sp1] log_H;
    for (j in 1:Jm1) {
      log_H[j, :S] = log(matrix_exp(Q * tau[j])[:S]);
      log_H[j, Sp1] = append_col(rep_row_vector(negative_infinity(), S), 0);
    }
    array[J] matrix[S, K_max] logit_p;
    for (j in 1:J) {
      logit_p[j, :, :K[j]] = rep_matrix(logit(p[:, j]), K[j]);
    }
    /* Code change for individual effects
    array[Jm1] matrix[Sp1, Sp1] log_H_j;
    for (j in 1:Jm1) {
      log_H_j[j, :S] = log(matrix_exp(Q * tau[j])[:S]);
      log_H_j[j, Sp1] = append_col(rep_row_vector(negative_infinity(), S), 0);
    }
    array[I, Jm1] matrix[Sp1, Sp1] log_H = rep_array(log_H_j, I);
    array[I, J] matrix[S, K_max] logit_p;
    for (j in 1:J) {
      vector[S] logit_p_j = logit(p[:, j]);
      for (i in 1:I) {
        logit_p[i, j, :, :K[j]] = rep_matrix(logit_p_j, K[j]);
      }
    } // */
    log_lik = cjs_me_rd(y, f_l, K, g, log_H, logit_p, log_E, log_eta);
  }
}
