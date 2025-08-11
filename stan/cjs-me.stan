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
  int Jm1 = J - 1, Sm1 = S - 1, Sp1 = S + 1;
  array[I, 2] int f_l = first_last(y);
}

parameters {
  vector<lower=0>[S] h;  // mortality hazard rates
  row_vector<lower=0>[S * Sm1] q;  // transition rates
  matrix<lower=0, upper=1>[S, Jm1] p;  // detection probabilities
  vector<lower=0, upper=1>[Sm1] delta;  // event probabilities
  column_stochastic_matrix[S, J] eta;  // initial state probabilities
}

transformed parameters {
  real lprior = gamma_lpdf(h | 1, 1) + gamma_lpdf(q | 1, 1)
                + beta_lpdf(to_vector(p) | 1, 1);
}

model {
  matrix[Sp1, Sp1] Q = rate_matrix(h, q);
  matrix[S, S] log_E = log(triangular_bidiagonal_stochastic_matrix(delta));
  matrix[S, J] log_eta = log(eta);  
  array[Jm1] matrix[Sp1, Sp1] log_H;
  for (j in 1:Jm1) {
    log_H[j, :S] = log(matrix_exp(Q * tau[j])[:S]);
    log_H[j, Sp1] = append_col(rep_row_vector(negative_infinity(), S), 0);
  }
  matrix[S, Jm1] logit_p = logit(p);
  /* Code change for individual effects
  array[Jm1] matrix[Sp1, Sp1] log_H_j;
  for (j in 1:Jm1) {
    log_H_j[j, :S] = log(matrix_exp(Q * tau[j])[:S]);
    log_H_j[j, Sp1] = append_col(rep_row_vector(negative_infinity(), S), 0);
  }
  array[I, Jm1] matrix[Sp1, Sp1] log_H = rep_array(log_H_j, I);
  array[I] matrix[S, Jm1] logit_p = rep_array(logit(p), I); // */
  target += sum(cjs_me(y, f_l, log_H, logit_p, log_E, log_eta));
  target += lprior;
}

generated quantities {
  vector[I] log_lik;
  {
    matrix[Sp1, Sp1] Q = rate_matrix(h, q);
    matrix[S, S] log_E = log(triangular_bidiagonal_stochastic_matrix(delta));
    matrix[S, J] log_eta = log(eta);  
    array[Jm1] matrix[Sp1, Sp1] log_H;
    for (j in 1:Jm1) {
      log_H[j, :S] = log(matrix_exp(Q * tau[j])[:S]);
      log_H[j, Sp1] = append_col(rep_row_vector(negative_infinity(), S), 0);
    }
    matrix[S, Jm1] logit_p = logit(p);
    /* Code change for individual effects
    array[Jm1] matrix[Sp1, Sp1] log_H_j;
    for (j in 1:Jm1) {
      log_H_j[j, :S] = log(matrix_exp(Q * tau[j])[:S]);
      log_H_j[j, Sp1] = append_col(rep_row_vector(negative_infinity(), S), 0);
    }
    array[I, Jm1] matrix[Sp1, Sp1] log_H = rep_array(log_H_j, I);
    array[I] matrix[S, Jm1] logit_p = rep_array(logit(p), I); // */
    log_lik = cjs_me(y, f_l, log_H, logit_p, log_E, log_eta);
  }
}
