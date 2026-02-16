functions {
  #include ../stan/util.stanfunctions
  #include ../stan/js.stanfunctions
  #include ../stan/js-rng.stanfunctions
}

data {
  int<lower=1> I_max,  // maximum number of individuals
               J_max,  // maximum number of surveys
               K_max,  // maximum number of secondaries
               M;  // number of sites
  array[M] int<lower=1> I, J;  // number of individuals and surveys
  array[M, J_max] int<lower=0, upper=K_max> K;  // number of secondaries
  int<lower=2> S;  // number of alive states
  matrix<lower=0>[J_max - 1, M] tau;  // survey intervals
  array[M, I_max, J_max, K_max] int<lower=0, upper=S + 1> y;  // detection history
  array[M] int<lower=1> I_aug;  // number of augmented individuals
  real tau_mean;
}

transformed data {
  int I_sum = sum(I), J_sum = sum(J), K_sum = 0, Sm1 = S - 1, Sp1 = S + 1,
      T = S * Sm1;
  array[M] int I_all = I, Jm1 = J;
  array[M, I_max, 2] int f_l;
  matrix[J_max - 1, M] tau_scl, log_tau_scl;
  array[M, J_max] real dates = rep_array(0, M, J_max);
  for (m in 1:M) {
    I_all[m] += I_aug[m];
    Jm1[m] -= 1;
    f_l[m, :I[m]] = first_last(y[m, :I[m], :J[m]]);
    tau_scl[:Jm1[m], m] = tau[:Jm1[m], m] / tau_mean;
    log_tau_scl[:Jm1[m], m] = log(tau_scl[:Jm1[m], m]);
    vector[Jm1[m]] tau_cumsum = cumulative_sum(tau_scl[:Jm1[m], m]);
    for (j in 2:J[m]) {
      dates[m, j] = tau_cumsum[j - 1];
    }
    K_sum += sum(K[m]);
  }
  real m_z_scale = inv(sqrt(1 - inv(M)));
}

parameters {
  real<lower=0> h_a;  // mortality rate intercept
  real log_h_b;  // Bd infection
  vector<lower=0>[2] log_h_t;  // mortality rate site and Bd effect scales
  array[2] sum_to_zero_vector[M] log_h_z;  // site effects
  row_vector<lower=0>[T] q_a;  // transition rate intercepts
  vector<lower=0>[T] log_q_t;  // transition rate site effect scales
  array[T] sum_to_zero_vector[M] log_q_z;  // site effects
  real<lower=0, upper=1> p_a;  // detection probability intercept
  real<lower=0> logit_p_t;  // detection site effect scale
  sum_to_zero_vector[M] logit_p_z;  // detection site effects
  real<lower=0> mu_a, log_mu_t;  // entry rate scales
  sum_to_zero_vector[M] log_mu_z;  // entry site effects
  vector[J_sum - M] u;  // unnormalised entry probabilities
  vector<lower=0, upper=1>[M] psi;  // inclusion probabilities
  vector<lower=0>[5] gp_t;  // GP scales
  cholesky_factor_corr[5] gp_O_L;  // GP correlations 
  real<lower=0> gp_ell;  // GP length-scale
  matrix[J_sum - M, 5] gp_z_raw;  // GP z-scores
}

transformed parameters {
  // site-level parameters
  matrix[S, M] log_h_m = log(h_a) + rep_matrix(log_h_t[1] * log_h_z[1]', S);
  log_h_m[2] += log_h_b + log_h_t[2] * log_h_z[2]';
  matrix[M, T] log_q_m = rep_matrix(log(q_a), M);
  for (t in 1:T) {
    log_q_m[:, t] += log_q_t[t] * log_q_z[t];
  }
  vector[M] logit_p_m = logit(p_a) + logit_p_t * logit_p_z,
            log_mu_m = log(mu_a) + log_mu_t * log_mu_z;
  
  // primary-level parameters
  array[M] matrix[J_max, 5] gp_z, gp;
  array[M] matrix[S, J_max - 1] h;
  array[M] matrix[J_max, T] q;
  array[M] matrix[S, J_max] eta;
  array[M, J_max] matrix[S, K_max] logit_p;
  matrix[J_max, M] log_alpha = append_row(log_mu_m', log_tau_scl), log_beta;
  for (m in 1:M) {
    int J_m = J[m], Jm1_m = Jm1[m];
    for (d in 1:5) {
      gp_z[m, :J_m, d] = sum_to_zero_jacobian(segment(gp_z_raw[:, d], 
                                                      m == 1 ? 1 : Jm1[m - 1] + 1, 
                                                      Jm1_m));
    }
    gp[m, :J_m] = cholesky_decompose(gp_exp_quad_cov(dates[m, :J_m], 1, gp_ell))
                  * gp_z[m, :J_m]
                  * diag_post_multiply(gp_O_L', gp_t);
    h[m, :, :Jm1_m] = exp(rep_matrix(log_h_m[:, m], Jm1_m)
                          + rep_matrix(gp[m, 2:J_m, 1]', S));
    q[m, :J_m] = exp(rep_matrix(log_q_m[m], J_m) + gp[m, :J_m, 2:3]);
    logit_p[m, :J_m] = rep_array(rep_matrix(logit_p_m[m], S, K_max), J_m);
    for (j in 1:J_m) {
      eta[m, :, j] = reverse(q[m, j] / sum(q[m, j]))';
      logit_p[m, j] += gp[m, j, 4];
    }
    log_alpha[:J_m, m] += gp[m, :J_m, 5];
    log_beta[:J_m, m] = log_softmax(log_alpha[:J_m, m]);
  }
  real lprior = gamma_lpdf(h_a | 1, 3)
                + std_normal_lpdf(log_h_b)
                + exponential_lpdf(log_h_t | 2)
                + gamma_lpdf(q_a | 1, 3)
                + exponential_lpdf(log_q_t | 2)
                + beta_lpdf(p_a | 1, 2)
                + exponential_lpdf(logit_p_t | 2) 
                + gamma_lpdf(mu_a | 1, 1)
                + exponential_lpdf(log_mu_t | 2)
                + exponential_lpdf(gp_t | 2)
                + lkj_corr_cholesky_lpdf(gp_O_L | 1)
                + inv_gamma_lpdf(gp_ell | 1, 1);
}

model {
  for (m in 1:M) {
    array[Jm1[m]] matrix[S, S] log_H;
    for (j in 1:Jm1[m]) {
      log_H[j] = log(matrix_exp(rate_matrix(h[m, :, j], q[m, j + 1])[:S, :S] 
                                * tau_scl[j, m]));
    }
    tuple(vector[I[m]], vector[2], matrix[J[m], I[m]], vector[J[m]], 
          array[I[m]] matrix[S, J[m]], matrix[S, J[m]]) lp =
      js_ms_rd(y[m, :I[m], :J[m]], f_l[m, :I[m]], K[m, :J[m]], log_H, 
               logit_p[m, :J[m]], log_beta[:J[m], m], log(eta[m, :, :J[m]]), 
               psi[m]);
    target += sum(lp.1) + I_aug[m] * log_sum_exp(lp.2);
    target += std_normal_lupdf(to_vector(gp_z[m, :J[m]]));
  }
  target += lprior;
  target += normal_lupdf(log_h_z[1] | 0, m_z_scale)
            + normal_lupdf(log_h_z[2] | 0, m_z_scale)
            + normal_lupdf(log_q_z[1] | 0, m_z_scale)
            + normal_lupdf(log_q_z[2] | 0, m_z_scale)
            + normal_lupdf(logit_p_z | 0, m_z_scale)
            + normal_lupdf(log_mu_z | 0, m_z_scale);
}

generated quantities {
  vector[I_sum] log_lik;
  array[M, S, J_max] int N = rep_array(0, M, S, J_max),
                         B = rep_array(0, M, S, J_max),
                         D = rep_array(0, M, S, J_max);
  array[M] int N_super;
  {
    matrix[Sp1, Sp1] Q = rep_matrix(0, Sp1, Sp1);
    for (m in 1:M) {
      array[Jm1[m]] matrix[Sp1, Sp1] log_H;
      for (j in 1:Jm1[m]) {
        Q = rate_matrix(h[m, :, j], q[m, j + 1]);
        log_H[j, :S] = log(matrix_exp(Q * tau[j, m])[:S]);
        log_H[j, Sp1] = append_col(rep_row_vector(negative_infinity(), S), 0);
      }
      tuple(vector[I[m]], vector[2], matrix[J[m], I[m]], vector[J[m]],
          array[I[m]] matrix[S, J[m]], matrix[S, J[m]]) lp =
        js_ms_rd(y[m, :I[m], :J[m]], f_l[m, :I[m]], K[m, :J[m]], log_H[:, :S, :S],
                 logit_p[m, :J[m]], log_beta[:J[m], m], log(eta[m, :, :J[m]]),
                 psi[m]);
      log_lik[sum(I[:m - 1]) + 1:sum(I[:m])] = lp.1;
      tuple(array[S, J[m]] int, array[S, J[m]] int, array[S, J[m]] int, int)
        latent = js_ms_rd_rng(lp, y[m, :I[m], :J[m]], f_l[m, :I[m]], K[m, :J[m]],
                              log_H, logit_p[m, :J[m]], I_aug[m]);
      N[m, :, :J[m]] = latent.1;
      B[m, :, :J[m]] = latent.2;
      D[m, :, :J[m]] = latent.3;
      N_super[m] = latent.4;
    }
  }
  matrix[5, 5] gp_O = multiply_lower_tri_self_transpose(gp_O_L);
}
