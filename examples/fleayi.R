# load packages
if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, cmdstanr, loo, tidybayes, posterior, ggh4x, MetBrewer)
source(here("sbc/util.R"))
options(mc.cores = 8)
theme_set(my_theme())

# read or prepare data for Stan
file_name <- here("examples/data/fleayi-stan-data.rds")
if (file.exists(file_name)) {
  stan_data <- read_rds(file_name)
  dates <- read_rds(here("examples/data/fleayi-dates.rds"))
} else {
  # prepare and modify detection histories
  I <- c(462, 136, 88)
  I_max <- max(I)
  J_max <- 23
  K_max <- 3
  M <- 3
  dh <- read_csv("https://raw.githubusercontent.com/mhollanders/mfleayi-adults/master/data/ch.csv", 
                 col_names = F) |> 
    unlist() |> 
    array(c(I_max, J_max, K_max, M)) |> 
    aperm(c(4, 1, 2, 3))
  read_csv("https://raw.githubusercontent.com/mhollanders/mfleayi-adults/master/data/sex.csv", 
           col_names = F)
  K_raw <- apply(dh[,1,,], 1:2, \(x) length(which(!is.na(x))))
  J <- apply(K_raw, 1, \(x) sum(x > 0))
  J_max <- max(J)
  y <- array(0, c(M, I_max, J_max, K_max))
  K <- matrix(0, M, J_max)
  for (m in 1:M) {
    surveyed <- which(K_raw[m, ] > 0)
    y[m, 1:I[m], 1:J[m], ] <- dh[m, 1:I[m], surveyed, ]
    K[m, 1:J[m]] <- K_raw[m, surveyed]
  }
  
  # remove known females; assume NA are male
  sex <- read_csv("https://raw.githubusercontent.com/mhollanders/mfleayi-adults/master/data/sex.csv") |> 
    mutate_all(~replace_na(., 0))
  I <- I - colSums(sex)
  I_max <- max(I)
  for (m in 1:M) {
    males <- which(pull(sex, m) == 0)[1:I[m]]
    y[m, 1:I[m], , ] <- y[m, males, , ]
  }
  y <- y[, 1:I_max, , ]
  y[is.na(y)] <- 0

  # prepare dates
  dates <- read_csv("https://raw.githubusercontent.com/mhollanders/mfleayi-adults/master/data/dates.csv") |> 
    mutate_all(dmy) |> 
    mutate(tuntable = if_else(tuntable == "2020-02-01", 
                              ymd("2021-02-01"), 
                              tuntable)) |> 
    pivot_longer(everything(), names_to = "site", values_to = "date") |> 
    mutate(site = factor(site, levels = c("brindle", "tuntable", "batcave")),
           m = as.integer(site)) |> 
    arrange(site, date) |> 
    mutate(j = row_number(), .by = m) |> 
    left_join(t(K) |> 
                as_tibble() |> 
                set_names(1:M) |> 
                pivot_longer(everything(), names_to = "m", values_to = "K") |> 
                mutate(m = as.integer(m)) |> 
                mutate(j = row_number(), .by = m)) |> 
    filter(K > 0) |> 
    mutate(date_fct = factor(date) |> as.integer())
  write_rds(dates, here("examples/fleayi-dates.rds"))
  
  # tau and dates
  tau <- matrix(0, J_max - 1, M)
  date <- dates |> distinct(date) |> pull(date) |> as.integer()
  J_star <- length(date)
  date_idx <- matrix(0, M, J_max)
  for (m in 1:M) {  
    dates_m <- dates |> 
      filter(as.integer(site) == .env$m)
    tau[1:(J[m] - 1), m] <- dates_m |> 
      mutate(tau = as.numeric(difftime(date, lag(date), units = "weeks"))) |> 
      drop_na() |> 
      pull(tau)
    date_idx[m, 1:J[m]] <- dates_m |> 
      pull(date_fct) |> 
      as.integer()
  }
   
  # data for Stan
  stan_data <- list(I_max = I_max, 
                    J_max = J_max, 
                    K_max = K_max,
                    M = M,
                    J_star = J_star,
                    I = I, 
                    J = J, 
                    K = K, 
                    S = 2, 
                    tau = tau,
                    date = date / 7,  # weekly date intervals
                    date_idx = date_idx,
                    y = y, 
                    I_aug = c(200, 100, 50),
                    tau_mean = mean(tau[tau > 0])) |> 
    glimpse()
  write_rds(stan_data, file_name)
}

mods <- list(cmdstan_model(here("examples/js-ms-rd-fleayi.stan")),
             cmdstan_model(here("examples/js-ms-rd-fleayi2.stan")),
             cmdstan_model(here("examples/js-ms-rd-fleayi3.stan")))
fits <- map(mods, 
            ~.$sample(stan_data, init = 0.1, chains = getOption("mc.cores"), 
                      iter_warmup = 200, iter_sampling = 500, 
                      show_exceptions = F))
loos <- map(fits, ~.$loo())
loo_compare(loos)

map(fits, ~.$time())

fits[[1]]$draws("log_alpha") |> mcmc_trace()

# fit
fit_name <- here("examples/fit.rds")
if (file.exists(fit_name)) {
  fit <- read_rds(fit_name)
} else {
  mod <- cmdstan_model(here("examples/js-ms-rd-fleayi.stan"))
  fit4 <- mod$sample(stan_data, init = 0.1, chains = getOption("mc.cores"), 
                    iter_warmup = 200, iter_sampling = 200, 
                    show_exceptions = T)
  fit$save_object(fit_name)
}

fit3$draws("log_mu_m") |> mcmc_intervals()

map(fits, ~spread_rvars(., log_beta[j, m])) |> 
  list_rbind(names_to = "mod") |> 
  left_join(dates)

# beta
bind_rows(spread_rvars(fits[[1]], beta[j, m]) |> 
            mutate(mod = "Dirichlet"),
          spread_rvars(fits[[2]], log_beta[j, m]) |> 
            mutate(mod = "LN GP",
                   beta = exp(log_beta)) |> 
            select(-log_beta)) |> 
  left_join(dates) |> 
  drop_na() |> 
  ggplot(aes(date, ydist = beta)) + 
  facet_wrap(~ site, ncol = 1) + 
  stat_pointinterval(aes(colour = mod), 
                     position = position_dodge(width = 20))

fit3 |> 
  spread_rvars(log_beta[j, m]) |> 
  left_join(dates) |> 
  drop_na() |> 
  ggplot(aes(date, ydist = exp(log_beta))) + 
  facet_wrap(~ site, ncol = 1) + 
  stat_pointinterval()

fit |> 
  spread_rvars(log_alpha[j, m]) |> 
  left_join(dates) |> 
  drop_na() |> 
  ggplot(aes(date, ydist = log_alpha)) + 
  facet_wrap(~ site, ncol = 1) + 
  stat_pointinterval()

fit$draws("gp_rho") |> mcmc_intervals()
  

# log_alpha
bind_rows(spread_rvars(fits[[1]], log_alpha[j, m]) |> 
            mutate(mod = "Dirichlet GP"),
          spread_rvars(fits[[2]], log_alpha[j, m]) |> 
            mutate(mod = "LN GP"),
          spread_rvars(fits[[3]], log_alpha[j, m]) |> 
            mutate(mod = "Dirichlet")) |> 
  left_join(dates) |> 
  drop_na() |> 
  ggplot(aes(date, ydist = log_alpha)) + 
  facet_wrap(~ site, ncol = 1) + 
  stat_pointinterval(aes(colour = mod), 
                     position = position_dodge(width = 20))

# plots
fit2 |> 
  spread_rvars(N[m, s, j]) |> 
  mutate(N_sum = rvar_sum(N), .by = c(m, j)) |> 
  mutate(prev = N / N_sum) |> 
  filter(s == 2) |> 
  left_join(dates, by = c("m", "j")) |> 
  pivot_longer(c(N_sum, prev)) |> 
  ggplot(aes(date, ydist = value)) + 
  facet_grid2(factor(m, 
                     labels = str_c(c("Brindle", "Tuntable", "Bat Cave"), 
                                    " Creek")) ~ 
                factor(name, 
                       labels = c("Population Size", "Infection Prevalence")),
              scales = "free", independent = "y") + 
  stat_pointinterval(point_interval = median_hdci,
                     .width = 0.95, size = 0.5, linewidth = 0.5) + 
  facetted_pos_scales(y = list(scale_y_continuous(breaks = seq(40, 120, 40),
                                                  limits = c(0, 145),
                                                  expand = c(0, 0)),
                               scale_y_continuous(breaks = seq(0.2, 0.8, 0.2),
                                                  limits = c(0, 0.9),
                                                  expand = c(0, 0)),
                               scale_y_continuous(breaks = seq(20, 60, 20),
                                                  limits = c(0, 72.5),
                                                  expand = c(0, 0)),
                               scale_y_continuous(breaks = seq(0.2, 0.8, 0.2),
                                                  limits = c(0, 0.9),
                                                  expand = c(0, 0)),
                               scale_y_continuous(breaks = seq(7, 21, 7),
                                                  limits = c(0, 24.5),
                                                  expand = c(0, 0)),
                               scale_y_continuous(breaks = seq(0.2, 0.8, 0.2),
                                                  limits = c(0, 0.9),
                                                  expand = c(0, 0)))) +
  scale_x_date(date_labels = "%b %y") +
  labs(x = "Primary", y = "95% HDI")
ggsave(here("figs/fig-fleayi.jpg"), width = 6, height = 6, dpi = 600)

fits[[2]] |> 
  spread_rvars(gp[m, j, d]) |> 
  left_join(dates |> 
              distinct(m, j, date)) |> 
  ggplot(aes(date, ydist = gp)) + 
  facet_wrap(~ factor(d, labels = c("Mortality", "Gaining Bd", "Clearing Bd", 
                                    "Detection", "Entry")), 
             ncol = 2) + 
  geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.2, colour = "#333333") + 
  stat_pointinterval(aes(colour = factor(m, labels = c("Brindle Creek", 
                                                       "Tuntable Creek", 
                                                       "Bat Cave Creek"))),
                     point_interval = median_hdci,
                     .width = 0.90, size = 0.25, linewidth = 0.25, alpha = 0.6,
                     shape = 16) +
  scale_x_date(date_labels = "%b %y") +
  scale_y_continuous(expand = c(0, 0)) +
  scale_colour_met_d("Egypt") + 
  labs(x = "Primary", y = "95% HDI", colour = "Site") + 
  theme(legend.position = "inside",
        legend.position.inside = c(0.75, 0.15))
ggsave(here("figs/fig-gp.jpg"), width = 6, height = 6, dpi = 600)

fits[[2]]$draws("gp_rho") |> mcmc_intervals()
