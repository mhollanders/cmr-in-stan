# load packages
if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, cmdstanr, loo, tidybayes, ggh4x)
source(here("sbc/util.R"))
cores <- 8
options(mc.cores = cores)
theme_set(my_theme())

# prepare data
toads <- read_csv(here("examples/data/toads.csv"))
dh <- as.matrix(toads[, -(1:3)])
dates <- mdy(colnames(dh))
J <- length(dates)
stan_data <- list(I = nrow(dh),
                  J = J,
                  tau = as.numeric(diff(dates, units = "weeks")), 
                  y = dh, 
                  I_aug = 250) |> 
  glimpse()



# fit and compare
mod <- cmdstan_model(here("examples/js-toads.stan"))
configs <- expand_grid(dirichlet = 0:1, intervals = 0:1) |> 
  mutate(mod = row_number())
fits <- map(configs$mod, ~{
  stan_data$dirichlet <- configs$dirichlet[.]
  stan_data$intervals <- configs$intervals[.]
  mod$sample(stan_data, refresh = 0, chains = cores, #threads_per_chain = 1,
             iter_warmup = 200, iter_sampling = 500, show_exceptions = F)
             # init = mod$pathfinder(stan_data, sig_figs = 14,
             #                       num_paths = cores, num_threads = cores))
})
loos <- map(fits, ~.$loo())
loo_compare(loos)

map(fits, ~gather_rvars(., mu, gamma)) |> 
  list_rbind(names_to = "mod") |> 
  left_join(configs) |> 
  ggplot(aes(xdist = .value, y = factor(mod) |> fct_rev())) + 
  facet_wrap(~ .variable, nrow = 1) + 
  stat_pointinterval()

fits[[4]]$draws("log_beta") |> mcmc_trace()
ggsave("trace-log-beta-cpp.png", width = 6, height = 6)

map(fits[1:2], ~spread_rvars(., log_alpha[j])) |> 
  list_rbind(names_to = "mod") |> 
  ggplot(aes(factor(j, labels = dates) |> ymd(),
             ydist = exp(log_alpha))) + 
  facet_wrap(~ mod, ncol = 1) + 
  stat_pointinterval()

map(fits, ~mcmc_trace(.$draws("alpha1")))

map(fits, ~gather_rvars(., log_beta[j], N[j])) |> 
  list_rbind(names_to = "mod") |> 
  left_join(configs) |> 
  ggplot(aes(factor(j, labels = dates) |> ymd(), 
             ydist = if_else(.variable == "log_beta",
                             exp(.value),
                             .value))) + 
  facet_grid(.variable ~ factor(dirichlet, labels = c("Logistic-Normal", "Dirichlet")),
             scales = "free_y") +
  stat_pointinterval(aes(colour = factor(intervals, labels = c("Ignored", "Accommodated"))),
                     point_interval = median_hdci, .width = 0.95,
                     size = 0.1, linewidth = 0.1,
                     position = position_dodge(width = 1)) + 
  scale_x_date(date_labels = "%b %y", 
               date_breaks = "3 week") + 
  facetted_pos_scales(y = list(scale_y_continuous(breaks = seq(0.2, 0.8, 0.2),
                                                  limits = c(0, 0.9),
                                                  expand = c(0, 0)),
                               scale_y_continuous(breaks = seq(50, 150, 50),
                                                  limits = c(0, 175),
                                                  expand = c(0, 0)))) +
  labs(x = "Survey", y = "95% HDI", colour = "Prior")

;map(fits, ~gather_rvars(., log_beta[j], N[j])) |> 
  list_rbind(names_to = "mod") |> 
  left_join(configs) |> 
  ggplot(aes(factor(j, labels = dates) |> ymd(), 
             ydist = .value)) + 
  facet_grid(.variable ~ factor(dir, labels = c("Logistic-Normal", "Dirichlet")),
             scales = "free_y") +
  stat_pointinterval(aes(colour = factor(intervals, labels = c("Ignored", "Accommodated"))),
                     point_interval = median_hdci, .width = 0.95,
                     size = 0.1, linewidth = 0.1,
                     position = position_dodge(width = 1)) + 
  scale_x_date(date_labels = "%b %y", 
               date_breaks = "3 week") + 
  facetted_pos_scales(y = list(scale_y_continuous(breaks = seq(-80, -20, 20),
                                                  limits = c(-90, 0), 
                                                  expand = c(0, 0)),
                               scale_y_continuous(breaks = seq(50, 150, 50),
                                                  limits = c(0, 175),
                                                  expand = c(0, 0)))) +
  labs(x = "Survey", y = "95% HDI", colour = "Intervals")
?list_rbind

fits[[3]]$draws("alpha1") |> mcmc_trace()


# fit and compare
mods <- list(cmdstan_model(here("stan/js6.stan")),
             cmdstan_model(here("stan/js2.stan")),
             cmdstan_model(here("examples/js0.stan")))
fits <- map(mods, 
            ~.$sample(stan_data, init = 0.1, chains = getOption("mc.cores"),
                      iter_warmup = 200, iter_sampling = 500, refresh = 0,
                      show_exceptions = F))
map(fits, ~.$loo()) |> loo_compare()


fit <- mods[[3]]$sample(stan_data, init = 0.1, chains = getOption("mc.cores"),
                        iter_warmup = 200, iter_sampling = 500, refresh = 0,
                        show_exceptions = F, seed = 1)
fit2 <- mods[[3]]$sample(stan_data, init = 0.1, chains = getOption("mc.cores"),
                        iter_warmup = 200, iter_sampling = 500, refresh = 0,
                        show_exceptions = F, seed = 1)

fits <- map(0:1, ~{
  stan_data$flag <- .
  mods[[3]]$sample(stan_data, init = 0.1, chains = getOption("mc.cores"),
                   iter_warmup = 200, iter_sampling = 500, refresh = 0,
                   show_exceptions = F, seed = 1)
})
map(fits, ~.$summary("lp__"))

fits[[1]]$draws("beta") |> mcmc_trace()
ggsave("trace-u.png", width = 8, height = 8)


wrap_plots(
  gather_rvars(fits[[3]], beta[j], N[j]) |> 
    mutate(mod = "Intervals Ignored") |> 
    ggplot(aes(factor(j, labels = dates) |> ymd(), ydist = .value)) + 
    facet_grid(.variable ~ mod,
               scales = "free_y") +
    stat_pointinterval(point_interval = median_hdci, .width = 0.95,
                       size = 0.1, linewidth = 0.1,
                       position = position_dodge(width = 1)) + 
    scale_x_date(date_labels = "%b %y", 
                 date_breaks = "3 week") + 
    facetted_pos_scales(y = list(scale_y_continuous(breaks = seq(0.2, 0.8, 0.2), 
                                                    limits = c(0, 0.9),
                                                    expand = c(0, 0)),
                                 scale_y_continuous(breaks = seq(50, 150, 50),
                                                    limits = c(0, 175), 
                                                    expand = c(0, 0)))) + 
    labs(x = "Survey", y = "95% HDI"),
  
  map(fits[1:2],
      ~gather_rvars(., beta[j], N[j])) |> 
    list_rbind(names_to = "prior") |>
    mutate(mod = "Intervals Accommodated") |> 
    ggplot(aes(factor(j, labels = dates) |> ymd(), ydist = .value)) + 
    facet_grid(.variable ~ mod, 
               scales = "free_y", 
               labeller = labeller(.variable = label_parsed)) +
    stat_pointinterval(aes(colour = factor(prior, labels = c("Dirichlet", "Logistic Normal"))),
                       point_interval = median_hdci, .width = 0.95,
                       size = 0.1, linewidth = 0.1,
                       position = position_dodge(width = 1)) + 
    scale_x_date(date_labels = "%b %y", 
                 date_breaks = "3 week") + 
    facetted_pos_scales(y = list(scale_y_continuous(breaks = seq(0.2, 0.8, 0.2), 
                                                    limits = c(0, 0.9),
                                                    expand = c(0, 0)),
                                 scale_y_continuous(breaks = seq(50, 150, 50),
                                                    limits = c(0, 175), 
                                                    expand = c(0, 0)))) + 
    labs(x = "Survey", y = "95% HDI", colour = "Prior") + 
    theme(legend.justification = c("right", "top"), legend.position = c(0.99, 0.99)),
  axis_titles = "collect"
)
?facet_grid2

facet_
  
# plot
map(fits,
    ~gather_rvars(., beta[j], N[j])) |> 
  list_rbind(names_to = "mod") |> 
  ggplot(aes(factor(j, labels = dates) |> ymd(), ydist = .value)) + 
  facet_grid(.variable ~ factor(mod, labels = c("Intervals Accommodated Dirichlet", 
                                                "Intervals Accommodated LN", 
                                                "Intervals Ignored")), 
             scales = "free_y", 
             labeller = labeller(.variable = label_parsed)) +
  stat_pointinterval(point_interval = median_hdci,
                     .width = 0.95,
                     size = 0.1, 
                     linewidth = 0.1) + 
  scale_x_date(date_labels = "%b %y", 
               date_breaks = "3 week") + 
  facetted_pos_scales(y = list(scale_y_continuous(breaks = seq(0.2, 0.8, 0.2), 
                                                  limits = c(0, 0.9),
                                                  expand = c(0, 0)),
                               scale_y_continuous(breaks = seq(50, 150, 50),
                                                  limits = c(0, 175), 
                                                  expand = c(0, 0)))) + 
  labs(x = "Survey", y = "95% HDI")

# plot
map(fits,
    ~gather_rvars(., beta[j], N[j])) |> 
  list_rbind(names_to = "mod") |> 
  mutate(prior = factor(case_when(mod == 1 ~ "Dirichlet", 
                                  mod == 2 ~ "Logistic-Normal",
                                  mod == 3 ~ "Dirichlet "),
                        levels = c("Dirichlet", "Logistic-Normal", "Dirichlet ")),
         mod = mod > 2) |> 
  ggplot(aes(factor(j, labels = dates) |> ymd(), ydist = .value)) + 
  facet_grid(.variable ~ factor(mod, labels = c("Intervals Accommodated",
                                                "Intervals Ignored")), 
             scales = "free_y", 
             labeller = labeller(.variable = label_parsed)) +
  stat_pointinterval(aes(colour = prior, alpha = mod), 
                     point_interval = median_hdci, .width = 0.95,
                     size = 0.1, linewidth = 0.1, 
                     position = position_dodge(width = 2)) + 
  scale_colour_manual(values = c("#dd5129", "#43b284", "#333333")) + 
  scale_alpha_manual(values = c(0.8, 1), guide = F) + 
  scale_x_date(date_labels = "%b %y", 
               date_breaks = "3 week") + 
  facetted_pos_scales(y = list(scale_y_continuous(breaks = seq(0.2, 0.8, 0.2), 
                                                  limits = c(0, 0.9),
                                                  expand = c(0, 0)),
                               scale_y_continuous(breaks = seq(50, 150, 50),
                                                  limits = c(0, 175), 
                                                  expand = c(0, 0)))) + 
  theme(legend.justification = c("right", "top"), 
        legend.position = c(0.99, 0.99)) + 
  labs(x = "Survey", y = "95% HDI", colour = "Prior")


ggsave(here("figs/fig-toad2.jpg"), width = 7, height = 6, dpi = 600)
