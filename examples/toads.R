# load packages
if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, cmdstanr, loo, tidybayes, ggh4x)
source(here("sbc/util.R"))
options(mc.cores = 8)
theme_set(my_theme())

# prepare data
toads <- read_csv(here("analysis/toads.csv"))
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
mods <- list(cmdstan_model(here("stan/js.stan")),
             cmdstan_model(here("analysis/js0.stan")))
fits <- map(mods, 
            ~.$sample(stan_data, chains = getOption("mc.cores"),
                      iter_warmup = 200, iter_sampling = 500, refresh = 0,
                      show_exceptions = F))
map(fits, ~.$loo()) |> loo_compare()

# plot
map(fits, 
    ~gather_rvars(., beta[j], N[j])) |> 
  list_rbind(names_to = "mod") |> 
  ggplot(aes(factor(j, labels = dates) |> ymd(), ydist = .value)) + 
  facet_grid(.variable ~ factor(mod, labels = c("Intervals Accommodated", 
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
ggsave(here("figs/fig-toad.png"), width = 6, height = 5, dpi = 600)
