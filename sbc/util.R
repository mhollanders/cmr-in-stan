# ggplot theme
my_theme <- function(base_size = 10,
                     base_family = "", 
                     base_line_size = base_size / 20 / 2, 
                     base_rect_size = base_size / 20) {
  # set colour and line size
  my_black <- "#333333"
  half_line <- base_size / 2
  # theme
  theme_grey(base_size = base_size,
             base_family = base_family, 
             base_line_size = base_line_size,
             base_rect_size = base_rect_size) %+replace%
    # updates
    theme(axis.line = element_blank(), # element_line(colour = my_black),
          axis.text = element_text(colour = my_black, size = rel(0.9)),
          axis.title = element_text(colour = my_black, size = rel(1)),
          axis.ticks =  element_line(colour = my_black),
          legend.key = element_rect(fill = "white", colour = NA),
          legend.text = element_text(size = rel(0.9)),
          panel.background = element_rect(fill = NA, colour = NA),
          panel.border = element_rect(fill = NA, colour = my_black),
          panel.grid = element_blank(),
          plot.margin = margin(10, 10, 10, 10),
          plot.title = element_text(size = rel(1.1), hjust = 0, vjust = 1, margin = margin(b = half_line)),
          strip.background = element_rect(fill = my_black, colour = my_black, linewidth = base_size / 2), 
          strip.text = element_text(colour = "white", size = rel(0.9), margin = margin(rep(0.8 * half_line, 4))),
          complete = TRUE)
}

# transition rate matrix from vectors of mortality and transition rates
rate_matrix <- function(h, q) {
  S <- length(h)
  Sp1 <- S + 1 ; Sm1 = S - 1
  Q <- matrix(0, Sp1, Sp1)
  Q[1:S, Sp1] <- h
  q_s <- head(q, Sm1)
  Q[1, 1] <- -(h[1] + sum(q_s))
  Q[1, 2:S] <- q_s
  if (S > 2) {
    ss <- S
    for (s in 2:Sm1) {
      q_s <- q[ss:(ss + S - 2)]
      Q[s, 1:(s - 1)] <- head(q_s, s - 1)
      Q[s, s] <- -(h[s] + sum(q_s))
      Q[s, (s + 1):S] <- tail(q_s, S - s)
      ss <- ss + Sm1
    }
  }
  q_s <- tail(q, Sm1)
  Q[S, 1:Sm1] <- q_s
  Q[S, S] <- -(h[S] + sum(q_s))
  Q
}

# random categorical draws
rcat <- function(n, prob) {
  rmultinom(n, size = 1, prob) |>
    apply(2, \(x) which(x == 1))
}

# plot ECDF-diff and estimates plots together
plot_sbc <- function(sbc, ..., nrow = NULL, ncol = NULL) {
  patchwork::wrap_plots(
    plot_ecdf_diff(sbc, ...) +
      ggplot2::facet_wrap(~ group, 
                          nrow = nrow, 
                          ncol = ncol) + 
      ggplot2::theme(legend.position = "none"), 
    plot_sim_estimated(sbc, ...) +
      ggplot2::facet_wrap(~ variable, 
                          nrow = nrow, 
                          ncol = ncol, 
                          scales = "free"),
    ncol = 1
  )
}
