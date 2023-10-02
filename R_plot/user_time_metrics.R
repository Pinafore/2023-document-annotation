

plot_with_markers <- function(data, x_var, y_var, group_var, y_label) {
  marked_points <- subset(data, as.integer(get(x_var)) %% 5 == 0)
  
  p <- ggplot(data, aes_string(x = x_var, y = y_var, color = group_var)) +
    geom_smooth(se = TRUE, alpha = 0.7, size = 1, linetype = "solid") +
    geom_point(data = marked_points, aes_string(x = x_var, y = y_var), alpha = 0.7, size = 3) +
    labs(title = NULL, x = "Minutes Elapsed", y = y_label) +
    theme(
      axis.title.x = element_text(size = rel(1.5)), # Adjust the size here
      axis.title.y = element_text(size = rel(1.5)),
      legend.position="none")  # Remove legend
  return(p)
}

# Create the plots
p1_2 <- plot_with_markers(excluded_time_sheet, "minutesPassed", "Purity", "group", "Purity") + theme(axis.title.x = element_blank())
p2_2 <- plot_with_markers(excluded_time_sheet, "minutesPassed", "randIndex", "group", "ARI") + theme(axis.title.x = element_blank())
p3_2 <- plot_with_markers(excluded_time_sheet, "minutesPassed", "NMI", "group", "ANMI") 

# Extract the legend as a grob with the modified p3 object
legend_grob <- ggplotGrob(p3 + theme(legend.position="top", legend.direction="horizontal", legend.text=element_text(size=12)))$grobs[[which(ggplotGrob(p3)$layout$name == "guide-box")]]

# Arrange the plots with grid.arrange with the modified legend grob
grid.arrange(
  legend_grob,
  p1_2,
  p2_2,
  p3_2,
  ncol = 1, heights = c(0.1, 1, 1, 1)
)