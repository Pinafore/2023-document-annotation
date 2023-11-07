library(gridExtra)
# Find the global minimum and maximum y values across all datasets
global_y_min <- min(min(DocsLabeled$purity), min(DocsLabeled$randIndex), min(DocsLabeled$NMI))  # Replace with your actual datasets and variable names
global_y_max <- max(max(DocsLabeled$purity), max(DocsLabeled$randIndex), max(DocsLabeled$NMI))  # Replace with your actual datasets and variable names

# Define ylim within each plot_with_markers call to set the y-axis limits
plot_with_markers <- function(data, plot_var, y_var, y_label) {
  marked_points <- subset(data, numDocsLabeled %% 10 == 0)
  p <- ggplot(data, aes_string(x = "numDocsLabeled", y = plot_var, color = "group")) +
    geom_smooth(se = TRUE, level = 0.5, alpha = 0.7, size = 1, linetype = "solid") +
    geom_point(data = marked_points, aes_string(x = "numDocsLabeled", y = plot_var), alpha = 0.7, size = 3) +
    labs(title = NULL, x = "Number Documents Labeled", y = y_label) +
    theme(
      axis.title.x = element_text(size = rel(1.5)),
      axis.title.y = element_text(size = rel(1.5))
    ) +
    ylim(global_y_min, global_y_max)  # set the same y-axis limits for all plots
  return(p)
}

# Create the plots using this modified function and arrange them as before.
# ...Continue from the previous modification to plot_with_markers function...

# After modifying the plot_with_markers function, create the plots using the modified function.

# Your left column plots
p1_2 <- plot_with_markers(excluded_time_sheet, "minutesPassed", "Purity", "group", "Purity") + theme(axis.title.x = element_blank(), legend.position = "none") + ylim(0.2, global_y_max)
p2_2 <- plot_with_markers(excluded_time_sheet, "minutesPassed", "randIndex", "group", "ARI") + theme(axis.title.x = element_blank(), legend.position = "none") + ylim(global_y_min, 0.3)
p3_2 <- plot_with_markers(excluded_time_sheet, "minutesPassed", "NMI", "group", "ANMI") + theme(legend.position = "none") + ylim(global_y_min, 0.5)

# Your right column plots
p1_1 <- plot_with_markers(DocsLabeled, "purity", "purity", "Purity") + theme(axis.title.x = element_blank(), legend.position = "none") + ylab(NULL) + ylim(0.2, global_y_max)
p2_1 <- plot_with_markers(DocsLabeled, "randIndex", "randIndex", "ARI") + theme(axis.title.x = element_blank(), legend.position = "none") + ylab(NULL) + ylim(global_y_min, 0.3)
p3_1 <- plot_with_markers(DocsLabeled, "NMI", "NMI", "ANMI") + theme(legend.position = "none") + ylab(NULL) + ylim(global_y_min, 0.5)

# Define one plot with a legend for extracting the legend grob.
p_legend <- plot_with_markers(DocsLabeled, "NMI", "NMI", "ANMI") + theme(legend.position = "top") + ylim(global_y_min, global_y_max)

# Extract the legend grob
legend_grob <- gtable_filter(ggplotGrob(p_legend), "guide-box")

# Arrange your plots with the legend centered at the top
grid.arrange(
  arrangeGrob(p1_2, p2_2, p3_2, ncol = 1),
  arrangeGrob(p1_1, p2_1, p3_1, ncol = 1),
  ncol = 2,
  top = legend_grob
)
