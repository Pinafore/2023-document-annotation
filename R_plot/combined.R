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



# Reading the synthetic experiment file
# congressional_results <- read_csv("./DocsLabeled.csv")

# Create the plots with markers on the smoothed curve at every 50th document
plot_with_markers <- function(data, plot_var, y_var, y_label) {
  # Get the marked points, here we are subsetting data to include only every 10th document
  marked_points <- subset(data, numDocsLabeled %% 10 == 0)
  
  # Create plot with markers on the smoothed curve
  p <- ggplot(data, aes_string(x = "numDocsLabeled", y = plot_var, color = "group")) +
    geom_smooth(se = TRUE, level = 0.5, alpha = 0.7, size = 1, linetype = "solid") +  # Reduced grey error region
    geom_point(data = marked_points, aes_string(x = "numDocsLabeled", y = plot_var), alpha = 0.7, size = 3) +
    labs(title = NULL, x = "Number Documents Labeled", y = y_label) +
    theme(
      axis.title.x = element_text(size = rel(1.5)),  # Increased x-axis label font size
      axis.title.y = element_text(size = rel(1.5))   # Increased y-axis label font size
    )
  return(p)
}

# Create the plots
# Create the plots without removing the legend from p3
p1_1 <- plot_with_markers(DocsLabeled, "purity", "purity", "Purity") + theme(axis.title.x = element_blank(), legend.position = "none")+ ylab(NULL)
p2_1 <- plot_with_markers(DocsLabeled, "randIndex", "randIndex", "ARI") + theme(axis.title.x = element_blank(), legend.position = "none")+ ylab(NULL)
p3_1 <- plot_with_markers(DocsLabeled, "NMI", "NMI", "ANMI") + guides(color=guide_legend(title=NULL)) + theme(legend.position = "none")+ ylab(NULL)

# Extract the legend as a grob with the modified p3 object
legend_grob <- ggplotGrob(p3 + theme(legend.position="top", legend.direction="horizontal", legend.text = element_text(size = 12)))$grobs[[which(ggplotGrob(p3)$layout$name == "guide-box")]]

# After extracting the legend, now you can remove the legend from p3
# p3 <- p3 + theme(legend.position = "none")

# Arrange the plots in two columns, with plots from the second group in the left column
# and plots from the first group in the right column
grid.arrange(
  arrangeGrob(p1_2, p2_2, p3_2, ncol = 1),
  arrangeGrob(p1_1, p2_1, p3_1, ncol = 1),
  ncol = 2
)