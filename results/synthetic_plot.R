# Loading the necessary library
install.packages("readr") # To install readr package only
install.packages("tidyverse")
library(readr) # To load readr package only
library(tidyverse) # To load the entire tidyverse collection

library(readxl)

# Reading the synthetic experiment file
congressional_results <- read_csv("./DocsLabeled.csv")

# Create the plots with markers on the smoothed curve at every 50th document
plot_with_markers <- function(plot_var, y_var, y_label) {
  # Get the marked points
  marked_points <- add_markers_on_curve(congressional_results, "numDocsLabeled", y_var, "group")
  
  # Create plot with markers on the smoothed curve
  p <- ggplot(congressional_results, aes(x = numDocsLabeled, y = get(plot_var), color = group)) +
    geom_smooth(se = TRUE, alpha = 0.7, size = 1, linetype = "solid") +
    geom_point(data = marked_points, aes(x = numDocsLabeled, y = y_val), alpha = 0.7, size = 3) +
    labs(title = NULL, x = "Number Documents Labeled", y = y_label) +
    theme(
      axis.title.x = element_text(size = rel(1.5)), # Increased x-axis label font size
      axis.title.y = element_text(size = rel(1.5))  # Increased y-axis label font size
    )  
  return(p)
}

# Create the plots
p1 <- plot_with_markers("purity", "purity", "Purity") + theme(axis.title.x = element_blank())
p2 <- plot_with_markers("randIndex", "randIndex", "ARI") + theme(axis.title.x = element_blank())
p3 <- plot_with_markers("NMI", "NMI", "ANMI") + guides(color=guide_legend(title=NULL))

# Extract the legend as a grob with the modified p3 object
legend_grob <- ggplotGrob(p3 + theme(legend.position="top", legend.direction="horizontal", legend.text = element_text(size = 12)))$grobs[[which(ggplotGrob(p3)$layout$name == "guide-box")]]

# Arrange the plots with grid.arrange with the modified legend grob
grid.arrange(
  legend_grob,
  no_legend(p1),
  no_legend(p2),
  no_legend(p3),
  ncol = 1, heights = c(0.1, 1, 1, 1)
)