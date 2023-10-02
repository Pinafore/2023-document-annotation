library(ggplot2)
library(gridExtra)

ggplot(excluded_time_sheet, aes(x = minutesPassed, y = Purity, color = group)) +
    geom_smooth() +
    labs(title = "Time vs Purity", x = "Time (minutes)", y = "Purity") +
     theme_minimal()

ggplot(excluded_time_sheet, aes(x = minutesPassed, y = randIndex, color = group)) +
  geom_smooth() +
  labs(title = "Time vs ARI", x = "Time (minutes)", y = "ARI") +
  theme_minimal()

ggplot(excluded_time_sheet, aes(x = minutesPassed, y = NMI, color = group)) +
  geom_smooth() +
  labs(title = "Time vs ANMI", x = "Time (minutes)", y = "ANMI") +
  theme_minimal()

"Stack vertical plots for metrics"
p1 <- ggplot(DocsLabeled, aes(x = numDocsLabeled, y = purity, color = group)) +
  geom_smooth() +
  labs(title = "Num Documents Labeled vs Purity", x = "Num Documents Labeled", y = "Purity") +
  theme_minimal(aspect.ratio = 0.2)

p2 <- ggplot(DocsLabeled, aes(x = numDocsLabeled, y = randIndex, color = group)) +
  geom_smooth() +
  labs(title = "Num Documents Labeled vs ARI", x = "Num Documents Labeled", y = "ARI") +
  theme_minimal(aspect.ratio = 0.2)

p3 <- ggplot(DocsLabeled, aes(x = numDocsLabeled, y = NMI, color = group)) +
  geom_smooth() +
  labs(title = "Num Documents Labeled vs ANMI", x = "Num Documents Labeled", y = "ANMI") +
  theme_minimal(aspect.ratio = 0.2)

grid.arrange(p1, p2, p3, ncol=1)







p1 <- ggplot(DocsLabeled, aes(x = numDocsLabeled, y = purity, color = group)) +
  geom_smooth() +
  labs(title = "Num Documents Labeled vs Purity", x = "Num Documents Labeled", y = "Purity") +
  theme_minimal() +
  theme(aspect.ratio = 0.2)

p2 <- ggplot(excluded_time_sheet, aes(x = numDocsLabeled, y = randIndex, color = group)) +
  geom_smooth() +
  labs(title = "Num Documents Labeled vs ARI", x = "Num Documents Labeled", y = "ARI") +
  theme_minimal() +
  theme(aspect.ratio = 0.2)

p3 <- ggplot(excluded_time_sheet, aes(x = numDocsLabeled, y = NMI, color = group)) +
  geom_smooth() +
  labs(title = "Num Documents Labeled vs ANMI", x = "Num Documents Labeled", y = "ANMI") +
  theme_minimal() +
  theme(aspect.ratio = 0.2)


'''Docs labeled vs. metrics'''
grid.arrange(p1, p2, p3, ncol=1)


p1 <- ggplot(DocsLabeled, aes(x = numDocsLabeled, y = purity, color = group)) +
  geom_smooth(se = FALSE, alpha = 0.7, size = 1, linetype = "solid") +  # Setting se to FALSE removes the gray area. Adjust alpha, size, and linetype as needed
  labs(title = "Num Documents Labeled vs Purity", x = "Num Documents Labeled", y = "Purity")


p2 <- ggplot(DocsLabeled, aes(x = numDocsLabeled, y = randIndex, color = group)) +
  geom_smooth(se = FALSE, alpha = 0.7, size = 1, linetype = "solid") +  # Setting se to FALSE removes the gray area. Adjust alpha, size, and linetype as needed
  labs(title = "Num Documents Labeled vs ARI", x = "Num Documents Labeled", y = "ARI")


p3 <- ggplot(DocsLabeled, aes(x = numDocsLabeled, y = NMI, color = group)) +
  geom_smooth(se = FALSE, alpha = 0.7, size = 1, linetype = "solid") +  # Setting se to FALSE removes the gray area. Adjust alpha, size, and linetype as needed
  labs(title = "Num Documents Labeled vs ANMI", x = "Num Documents Labeled", y = "ANMI")

grid.arrange(p1, p2, p3, ncol=1, heights=c(1, 1, 1))

grid.arrange(p1, p2, p3, ncol=1)


# Install and load the necessary libraries
packages <- c("ggplot2", "gridExtra", "gtable", "grid")
new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)
lapply(packages, require, character.only = TRUE)

# Function to remove legends
no_legend <- function(p) p + theme(legend.position = "none")

# Function to add markers on the smoothed curve for every 50th document
add_markers_on_curve <- function(df, x_var, y_var, group_var) {
  # Create an empty data frame to store the points to mark
  marked_points <- data.frame()
  
  for (group in unique(df[[group_var]])) {
    # Filter data for each group
    group_data <- df[df[[group_var]] == group,]
    
    for (x_val in seq(50, max(group_data[[x_var]]), by = 50)) {
      # Predict y using loess for every 50th document
      pred_y <- predict(loess(formula = as.formula(paste0(y_var, " ~ ", x_var)), data = group_data), newdata = data.frame(numDocsLabeled = x_val))
      marked_points <- rbind(marked_points, data.frame(numDocsLabeled = x_val, y_val = pred_y, group = group))
    }
  }
  return(marked_points)
}

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


# Create the plots with markers on the smoothed curve at every 5th minute
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
p1 <- plot_with_markers(excluded_time_sheet, "minutesPassed", "Purity", "group", "Purity") + theme(axis.title.x = element_blank())
p2 <- plot_with_markers(excluded_time_sheet, "minutesPassed", "randIndex", "group", "ARI") + theme(axis.title.x = element_blank())
p3 <- plot_with_markers(excluded_time_sheet, "minutesPassed", "NMI", "group", "ANMI") 

# Extract the legend as a grob with the modified p3 object
legend_grob <- ggplotGrob(p3 + theme(legend.position="top", legend.direction="horizontal", legend.text=element_text(size=12)))$grobs[[which(ggplotGrob(p3)$layout$name == "guide-box")]]

# Arrange the plots with grid.arrange with the modified legend grob
grid.arrange(
  legend_grob,
  p1,
  p2,
  p3,
  ncol = 1, heights = c(0.1, 1, 1, 1)
)