plot_with_markers <- function(data, x, y, group_col, y_label, hide_x=FALSE) {
  # Filtering the data to only include rows where the x column is divisible by 50
  marker_data <- data[data[[x]] %% 5 == 0,]
  
  p <- ggplot(data, aes_string(x=x, y=y, color=group_col)) + 
    geom_smooth(aes(group=interaction(data[[group_col]])), method='loess', se=FALSE) + 
    geom_point(data=marker_data, aes_string(size=group_col, shape=group_col), show.legend=FALSE, size=2) +  # Adjust size here
    labs(y = y_label) + 
    theme_minimal() + 
    theme(legend.position="none", axis.title.x=element_blank())
  
  if (hide_x) {
    p <- p + theme(axis.text.x=element_blank(), axis.ticks.x=element_blank())
  }
  return(p)
}





# Create the plots
p1_1 <- plot_with_markers(excluded_time_sheet, "minutesPassed", "Purity", "group", "Purity", hide_x=TRUE)
p2_1 <- plot_with_markers(excluded_time_sheet, "minutesPassed", "randIndex", "group", "ARI", hide_x=TRUE)
p3_1 <- plot_with_markers(excluded_time_sheet, "minutesPassed", "NMI", "group", "ANMI")

# Create the plots
p1_2 <- plot_with_markers(newsgroup_results, "numDocsLabeled", "purity", "group", "Purity", hide_x=TRUE)
p2_2 <- plot_with_markers(newsgroup_results, "numDocsLabeled", "randIndex", "group", "ARI", hide_x=TRUE)
p3_2 <- plot_with_markers(newsgroup_results, "numDocsLabeled", "NMI", "group", "ANMI")

header_1 <- textGrob("Bills", gp = gpar(fontsize = 14, col = "black"), hjust = 0.25, x=0.55, y=0.15)
header_2 <- textGrob("20newsgroup", gp = gpar(fontsize = 14, col = "black"), hjust = 0.15, x=0.35, y=0.15)

y_label_purity <- textGrob("Purity", gp = gpar(fontsize = 14, col = "black"), rot=90, hjust=0.5, vjust=0.5)
y_label_ari <- textGrob("ARI", gp = gpar(fontsize = 14, col = "black"), rot=90, hjust=0.5, vjust=0.5)
y_label_anmi <- textGrob("ANMI", gp = gpar(fontsize = 14, col = "black"), rot=90, hjust=0.5, vjust=0.5)

# Arrange plots and headers using grid.arrange
legend_grob_2 <- ggplotGrob(p3 + theme(legend.position="top", legend.direction="horizontal", legend.text=element_text(size=12)))$grobs[[which(ggplotGrob(p3)$layout$name == "guide-box")]]

# Arrange the plots, headers, y-labels, and legend using grid.arrange
y_label_purity_right <- textGrob("Purity", gp = gpar(fontsize = 14, col = "black"), rot=90, hjust=0.5, vjust=0.5)
y_label_ari_right <- textGrob("ARI", gp = gpar(fontsize = 14, col = "black"), rot=90, hjust=0.5, vjust=0.5)
y_label_anmi_right <- textGrob("ANMI", gp = gpar(fontsize = 14, col = "black"), rot=90, hjust=0.5, vjust=0.5)

# Adjust your grid.arrange() function
combined_plot <- grid.arrange(
  top = grid.arrange(legend_grob_2, ncol=1),
  grid.arrange(arrangeGrob(header_1, header_2, ncol=2), ncol=1),
  arrangeGrob(y_label_purity, p1_1, p1_2, y_label_purity_right, ncol=4, widths=c(1/15, 7/15, 7/15, 1/15)),
  arrangeGrob(y_label_ari, p2_1, p2_2, y_label_ari_right, ncol=4, widths=c(1/15, 7/15, 7/15, 1/15)),
  arrangeGrob(y_label_anmi, p3_1, p3_2, y_label_anmi_right, ncol=4, widths=c(1/15, 7/15, 7/15, 1/15)),
  bottom = textGrob("Number Documents Labeled", gp = gpar(fontsize = 14), hjust = 0.5),
  ncol = 1
)




library(grid)

# Set up a new page
grid.newpage()

# Split the viewport into two rows
# One for the legend (with a smaller height) and one for the plots
pushViewport(viewport(layout = grid.layout(2, 1, heights=c(1, 10))))

# Select the first (top) viewport for the legend
pushViewport(viewport(layout.pos.row=1, layout.pos.col=1))

# Draw the legend in the top viewport
grid.draw(legend_grob_2)

# Pop out to return to the main viewport
popViewport()

# Select the second (bottom) viewport for the plots
pushViewport(viewport(layout.pos.row=2, layout.pos.col=1))

# Draw the combined plots in the bottom viewport
grid.draw(combined_plot)

# Clean up by popping out of the plot viewport
popViewport()
