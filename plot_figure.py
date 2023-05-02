import pandas as pd
from plotnine import ggplot, aes, geom_line, theme_minimal

# Example dataset
data = {
    'iterations': [1, 2, 3, 4, 5],
    'accuracy': [0.6, 0.7, 0.8, 0.85, 0.9]
}

df = pd.DataFrame(data)

# Create a line plot of training iterations vs. accuracy
accuracy_plot = (
    ggplot(df, aes(x='iterations', y='accuracy'))
    + geom_line()
    + theme_minimal()
)

# Print the plot to the console
print(accuracy_plot)

# Save the plot to a file
accuracy_plot.save('accuracy_plot.png', dpi=300, width=6, height=4)
