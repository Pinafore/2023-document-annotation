# Set up the layout using the matrix
layout(layout_matrix)

# First Plot
boxplot(npmi ~ group, data = Coherence, main = "NPMI", ylab = "", xlab = "Group",
        col = c("lightcoral", "lightgreen", "plum"))
mtext("Coherence", side = 2, line = 3, font = 2) # Added Y-Axis title to the first plot

# Second Plot
boxplot(cv ~ group, data = Coherence, main = expression(bold(c[v])), ylab = "", xlab = "Group",
        col = c("lightcoral", "lightgreen", "plum"))

# Third Plot
boxplot(umass ~ group, data = Coherence, main = "UMass", ylab = "", xlab = "Group",
        col = c("lightcoral", "lightgreen", "plum"))

# Fourth Plot
boxplot(mentalChallenge ~ group, data=userRating, main="Mental Effort", 
        xlab="", ylab="", 
        col=c("#FF8080", "#CC99CC", "#99E699"))
mtext("Rating", side = 2, line = 3, font = 2) # Added Y-Axis title to the fourth plot

# Continue for other plots...
boxplot(confidence ~ group, data=userRating, main="Label Confidence", 
        xlab="", ylab="", 
        col=c("#FF8080", "#CC99CC", "#99E699"))

boxplot(topicRelevance ~ group, data=userRating, main="Keyword-Document Relevance", 
        xlab="", ylab="", 
        col=c("#FF8080", "#CC99CC", "#99E699"))

# Seventh Plot
boxplot(topicCoherence ~ group, data=userRating, main="Topic Keywords Coherence", 
        xlab="", ylab="", 
        col=c("#FF8080", "#CC99CC", "#99E699"))
mtext("Rating", side = 2, line = 3, font = 2) # Added Y-Axis title to the seventh plot

# Continue for other plots...
boxplot(topicDependence ~ group, data=userRating, main="Topic Keywords Dependence", 
        xlab="", ylab="", 
        col=c("#FF8080", "#CC99CC", "#99E699"))

boxplot(satisfaction ~ group, data=userRating, main="Satisfaction", 
        xlab="", ylab="", 
        col=c("#FF8080", "#CC99CC", "#99E699"))



# Boxplot for coherence


#boxplot(umass ~ group, data = Coherence, main = "UMass", ylab = "UMass", xlab = "Group",
  #      col = c("lightcoral", "lightgreen", "plum"))

#boxplot(cuci ~ group, data = Coherence, main = "UCI", ylab = "UCI", xlab = "Group",
       # col = c("lightcoral", "lightgreen", "plum"))
