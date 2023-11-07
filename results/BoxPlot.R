# Set up the layout using the matrix
layout(layout_matrix)
par(mfrow=c(2,3))

# First Plot
#boxplot(npmi ~ group, data = Coherence, main = "NPMI", ylab = "", xlab = "Group",
#col = c("#FF8080", "#CC99CC", "#99E699"))
#mtext("Coherence", side = 2, line = 3, font = 2) # Added Y-Axis title to the first plot
# "#66B2FF"\
boxplot(npmi ~ group, data = Coherence4, main = "NPMI", ylab = "", xlab = "Group",
col = c("#FF8080","#66B2FF", "#CC99CC", "#99E699"))


# Fourth Plot
#boxplot(mentalChallenge ~ group, data=userRating, main="Mental Effort", 
        #xlab="", ylab="", 
       # col=c("#FF8080", "#CC99CC", "#99E699"))
# mtext("Rating", side = 2, line = 3, font = 2) # Added Y-Axis title to the fourth plot

# Continue for other plots...
boxplot(confidence ~ group, data=userRating4, main="Confidence", 
        xlab="", ylab="", 
        col = c("#FF8080","#FFD700", "#CC99CC", "#99E699"))


#boxplot(topicRelevance ~ group, data=userRating, main="Topic-Document Relevance", 
     #   xlab="", ylab="", 
      #  col=c("#FF8080", "#CC99CC", "#99E699", "#FFD700"))

boxplot(highlightDependence ~ group, data=userRating4, main="Highligh Reliance", 
        xlab="", ylab="", 
        col = c("#FF8080","#FFD700", "#CC99CC", "#99E699"))

#mtext("Rating", side = 2, line = 3, font = 2) # Added Y-Axis title to the fourth plot

# Seventh Plot
boxplot(topicCoherence ~ group, data=userRating4, main="Topic Keywords Coherence", 
        xlab="", ylab="", 
        col = c("#FF8080","#FFD700", "#CC99CC", "#99E699"))

#mtext("Rating", side = 2, line = 3, font = 2) # Added Y-Axis title to the seventh plot

# Continue for other plots...
boxplot(topicDependence ~ group, data=userRating4, main="Topic-Keywords Reliance", 
        xlab="", ylab="", 
        col = c("#FF8080","#FFD700", "#CC99CC", "#99E699"))


boxplot(satisfaction ~ group, data=userRating4, main="Satisfaction", 
        xlab="", ylab="", 
        col = c("#FF8080","#FFD700", "#CC99CC", "#99E699"))




# Boxplot for coherence

# First Plot
#boxplot(npmi ~ group, data = Coherence, main = "NPMI", ylab = "", xlab = "Group",
        #col = c("lightcoral", "lightgreen", "plum"))
#mtext("Coherence", side = 2, line = 3, font = 2) # Added Y-Axis title to the first plot

# Second Plot
#boxplot(cv ~ group, data = Coherence, main = expression(bold(c[v])), ylab = "", xlab = "Group",
      #  col = c("lightcoral", "lightgreen", "plum"))

# Third Plot
#boxplot(cuci ~ group, data = Coherence, main = "UCI", ylab = "", xlab = "Group",
      #  col = c("lightcoral", "lightgreen", "plum"))

#boxplot(umass ~ group, data = Coherence, main = "UMass", ylab = "UMass", xlab = "Group",
  #      col = c("lightcoral", "lightgreen", "plum"))

#boxplot(cuci ~ group, data = Coherence, main = "UCI", ylab = "UCI", xlab = "Group",
       # col = c("lightcoral", "lightgreen", "plum"))
