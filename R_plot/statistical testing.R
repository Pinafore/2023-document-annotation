res.aov <- aov(topicCoherence ~ group, data = userRating)

summary(res.aov)

TukeyHSD(res.aov)


