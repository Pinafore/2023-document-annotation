'''Mental Effort
res.aov <- aov(mentalEffort ~ group, data = userRating)

summary(res.aov)

TukeyHSD(res.aov)
'''

'''confidence'''
res.aov <- aov(confidence ~ group, data = userRating4)

summary(res.aov)

TukeyHSD(res.aov)


'''Topic Coherence'''
#res.aov <- aov(topicCoherence ~ group, data = userRating4)
res.aov <- aov(topicCoherence ~ group, data = userRating)
summary(res.aov)

TukeyHSD(res.aov)


'''Highlight dependence'''
#res.aov <- aov(highlightDependence ~ group, data = userRating4)
res.aov <- aov(highlightDependence ~ group, data = userRating)
summary(res.aov)

TukeyHSD(res.aov)




'''Topic Dependence'''
res.aov <- aov(topicDependence ~ group, data = userRating)

summary(res.aov)

TukeyHSD(res.aov)

'''Satisfaction'''
res.aov <- aov(satisfaction ~ group, data = userRating4)

summary(res.aov)

TukeyHSD(res.aov)



# Coherence
res.aov <- aov(npmi ~ group, data = Coherence2)

summary(res.aov)

TukeyHSD(res.aov)
