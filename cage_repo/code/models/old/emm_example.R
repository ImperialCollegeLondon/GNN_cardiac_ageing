require(emmeans)

pigs$issoy=(pigs$source=='soy')+0
pigs$extrav=(pigs$source=='soy')+runif(nrow(pigs))


lm1 <- lm(log(conc) ~ issoy + factor(percent), data = pigs)
emm1 <- emmeans(lm1, "issoy")
pairs(emm1)
summary(lm1)
pairs(emmeans(lm1, c("issoy", "percent")))



lm2 <- lm(log(conc) ~ source + factor(percent), data = pigs)
emm2 <- emmeans(lm2, "source")
pairs(emm2)
summary(lm2)
pairs(emmeans(lm2, c("source", "percent")))



lm3 <- lm(log(conc) ~ issoy + factor(percent) + extrav, data = pigs)
emm3 <- emmeans(lm3, "issoy")
pairs(emm3)
summary(lm3)
pairs(emmeans(lm3, c("issoy", "extrav"), at = list(extrav = c(0.5, -0.5))))


lm4 <- lm(log(conc) ~ issoy + factor(percent) + extrav + issoy*extrav, data = pigs)
emm4 <- emmeans(lm4, "issoy")
emm4
pairs(emm4)
summary(lm4)
pairs(emmeans(lm4, c("issoy", "extrav"), at = list(extrav = c(0.5, -0.5))))


require(rstatix)
emmeans_test(
  pigs,
  log(conc) ~ issoy + factor(percent),
)
