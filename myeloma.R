#load packages
#install.packages("car")
library(car)
#install.packages("ggeffects")
library(ggeffects)
#install.packages("sjPlot")
library('sjPlot')

ds <- read.table("myeloma_work_R.csv", header = T, sep = ";")

head(ds, c(3, 7))

plot(ds$survival, ds$age)
plot(ds$year_diagnosis, ds$death, pch=16, col = "blue")

hist(ds$survival)

boxplot(ds$age)
boxplot(ds$death)

boxplot(ds$age ~ ds$death_specific)
boxplot(ds$death ~ ds$death_specific)


survival.glm<-glm(survival ~ age + death + death_specific + death_other, 
                  data = ds)


summary(survival.glm)

plot(survival.glm$residuals, pch = 16, col = "red")

anova(survival.glm)


#########
#produce added variable plots
avPlots(survival.glm)

#############################
fit3=lm(survival~age*death*year_diagnosis,data=ds)
summary(fit3)
ggpredict(fit3, interactive = TRUE)

#############################

set.seed(0)

ds$survival <- ds$age * ds$death * ds$year_diagnosis
ds$dep2 <- ds$age * ds$death * ds$year_diagnosis * runif(2067)
ds$dep3 <- ds$age * ds$death * ds$year_diagnosis * abs(rnorm(2067))

model1 <- lm(data = ds, survival ~ age + death + year_diagnosis)

library('ggplot2')
library('ggeffects')

ggplot(ggpredict(model1, terms = c("age [1,5,10]", "death", "year_diagnosis")), 
       aes(x, predicted, color = group)) + geom_line() + facet_wrap(~facet)


########################
model1 <- lm(data = ds, survival ~ age + death + year_diagnosis)
plot_model(model1, type = 'diag')
