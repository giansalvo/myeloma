# 
#     Models for estimate the survival time period for Myeloma affected patients
# 
#     Copyright (c) 2023 Giansalvo Gusinu
# 
#     Permission is hereby granted, free of charge, to any person obtaining a
#     copy of this software and associated documentation files (the "Software"),
#     to deal in the Software without restriction, including without limitation
#     the rights to use, copy, modify, merge, publish, distribute, sublicense,
#     and/or sell copies of the Software, and to permit persons to whom the
#     Software is furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included in
#     all copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#     IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#     FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#     THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#     LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#     FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#     DEALINGS IN THE SOFTWARE.
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
plot(ds$year_diagnosis, ds$death, pch=16)

hist(ds$survival)
hist(ds$death)
hist(ds$age)
hist(ds$year_diagnosis)
hist(ds$death_specific)
hist(ds$diagnosis_to_treat)

boxplot(ds$survival)
boxplot(ds$death)
boxplot(ds$age)
boxplot(ds$year_diagnosis)
boxplot(ds$death_specific)
boxplot(ds$diagnosis_to_treat)

boxplot(ds$age ~ ds$death_specific)
boxplot(ds$death ~ ds$death_specific)

survival.glm<-glm(survival ~ age + death + death_specific + death_other, 
                  data = ds)

summary(survival.glm)

plot(survival.glm$residuals, pch = 16)

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

