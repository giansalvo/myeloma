#
#https://rviews.rstudio.com/2017/09/25/survival-analysis-with-r/

library(survival)
library(ranger)
library(ggplot2)
library(dplyr)
library(ggfortify)
library(autoplotly)
library(survminer)
library(tidyverse) # function "%>%"
library("rms")

myeloma <- read.csv("myeloma_v6_survival_clean.csv", header=TRUE, sep = ";")
head(myeloma)


#############################################
# BARPLOTS
ggplot(myeloma, aes(x=race)) + geom_bar()
ggplot(myeloma, aes(x=sex)) + geom_bar()
ggplot(myeloma, aes(x=EMD)) + geom_bar()
ggplot(myeloma, aes(x=EMD_site)) + geom_bar()

#create pie chart
t <- table(myeloma$EMD_site)
pie(t)


##########################################
# Kaplan Meier Survival Curve
km <- with(myeloma, Surv(time, died))
head(km,80)
km_fit <- survfit(Surv(time, died) ~ 1, data=myeloma)
summary(km_fit, times = c(1, 30, 60, 90, 120, 150, 180))
ggsurvplot(km_fit, xlab = "months", pval = TRUE, conf.int = TRUE)

########################################
#Next, we look at survival curves by sex
km_sex_fit <- survfit(Surv(time, died) ~ sex, data=myeloma)
ggsurvplot(km_sex_fit, xlab = "months", pval = TRUE, conf.int = TRUE)

#perform log rank test
survdiff(Surv(time, died) ~ sex, data=myeloma)

########################################
#look at survival by age class 1
myel_AG1 <- mutate(myeloma, AG1 = ifelse((age < 60), "LT60", "OV60"),
              AG1 = factor(AG1))
km_AG_fit <- survfit(Surv(time, died) ~ AG1, data=myel_AG1)
ggsurvplot(km_AG_fit, xlab = "months", pval = TRUE, conf.int = TRUE)
#perform log rank test
survdiff(Surv(time, died) ~ AG1, data=myel_AG1)

########################################
#look at survival by age class 2
myel_AG2 <- mutate(myeloma, AG2 = ifelse((age < 65), "LT65", "OV65"),
                   AG2 = factor(AG2))
km_AG_fit <- survfit(Surv(time, died) ~ AG2, data=myel_AG2)
ggsurvplot(km_AG_fit, xlab = "months", pval = TRUE, conf.int = TRUE)
#perform log rank test
survdiff(Surv(time, died) ~ AG2, data=myel_AG2)

########################################
#look at survival by age class 3
myel_AG3 <- mutate(myeloma, AG3 = ifelse((age < 70), "LT70", "OV70"),
                   AG3 = factor(AG3))
km_AG_fit <- survfit(Surv(time, died) ~ AG3, data=myel_AG3)
ggsurvplot(km_AG_fit, xlab = "months", pval = TRUE, conf.int = TRUE)
#perform log rank test
survdiff(Surv(time, died) ~ AG3, data=myel_AG3)


########################################
#look at survival by race
km_race_fit <- survfit(Surv(time, died) ~ race, data=myeloma)
ggsurvplot(km_race_fit, xlab = "months", pval = TRUE, conf.int = TRUE)
#perform log rank test
survdiff(Surv(time, died) ~ race, data=myeloma)

########################################
#look at survival by EMD
km_EMD_fit <- survfit(Surv(time, died) ~ EMD, data=myeloma)
ggsurvplot(km_EMD_fit, xlab = "months", pval = TRUE, conf.int = TRUE)
#perform log rank test
survdiff(Surv(time, died) ~ EMD, data=myeloma)


#####################################
# Fit Cox Model
cox <- coxph(Surv(time, died) ~ sex + age_class1 + race + EMD, data = myeloma)
summary(cox)

ggforest(cox, data = myeloma)

cox_fit <- survfit(cox)
#plot(cox_fit, main = "cph model", xlab="Days")
autoplot(cox_fit)

# The plots show how the effects of the covariates change over time.
aa_fit <-aareg(Surv(time, died) ~ sex + age_class1 + race, 
               data = myeloma)
summary(aa_fit)  # provides a more complete summary of results
autoplot(aa_fit)

###########################################################
###########################################################
# NOMOGRAM
mod.cox <- cph(Surv(time, died) ~ sex + age_class1 + race + EMD, data = myeloma,  surv=TRUE)
ddist <- datadist(myeloma)
options(datadist='ddist')
surv.cox <- Survival(mod.cox)
nom.cox <- nomogram(mod.cox, 
  fun=list(function(x) surv.cox(1, x)),
  funlabel=c("Survival Probability"),
  lp=FALSE)
plot(nom.cox)
#plot(nom.cox, 
     fun.side=list(c(rep(c(1,3),5),1,1,1,1), 
                   c(1,1,1,rep(c(3,1)))))
print(nom.cox)

###########################################################
############################################################
#Random Forests Model
# ranger model
r_fit <- ranger(Surv(time, died) ~ sex + age_class1 + race + EMD,
                data = myeloma,
                mtry = 4,
                importance = "permutation",
                splitrule = "extratrees",
                verbose = TRUE)

# Average the survival models
death_times <- r_fit$unique.death.times 
surv_prob <- data.frame(r_fit$survival)
avg_prob <- sapply(surv_prob,mean)

# Plot the survival models for each patient

plot(r_fit$unique.death.times,r_fit$survival[1,], 
     type = "l", 
     ylim = c(0,1),
     col = "red",
     xlab = "Months",
     ylab = "survival",
     main = "Patient Survival Curves")

#
cols <- colors()
for (n in sample(c(2:dim(myeloma)[1]), 20)){
  lines(r_fit$unique.death.times, r_fit$survival[n,], type = "l", col = cols[n])
}
lines(death_times, avg_prob, lwd = 2)
legend(500, 0.7, legend = c('Average = black'))

vi <- data.frame(sort(round(r_fit$variable.importance, 4), decreasing = TRUE))
names(vi) <- "importance"
head(vi)
     
#############################################
#Finally, to provide an “eyeball comparison” of the three survival curves, I’ll plot them on the same graph.The following code pulls out the survival data from the three model objects and puts them into a data frame
# Set up for ggplot
kmi <- rep("KM",length(km_fit$time))
km_df <- data.frame(km_fit$time,km_fit$surv,kmi)
names(km_df) <- c("Time","Surv","Model")

coxi <- rep("Cox",length(cox_fit$time))
cox_df <- data.frame(cox_fit$time,cox_fit$surv,coxi)
names(cox_df) <- c("Time","Surv","Model")

rfi <- rep("RF",length(r_fit$unique.death.times))
rf_df <- data.frame(r_fit$unique.death.times,avg_prob,rfi)
names(rf_df) <- c("Time","Surv","Model")

plot_df <- rbind(km_df,cox_df,rf_df)

p <- ggplot(plot_df, aes(x = Time, y = Surv, color = Model))
p + geom_line()



