#
#https://rviews.rstudio.com/2017/09/25/survival-analysis-with-r/

library(survival)
library(ranger)
library(ggplot2)
library(dplyr)
library(ggfortify)

myeloma <- read.csv("myeloma_kaplan_clean.csv", header=TRUE, sep = ";")
head(myeloma)

# Kaplan Meier Survival Curve
km <- with(myeloma, Surv(time, died))
head(km,80)
km_fit <- survfit(Surv(time, died) ~ 1, data=myeloma)
summary(km_fit, times = c(1,30,60,90, 120))

#plot(km_fit, xlab="Days", main = 'Kaplan Meyer Plot') #base graphics is always ready
autoplot(km_fit)

#Next, we look at survival curves by sex
km_sex_fit <- survfit(Surv(time, died) ~ sex, data=myeloma)
autoplot(km_sex_fit)

#look at survival by age
#vet <- mutate(myeloma, AG = ifelse((age < 60), "LT60", "OV60"),
#              AG = factor(AG),
#              trt = factor(trt,labels=c("standard","test")),
#              prior = factor(prior,labels=c("N0","Yes")))

km_AG_fit <- survfit(Surv(time, died) ~ age_class, data=myeloma)
autoplot(km_AG_fit)

# Fit Cox Model
cox <- coxph(Surv(time, died) ~ sex + age_class + race + diagnosis + first_malignant + AJCC , data = myeloma)
summary(cox)

cox_fit <- survfit(cox)
#plot(cox_fit, main = "cph model", xlab="Days")
autoplot(cox_fit)

# The plots show how the effects of the covariates change over time.
aa_fit <-aareg(Surv(time, died) ~ sex + age_class + race, 
               data = myeloma)
summary(aa_fit)  # provides a more complete summary of results
autoplot(aa_fit)

###########################################################
############################################################
#Random Forests Model
# ranger model
r_fit <- ranger(Surv(time, died) ~ sex + age_class + race + diagnosis + first_malignant + AJCC,
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
for (n in sample(c(2:dim(vet)[1]), 20)){
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



