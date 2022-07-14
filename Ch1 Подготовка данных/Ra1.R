
# import data -------------------------------------------------------------

if ( !exists("heart_failure_clinical_records_dataset_orig") ) {
  print("No data, tryin to load it")
  
  library(readr)
  
  heart_failure_clinical_records_dataset_orig <- 
    read_csv(
      "https://raw.githubusercontent.com/Alex-Samarkin/Reghression-Analysis-Book-2022/main/Ch1%20%D0%9F%D0%BE%D0%B4%D0%B3%D0%BE%D1%82%D0%BE%D0%B2%D0%BA%D0%B0%20%D0%B4%D0%B0%D0%BD%D0%BD%D1%8B%D1%85/heart_failure_clinical_records_dataset_orig.csv")
  
  View(heart_failure_clinical_records_dataset_orig)
} else {
  print("Found data, show it")
  
  View(heart_failure_clinical_records_dataset_orig)
}

# стандартная описательная статистика -------------------------------------

summary(heart_failure_clinical_records_dataset_orig)
s <- summary(heart_failure_clinical_records_dataset_orig)

# описательная статистика -------------------------------------------------

library(Hmisc)
d <- Hmisc::describe(heart_failure_clinical_records_dataset_orig)
h <- html(d)
write(h,file = "h.html")
browseURL("h.html")

# описательная статистика 2 -----------------------------------------------

library(pastecs)
pastecs::stat.desc(heart_failure_clinical_records_dataset_orig)
d1<-pastecs::stat.desc(heart_failure_clinical_records_dataset_orig)
h1<-html(d1,rownames = TRUE)

# описательная статистика 3 -----------------------------------------------

library(psych)
psych::describe(heart_failure_clinical_records_dataset_orig)
d2<-psych::describe(heart_failure_clinical_records_dataset_orig)


# графика -----------------------------------------------------------------

library(tidyverse)
dat <- heart_failure_clinical_records_dataset_orig %>% select(1,3,5,7:9,12)

plot(dat)

hist(dat[1:4])
hist(dat[5:7])

par(mfrow=c(1,1))
boxplot(dat,horizontal = TRUE)

for (n in names(dat)) {
  boxplot(dat[n],horizontal = TRUE)
  title(n)
}


# преобразование данных ---------------------------------------------------

library(car)
summary(p1 <- powerTransform(dat))
dat1 = bcPower(dat,p1$roundlam)
dat1 = as_data_frame(scale(dat1))

plot(dat1)

boxplot(dat1)


# корреляционный анализ ---------------------------------------------------

summary(cm<-stats::cor(dat1))
cm2 <- Hmisc::rcorr(as.matrix(dat1)) 
print(cm2)

# печать корреляционной матрицы -------------------------------------------

library(corrplot)
corrplot.mixed(cor(dat1))

library(correlation)
plot(correlation::correlation(dat1))

# выбросы -----------------------------------------------------------------

library(outliers)

dat3 <-dat1
for (i in (1:5))
{
  dat3 <- outliers::rm.outlier(dat3,fill=TRUE)  
}

boxplot(dat1)
boxplot(dat3)


# изолирующее дерево ------------------------------------------------------

Var1 <- dat1$`age^0`
Var2 <- dat1$`platelets^0.5`

ggplot(dat1, aes(x = Var1, y = Var2)) + 
  geom_point(shape = 1, alpha = 0.8) +
  labs(x = "age", y = "platelets") +
  labs(alpha = "", colour="Legend")

library(solitude)
iforest <- solitude::isolationForest$new()
iforest$fit(dat1)

dat1$pred <- iforest$predict(dat1)
dat1$outlier <-as.factor(ifelse(dat1$pred$anomaly_score > 0.63, "outlier", "normal"))

ggplot(dat1, aes(x = Var1, y = Var2, color = outlier)) + 
  geom_point(shape = 5, alpha = 0.9) +
  labs(x = "x", y = "y") +
  labs(alpha = "", colour="Legend")

# кластеры ----------------------------------------------------------------

dat1$pred<-NULL
dat1$outlier<-NULL

library(cluster)
summary(cl1 <- kmeans(dat1,2) )
clusplot(dat1, cl1$cluster )

for (i in (2:6)) {
  summary(cl1 <- kmeans(dat1,i) )
  clusplot(dat1, cl1$cluster, shade = TRUE, color = TRUE)
}
for (i in (2:6)) {
  summary(cl1 <- pam(dat1,i) )
  clusplot(dat1, cl1$cluster, shade = TRUE, color = TRUE)
}


# факторный анализ --------------------------------------------------------

dat1.scale <- scale(dat1)

fit <- princomp(dat1.scale, cor=TRUE)
summary(fit) # print variance accounted for
loadings(fit) # pc loadings
plot(fit,type="lines") # scree plot
head(fit$scores) # the principal components
biplot(fit)

library(psych)
fit1 <- principal(dat1.scale, nfactors=3, rotate="varimax")
fit1 # print results
plot(fit1$values,type="lines") # scree plot
biplot.psych(fit1)

fit3 <- factanal(dat1.scale, 3, rotation="varimax")
print(fit3, digits=2, cutoff=.3, sort=TRUE)
# plot factor 1 by factor 2
load <- fit3$loadings[,1:2]
plot(load,type="n") # set up plot
text(load,labels=names(dat1),cex=.7) # add variable names
