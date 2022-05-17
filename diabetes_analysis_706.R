library(ggplot2)
library(caret)
library(readr)
library(readxl)
library(reshape2)
library(car)
library(gmodels)
library(klaR)
library(Metrics)
library(e1071)


 ########################################################################
##########################################################################
###                                                                    ###
### Statistics 793                                                     ###
### Independent Study                                                  ###
###                                                                    ###
### Comparing different supervised learning methods                    ###
### in predicting Diabetes Mellitus in patients                        ###
###                                                                    ###
### (Logistic regression, Naive Bayes, and Support Vector Machines)    ###
### for binary classification)                                         ###
###                                                                    ###
##########################################################################
 ########################################################################



# load data and format features as necessary
data <- data.frame(read_excel("F:/datasets/diabetes_data/Diabetes_Classification.xlsx", 
  col_types = c("skip", "numeric", "numeric", "numeric", "numeric", "numeric", "text", "numeric", 
  "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "text", "skip", "skip")))

data$Diabetes <- ifelse(data$Diabetes=="No diabetes", 0, 1)
data$Diabetes <- as.factor(data$Diabetes)
data$Gender <- as.factor(data$Gender)




###################################
#                                 #  
# Brief Exploratory data analysis #
#                                 #
###################################



total_patients <- nrow(data)

diabetes_patients <- nrow(subset(data, Diabetes=="Diabetes"))
prevalance_total <-positive_patients/total_patients

total_males <- nrow(subset(data, Gender=="male"))
prevalence_males <- nrow(subset(data, Gender=="male" & Diabetes=="Diabetes")) / total_males
prevalence_males <- round(prevalence_males, 3)

total_females <- nrow(subset(data, Gender=="female"))
prevalence_females <- nrow(subset(data, Gender=="female" & Diabetes=="Diabetes")) / total_females
prevalence_females <- round(prevalence_females, 2)

avg_age <- round(mean(data$Age), 2)
avg_age_diabetes <- mean(subset(data, Diabetes=="Diabetes")$Age)
avg_age_diabetes <- round(average_age_diabetes, 2)

avg_female_age <- round(mean(subset(data, Gender=="female")$Age), 2)
avg_male_age <- round(mean(subset(data, Gender=="male")$Age), 2)

gender_data <- data.frame(Gender=c("Males", "Females"),
  Count=c(total_males, total_females), Percent=c("16%", "15%"))

ggplot(gender_data, mapping=aes(x=Gender, y=Count, group=Percent)) + 
  geom_bar(stat="identity", fill="steelblue") + 
  scale_fill_brewer(palette="Dark2") + 
  geom_text(aes(label=Percent), hjust=2, size=6, color="white") +
  coord_flip() +
  theme_bw() + 
  labs(title="Prevalence of diabetes diagnoses", y="Patient Count", x="") +
  theme(legend.position="none")


# Creating training and test sets; evaluating multi-collinearity using correlation matrix & heat-map graph 
cor_data <- round(cor(data[,-c(6,15)]), 2)
cor_data <- melt(cor_data)
ggplot(data=cor_data, aes(x=Var1, y=Var2, fill=value)) + geom_tile() +
scale_fill_gradient2(low = "dodgerblue4", high = "darkorchid4", mid = "white", midpoint = 0,
                     limit = c(-1,1), space = "Lab", name="Pearson\nCorrelation")


# Consolidated data by removing variables that are actually functions of others, (e.g. BMI vs. weight)
# Systolic and diastolic pressure's correlation was over 70%, so "pulse pressure" was created which is 
# their difference weight and waist were removed because they are an indicator of BMI and thus had 
# relatively high correlation Cholesterol(total cholesterol) and HDL ("good" cholesterol) were removed 
# as HDL.Chol is a ratio of both Waist and hip were removed since "waist.hip.ratio" is a ratio of both

heat_data <- data[,-c(1,3,6,8,10,11,12,13,15)]
heat_data$pulse_pressure <- data[1:nrow(data),"Systolic.BP"] - data[1:nrow(data),"Diastolic.BP"]
cor_heat_melted <-melt(cor(heat_data))
ggplot(data=cor_heat_melted, aes(x=Var1, y=Var2, fill=value)) + geom_tile() +
    scale_fill_gradient2(low = "dodgerblue4", high = "darkorchid4", mid = "white", 
    midpoint = 0, limit = c(-1,1), space = "Lab", name="Pearson\nCorrelation")
diabetes.data <- data[, -c(1,3,8,10,11,12,13)]




###########################################################
#                                                         #
#  Building logistic model k-fold cross validation (n=10) #
#                                                         #
###########################################################

set.seed(5678)
indices <- sample(total_patients, round(total_patients*.75))
training_set <- diabetes.data[indices,]
test_set <- diabetes.data[-indices,]

set.seed(5678)
full_model <- glm(Diabetes ~ ., data=training_set, family="binomial")
step_model1 <- step(full_model, direction="both")
step_model2 <- step(full_model, direction="backward")

logistic <- glm(Diabetes ~ Glucose+Age, data=training_set, family="binomial")
vif(logistic)
summary(logistic)
plot(logistic)
# Cholesterol, Glucose, and Age are the chosen features by the step function
# However, Cholesterol is not srehown as statistically significant (it's p-value is .08)




# creating auxiliary models to verify linearity between predictors and the log-odds of the response variable
# by verifying the interaction term of the predictor and its log isn't statistically significant
age.linearity <- glm(Diabetes ~ Age+Age*log(Age), data=training_set, family=binomial(link="logit"))
glucose.linearity <- glm(Diabetes ~ Glucose+Glucose*log(Glucose), data=training_set, family=binomial(link="logit"))
age.logodds <- age.linearity$linear.predictors 

# generating scatter plots and running Box-Tidwelll test between the log-odds and predictors to verify linearity
logodds <- logistic$linear.predictors
plot(logodds ~ training_set$Glucose)
plot(logodds ~ training_set$Age)
boxTidwell(logodds ~ training_set$Glucose)
boxTidwell(logodds ~ training_set$Age)
boxTidwell(logodds ~ training_set$Chol.HDL.ratio)

# generating plots for displaying linearity (or lack-thereof) between the response and the predictors
ggplot(data=training_set, mapping=aes(x=predict(logistic), y=training_set$Glucose)) + geom_point(alpha=.5) +
  geom_smooth(method="glm", se=FALSE, method.args=list(family=binomial), col="steelblue", lwd=1.5) + theme_bw() +
  labs(title="Linearity between Glucose and Log Odds", x="Log Odds", y="Glucose")

ggplot(data=training_set, mapping=aes(x=predict(logistic), y=training_set$Age)) + geom_point(alpha=.5) +
  geom_smooth(method="glm", se=FALSE, method.args=list(family=binomial), col="steelblue", lwd=1.5) + theme_bw() +
  labs(title="Linearity between Age and Log Odds", x="Log Odds", y="Age")


# excluding outliers(as per cook's distance graph) from training data to evaluate their impact on models
# removing the 3 closest outliers to the border of the cook's distance graph significantly improved the 
# AIC (logistic3: 134.28 to logistic2: 97.35)
outlier.rows <- which(rownames(training_set) %in% c("347","297","41"))
outlier <- which(rownames(training_set) %in% c("347", "297"))
pruned_training_set <- training_set[-outlier.rows,]
pruned_training_set2 <- training_set[-outlier,]
set.seed(5678)
logistic2 <- glm(Diabetes ~ Glucose+Age, data=pruned_training_set, family=binomial(link="logit"))
logistic3 <- glm(Diabetes ~ Glucose+Age, data=pruned_training_set2, family=binomial(link="logit"))

prob <- logistic$fitted.values
ggplot(data=training_set, mapping=aes(x=Glucose, y=Diabetes)) + geom_point(alpha=.5) +
  geom_smooth(method="glm", se=FALSE, method.args=list(family=binomial), col="steelblue", lwd=1.5)


# final logistic model chosen
logistic.model <- logistic2


# Training and validating using k-fold cross-validation (k=10)
set.seed(5678)
logistic.spec <- trainControl(method="cv", number=10, savePredictions="all", classProbs=F)

logistic.validation <- train(Diabetes ~ Glucose+Age,
  data=pruned_training_set, method="glm", family="binomial", trControl=logistic.spec)

# generating predictions and validating against the test set
logistic.prob <- predict(logistic.model, newdata=test_set, type="response")
predictions <- ifelse(logistic.prob > .5, 1, 0)
predictions <- as.factor(predictions)
confusionMatrix(test_set$Diabetes, predictions)

plot(logistic.model)
# Accuracy on test set: 92.86

mean(predictions==test_set$Diabetes)
# Accuracy on test set: 92.86



###################################################################
#                                                                 # 
# Building naive-bayes model using k-fold cross validation (n=10) # 
#                                                                 #
###################################################################

set.seed(5678)
bayes <- NaiveBayes(Diabetes ~., data=training_set)
bayes.predictions <- predict(bayes.model, training_set)
mean(bayes.predictions$class==training_set$Diabetes)

set.seed(5678)
bayes.model <- train(Diabetes ~., data=training_set, method="nb", trControl=trainControl("cv", number=10))

nb.predictions <- predict(bayes.model, test_set)
mean(nb.predictions == test_set$Diabetes)
# Accuracy on test set: 93.88

confusionMatrix(nb.predictions, test_set$Diabetes)
# Accuracy on test set: 93.88




###############################################################################
#                                                                             #
# Building support vector machine model using k-fold cross validation (n=10)  #
#                                                                             #
###############################################################################

set.seed(5678)
svm.model1 <- svm(Diabetes ~., data=training_set, kernel="linear", cost=10, scale=F)
p<-predict(svm.model1, newdata=test_set)
confusionMatrix(p, test_set$Diabetes)
#Accuracy on test set: 91.84


# svm model creation and generating predictions on test data
set.seed(5678)
svm.model <- train(Diabetes ~., data=training_set, method="svmPoly", trControl=trainControl("cv", number=10),
  preProcess=c("center", "scale"), tuneLength=4)

svm.predictions <- predict(svm.model, test_set)

confusionMatrix(svm.predictions, test_set$Diabetes)

mean(svm.predictions == test_set$Diabetes)
#Accuracy on test set: 89.8





