library(dplyr)
library(ggplot2)
library(rpart)
library(rpart.plot)
library(corrplot)
library(rsample)
library(e1071)
library(caret)
library(tidyr)
library(klaR)

set.seed(12345)
setwd("~/School/data101/finalproject/")
cirrhosis_data <- read.csv("cirrhosis.csv")

# ============================
# 1. Data Loading & Cleaning
# ============================

# remove ID column and N_Days column (not predictors)
cirrhosis_data <- subset(cirrhosis_data, select = -c(ID))
cirrhosis_data <- subset(cirrhosis_data, select = -c(N_Days))

# convert categorical data to factors with proper labels
cirrhosis_data$Status <- as.factor(cirrhosis_data$Status)
cirrhosis_data$Drug <- as.factor(cirrhosis_data$Drug)
cirrhosis_data$Sex <- as.factor(cirrhosis_data$Sex)
cirrhosis_data$Ascites <- as.factor(cirrhosis_data$Ascites)
cirrhosis_data$Hepatomegaly <- as.factor(cirrhosis_data$Hepatomegaly)
cirrhosis_data$Spiders <- as.factor(cirrhosis_data$Spiders)
cirrhosis_data$Edema <- as.factor(cirrhosis_data$Edema)

# remove instances where Status == CL, 
# excluding patients with liver transplants 
# will focus purely on outcomes of death vs. censored
cirrhosis_data <- cirrhosis_data %>% 
  filter(Status != "CL") %>%
  droplevels()

# overview of dataset
str(cirrhosis_data)
summary(cirrhosis_data)
head(cirrhosis_data)
table(cirrhosis_data$Status)

# ensure there are no missing values that need handling
colSums(is.na(cirrhosis_data))

# drop the last 100 rows where there is a large amount of missingness (when Drug == NA)
cirrhosis_data <- cirrhosis_data[!is.na(cirrhosis_data$Drug), ]
colSums(is.na(cirrhosis_data))

# impute missing data using the mean
cirrhosis_data <- cirrhosis_data %>%
  mutate(
    Cholesterol = ifelse(is.na(Cholesterol), mean(Cholesterol, na.rm = TRUE), Cholesterol),
    Tryglicerides = ifelse(is.na(Tryglicerides), mean(Tryglicerides, na.rm = TRUE), Tryglicerides),
    Copper = ifelse(is.na(Copper), mean(Copper, na.rm = TRUE), Copper),
    Platelets = ifelse(is.na(Platelets), mean(Platelets, na.rm = TRUE), Platelets)
  )
colSums(is.na(cirrhosis_data))
table(cirrhosis_data$Status)

# =========
# 2. Plots
# =========

# Status distribution
ggplot(cirrhosis_data, aes(x = Status)) + 
  geom_bar(fill = "steelblue") + 
  labs(title = "Distribution of Patient Status")

# Age distribution
age_years = cirrhosis_data$Age / 365
ggplot(cirrhosis_data, aes(x = age_years)) +
  geom_histogram(bins = 30, fill = "darkgreen") +
  labs(title = "Age Distribution")

# Bilirubin vs. Status
ggplot(cirrhosis_data, aes(x = Status, y = Bilirubin)) +
  geom_boxplot(fill = "tomato") +
  labs(title = "Bilirubin by Patient Status")

# Albumin vs. Status
ggplot(cirrhosis_data, aes(x = Status, y = Albumin)) +
  geom_boxplot(fill = "skyblue") +
  labs(title = "Albumin Levels by Status")

# Copper vs. Status
ggplot(cirrhosis_data, aes(x = Status, y = Copper)) +
  geom_boxplot(fill = "salmon") +
  labs(title = "Copper Levels by Status")

# Platelets vs. Status
ggplot(cirrhosis_data, aes(x = Status, y = Platelets)) +
  geom_boxplot(fill = "lightgreen") +
  labs(title = "Platelets by Status")

# =========================
# 3. Correlation Analysis
# =========================

num_vars <- cirrhosis_data[, sapply(cirrhosis_data, is.numeric)]

corrplot(cor(num_vars), method = "color", tl.cex = 0.6)

# =====================
# 4. Train-Test Split
# =====================

split_obj <- initial_split(cirrhosis_data, prop = 0.8, strata = Status)
split_obj

train_data <- training(split_obj)
summary(train_data)

test_data <- testing(split_obj)
summary(test_data)

# ==============
# 5. Modeling
# ==============

# Decision tree
tree_model <- rpart(Status~., data = train_data, method = "class", control=rpart.control(cp=0.001))
rpart.plot(tree_model)
pred_tree <- predict(tree_model, test_data, type = "class")
confusionMatrix(pred_tree, test_data$Status) # Accuracy: 0.8475

# Naive Bayes
nb_model <- naiveBayes(Status~., data = train_data)
pred_nb <- predict(nb_model, test_data)
confusionMatrix(pred_nb, test_data$Status) # Accuracy: 0.8136

# k-fold cross validation, k=10
ctrl <- trainControl(method = "cv", number = 10, savePredictions = TRUE)
tree_cv <- train(Status~., data = train_data, method = "rpart", trControl = ctrl, tuneLength = 20)
print(tree_cv) # best cp value = 0.09263158 with accuracy 0.7442029 

nb_cv <- train(Status~., data = train_data, method = "nb", trControl = ctrl, tuneLength = 10)
print(nb_cv) 

# adjust grid to avoid probabilty = 0 from cross-validation
grid <- expand.grid(fL = 1, usekernel = FALSE, adjust = c(1))

# since there are still errors, apply log transformation to the numeric variables
num_vars <- c("Age", "Bilirubin", "Cholesterol", "Albumin", "Copper", "Alk_Phos", "SGOT", "Tryglicerides", "Platelets", "Prothrombin", "Stage")
train_data[num_vars] <- log1p(train_data[num_vars])
test_data[num_vars] <- log1p(test_data[num_vars])

# rerun cross-validation again with grid and log transformation
nb_cv_logged <- train(Status~., data = train_data, method = "nb", trControl = ctrl, tuneGrid = grid, tuneLength = 10)

# remake decision tree and naive baiyes with cross validation
new_tree <- rpart(Status~.,data=train_data,method="class",control=rpart.control(cp=0.09263158))
rpart.plot(new_tree)
pred_tree <- predict(new_tree, test_data, type = "class")
confusionMatrix(pred_tree, test_data$Status)

pred_nb_cv <- predict(nb_cv_logged, test_data)
confusionMatrix(pred_nb_cv, test_data$Status)

acc_nb <- confusionMatrix(pred_nb_cv, test_data$Status)$overall["Accuracy"]
acc_tree <- confusionMatrix(pred_tree, test_data$Status)$overall["Accuracy"]

results <- data.frame(Model = c("Naive Bayes", "Decision Tree"),Accuracy = c(acc_nb, acc_tree))

# Bar plot of cross-validated model accuracies 
ggplot(results, aes(x = Model, y = Accuracy)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  ylim(0, 1) +
  geom_text(aes(label = sprintf("%.2f", Accuracy)), vjust = -0.5, size = 4.5) +
  labs(title = "Model Accuracy Comparison", y = "Accuracy", x = "Model") +
  theme_minimal()

# cp == 0.001 decision tree accuracy
results$Accuracy[results$Model == "Decision Tree"] <- 0.8475

# Bar plot of best model accuracies 
ggplot(results, aes(x = Model, y = Accuracy)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  ylim(0, 1) +
  geom_text(aes(label = sprintf("%.2f", Accuracy)), vjust = -0.5, size = 4.5) +
  labs(title = "Model Accuracy Comparison", y = "Accuracy", x = "Model") +
  theme_minimal()

print(results)
