# Load necessary libraries
library(data.table)
library(dplyr)
library(tidyr)
library(readr)
library(ggplot2)
library(reshape2)
library(gridExtra)
library(caTools)
library(rpart)
library(rpart.plot)
library(caret)

# Set working directory
setwd("/Users/chenweichen/Desktop/CBA Paper for students")

# Read the data
data1 <- fread("sleep.csv", stringsAsFactors = TRUE, header = TRUE)
head(data1)
summary(data1)

# Color settings
colours <- rgb(0.8, 0.1, 0.3, 0.6)

#-------------------------------------------------------------------------------
# Check and visualize outliers for Age
boxplot(data1$Age, axes = FALSE, staplewex = 1, main = "Age", col = colours)
text(y = boxplot.stats(data1$Age)$stats, labels = 
       boxplot.stats(data1$Age)$stats, x = 1.25)

# Check and visualize outliers for Caffeine consumption
boxplot(data1$`Caffeine consumption`, axes = FALSE, staplewex = 1,
        main = "Caffeine Consumption", col = colours)
text(y = boxplot.stats(data1$`Caffeine consumption`)$stats, 
     labels = boxplot.stats(data1$`Caffeine consumption`)$stats, x = 1.25)
# Remove outliers for Caffeine consumption above 100
data1 <- data1[data1$`Caffeine consumption` <= 100]

# Check and visualize outliers for Sleep duration
boxplot(data1$`Sleep duration`, axes = FALSE, staplewex = 1,
        main = "Sleep Duration", col = colours)
text(y = boxplot.stats(data1$`Sleep duration`)$stats, 
     labels = boxplot.stats(data1$`Sleep duration`)$stats, x = 1.25)
# Remove outliers for Sleep duration (e.g., below 5.5 or above 9)
data1 <- data1 %>% filter(`Sleep duration` >= 5.5 & `Sleep duration` <= 9)

# Check and visualize outliers for Sleep efficiency
boxplot(data1$`Sleep efficiency`, axes = FALSE, staplewex = 1,
        main = "Sleep Efficiency", col = colours)
text(y = boxplot.stats(data1$`Sleep efficiency`)$stats,
     labels = boxplot.stats(data1$`Sleep efficiency`)$stats, x = 1.25)

# Check and visualize outliers for REM sleep percentage
boxplot(data1$`REM sleep percentage`, axes = FALSE, staplewex = 1, 
        main = "REM Sleep Percentage", col = colours)
text(y = boxplot.stats(data1$`REM sleep percentage`)$stats, 
     labels = boxplot.stats(data1$`REM sleep percentage`)$stats, x = 1.25)

# Check and visualize outliers for Deep sleep percentage
boxplot(data1$`Deep sleep percentage`, axes = FALSE, staplewex = 1,
        main = "Deep Sleep Percentage", col = colours)
text(y = boxplot.stats(data1$`Deep sleep percentage`)$stats, 
     labels = boxplot.stats(data1$`Deep sleep percentage`)$stats, x = 1.25)

# Check and visualize outliers for Light sleep percentage
boxplot(data1$`Light sleep percentage`, axes = FALSE, staplewex = 1,
        main = "Light Sleep Percentage", col = colours)
text(y = boxplot.stats(data1$`Light sleep percentage`)$stats, 
     labels = boxplot.stats(data1$`Light sleep percentage`)$stats, x = 1.25)

# Check and visualize outliers for Daily Steps
boxplot(data1$`Daily Steps`, axes = FALSE, staplewex = 1, 
        main = "Daily Steps", col = colours)
text(y = boxplot.stats(data1$`Daily Steps`)$stats, 
     labels = boxplot.stats(data1$`Daily Steps`)$stats, x = 1.25)
Q1 <- quantile(data1$`Daily Steps`)[[2]]
Q3 <- quantile(data1$`Daily Steps`)[[4]]
LL <- max(Q1 - 1.5 * IQR(data1$`Daily Steps`), min(data1$`Daily Steps`))
UL <- min(Q3 + 1.5 * IQR(data1$`Daily Steps`), max(data1$`Daily Steps`))
data1 <- data1[data1$`Daily Steps` >= LL & data1$`Daily Steps` <= UL]

#-------------------------------------------------------------------------------
# 1. Relationship between Daily Steps and Deep Sleep Percentage
p1 <- ggplot(data1, aes(x = `Daily Steps`, y = `Deep sleep percentage`)) +
  geom_point() +
  geom_smooth(method = "lm", col = "blue") +
  labs(title = "Relationship between Daily Steps and Deep Sleep Percentage", x = "Daily Steps", y = "Deep Sleep Percentage")

# Relationship between Daily Steps and Deep Sleep Percentage - Boxplot
p2 <- ggplot(data1, aes(x = cut(`Daily Steps`, breaks = 5), y = `Deep sleep percentage`)) +
  geom_boxplot(fill = "blue", color = "black") +
  labs(title = "Deep Sleep Percentage by Daily Steps", x = "Daily Steps", y = "Deep Sleep Percentage")
print(p2)

# 2. Relationship between Sleep Duration and Sleep Efficiency
p3 <- ggplot(data1, aes(x = `Sleep duration`, y = `Sleep efficiency`)) +
  geom_point() +
  geom_smooth(method = "lm", col = "blue") +
  labs(title = "Relationship between Sleep Duration and Sleep Efficiency", x = "Sleep Duration (hours)", y = "Sleep Efficiency")

# Relationship between Sleep Duration and Sleep Efficiency - Violin Plot
p4 <- ggplot(data1, aes(x = as.factor(`Sleep duration`), y = `Sleep efficiency`)) +
  geom_violin(fill = "blue") +
  labs(title = "Sleep Efficiency by Sleep Duration", x = "Sleep Duration (hours)", y = "Sleep Efficiency")

# 3. Relationship between Exercise Frequency and Sleep Efficiency
p5 <- ggplot(data1, aes(x = `Exercise frequency`, y = `Sleep efficiency`)) +
  geom_point() +
  geom_smooth(method = "lm", col = "blue") +
  labs(title = "Relationship between Exercise Frequency and Sleep Efficiency", x = "Exercise Frequency (days per week)", y = "Sleep Efficiency")

# Relationship between Exercise Frequency and Sleep Efficiency - Bar Plot
p6 <- ggplot(data1, aes(x = as.factor(`Exercise frequency`), y = `Sleep efficiency`)) +
  stat_summary(fun = mean, geom = "bar", fill = "blue", color = "black") +
  labs(title = "Average Sleep Efficiency by Exercise Frequency", x = "Exercise Frequency (days per week)", y = "Average Sleep Efficiency")
print(p6)

# Correlation matrix heatmap
cor_matrix <- cor(data1 %>% select(`Sleep duration`, `Sleep efficiency`, `REM sleep percentage`, `Deep sleep percentage`, `Light sleep percentage`, `Awakenings`, `Caffeine consumption`, `Alcohol consumption`, `Exercise frequency`, `Daily Steps`), use = "complete.obs")
melted_cor_matrix <- melt(cor_matrix)

ggplot(melted_cor_matrix, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0, limit = c(-1, 1), space = "Lab", name="Correlation") +
  theme_minimal() + 
  labs(title = "Correlation Heatmap") +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, size = 12, hjust = 1))

# Combine plots
grid.arrange(p1, p2, ncol = 1)
grid.arrange(p3, p4, ncol = 1)
grid.arrange(p5, p6, ncol = 1)

#-------------------------------------------------------------------------------

# Define features and target variable
X <- data1 %>% select(`Alcohol consumption`, Age, `Exercise frequency`, `Caffeine consumption`, `Daily Steps`)
y <- data1$`Sleep efficiency`

# Split the data into training and testing sets
set.seed(2)
trainIndex <- createDataPartition(y, p = .7, list = FALSE, times = 1)
X_train <- X[trainIndex,]
X_test <- X[-trainIndex,]
y_train <- y[trainIndex]
y_test <- y[-trainIndex]

# Train and evaluate the CART model
cart_model <- rpart(y_train ~ ., data = X_train, method = 'anova', control = rpart.control(minsplit = 20, cp = 0.01))
printcp(cart_model)
plotcp(cart_model, main = "Subtrees")

cart_pred <- predict(cart_model, X_test)
cart_rmse <- sqrt(mean((cart_pred - y_test)^2))
cart_r2 <- cor(cart_pred, y_test)^2
cart_mae <- mean(abs(cart_pred - y_test))

cart_results <- data.frame(
  Model = "CART",
  RMSE = cart_rmse,
  MAE = cart_mae,
  R_squared = cart_r2
)

print(cart_results)

rpart.plot(cart_model, main = "CART Model - Sleep Efficiency")
printcp(cart_model)
plotcp(cart_model, main = "CART Model - Complexity Parameter Plot")

# Select the best cp value and prune the model
best_cp <- cart_model$cptable[which.min(cart_model$cptable[,"xerror"]), "CP"]
pruned_cart_model <- prune(cart_model, cp = best_cp)
rpart.plot(pruned_cart_model, main = "Pruned CART Model - Sleep Efficiency")
pruned_cart_pred <- predict(pruned_cart_model, X_test)

pruned_cart_rmse <- sqrt(mean((pruned_cart_pred - y_test)^2))
pruned_cart_mae <- mean(abs(pruned_cart_pred - y_test))
pruned_cart_r2 <- cor(pruned_cart_pred, y_test)^2

# Display pruned model results
pruned_cart_results <- data.frame(
  Model = "Pruned CART",
  RMSE = pruned_cart_rmse,
  MAE = pruned_cart_mae,
  R_squared = pruned_cart_r2
)
print(pruned_cart_results)

# Feature importance
importance <- varImp(pruned_cart_model, scale = FALSE)
print(importance)

#-------------------------------------------------------------------------------

# Train a linear regression model
linear_model <- lm(y_train ~ ., data = X_train)
linear_summary <- summary(linear_model)

# Extract coefficients
coefficients <- linear_summary$coefficients
coefficients_df <- as.data.frame(coefficients)

# Rename columns for better readability
colnames(coefficients_df) <- c("Estimate", "Std. Error", "t value", "Pr(>|t|)")

# Print the coefficients data frame
print(coefficients_df)



linear_pred <- predict(linear_model, X_test)

linear_rmse <- sqrt(mean((linear_pred - y_test)^2))
linear_mae <- mean(abs(linear_pred - y_test))
linear_r2 <- cor(linear_pred, y_test)^2
linear_results <- data.frame(
  Model = "Linear Regression",
  RMSE = linear_rmse,
  MAE = linear_mae,
  R_squared = linear_r2
)

print(linear_results)

