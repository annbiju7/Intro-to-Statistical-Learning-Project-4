# Question 1a

setwd("C:/ann/fall 2023/stat 4360/project 4")
# Load required libraries
library(caret)
library(boot)
library(MASS)
library(leaps)
library(glmnet)

# Read the wine dataset
wine <- read.delim("wine.txt", sep = "\t", header = TRUE)

linear_regression_model <- function(data) {
  lm(Quality ~ ., data = data)
}
# Create a function to perform LOOCV
loocv <- function(data, model_func) {
  n <- nrow(data)
  mse_values <- numeric(n)
  for (i in 1:n) {
    # Split the data into training and test subsets
    train_data <- data[-i, ]
    test_data <- data[i, ]
    # Fit the model on the training data
    model <- model_func(train_data)
    model
    # Predict on the test data
    predicted <- predict(model, test_data)
    # Calculate the MSE for the test point
    mse_values[i] <- mean((test_data$Quality - predicted)ˆ2)
  }
  return(mean(mse_values))
}
linear_regression_model

# Calculate LOOCV MSE for the linear regression model
loocv_mse <- loocv(wine, linear_regression_model)

# Print the LOOCV MSE for linear regression
cat("LOOCV MSE for Linear Regression: ", loocv_mse, "\n")

#1b

best_subset_selection <- function(data) {
  pred <- colnames(data)[-1]
  best_model <- NULL
  best_adj_r2 <- -Inf
  for (k in 1:length(pred)) {
    subsets <- regsubsets(Quality ~ ., data = data, nvmax = k)
    summary_subsets <- summary(subsets)
    adj_r2_values <- summary_subsets$adjr2
    if (max(adj_r2_values) > best_adj_r2) {
      best_adj_r2 <- max(adj_r2_values)
      best_model <- which(adj_r2_values == best_adj_r2)
    }
  }
  return(best_model)
}

# Find the best subset of predictors
best_model_indices <- best_subset_selection(wine)

# Extract the names of the best predictors
best_predictors <- colnames(wine)[-1][best_model_indices]

# Fit the best model using the selected predictors
best_model <- lm(Quality ~ ., data = wine[, c("Quality", best_predictors)])

# Calculate LOOCV MSE for the best model using the selected predictors
loocv_mse_best_model <- loocv(wine[, c("Quality", best_predictors)], linear_regression_model)

loocv_mse_best_model

#QUESTION 1c

forward_stepwise_selection <- function(data) {
  all_predictors <- colnames(data)[-1] # Exclude the response variable (Quality)
  n <- nrow(data)
  best_predictors <- character(0)
  best_adj_r2 <- -Inf
  current_model <- NULL
  for (k in 1:length(all_predictors)) {
    remaining_predictors <- setdiff(all_predictors, best_predictors)
    adj_r2_values <- numeric(length(remaining_predictors))
    for (i in 1:length(remaining_predictors)) {
      candidate_predictors <- c(best_predictors, remaining_predictors[i])
      model <- lm(Quality ~ ., data = data[, c("Quality", candidate_predictors)])
      adj_r2_values[i] <- summary(model)$adj.r.squared
    }
    best_candidate <- which.max(adj_r2_values)
    if (adj_r2_values[best_candidate] <= best_adj_r2) {
      break
    }
    best_adj_r2 <- adj_r2_values[best_candidate]
    best_predictors <- c(best_predictors, remaining_predictors[best_candidate])
    current_model <- lm(Quality ~ ., data = data[, c("Quality", best_predictors)])
  }
  return(current_model)
}
# Find the best model using forward stepwise selection
best_forward_stepwise_model <- forward_stepwise_selection(wine)
best_forward_stepwise_model

loocv_mse_best_forward_stepwise_model <- loocv(wine[, c("Quality", best_predictors)], linear_regression)
loocv_mse_best_forward_stepwise_model

#1d
backward_stepwise_selection <- function(data) {
  all_predictors <- colnames(data)[-1] # Exclude the response variable (Quality)
  best_predictors <- all_predictors
  current_model <- lm(Quality ~ ., data = data)
  best_adj_r2 <- summary(current_model)$adj.r.squared
  while (length(best_predictors) > 1) {
    adj_r2_values <- numeric(length(best_predictors))
    for (i in 1:length(best_predictors)) {
      predictors_to_remove <- setdiff(best_predictors, best_predictors[i])
      4
      model <- lm(Quality ~ ., data = data[, c("Quality", predictors_to_remove)])
      adj_r2_values[i] <- summary(model)$adj.r.squared
    }
    best_candidate <- which.max(adj_r2_values)
    if (adj_r2_values[best_candidate] <= best_adj_r2) {
      break
    }
    best_adj_r2 <- adj_r2_values[best_candidate]
    best_predictors <- setdiff(best_predictors, best_predictors[best_candidate])
    current_model <- lm(Quality ~ ., data = data[, c("Quality", best_predictors)])
  }
  return(current_model)
}
# Compute the test MSE for the best model using the selected predictors
loocv_mse_best_backward_stepwise_model <- loocv(wine[, c("Quality", best_predictors)], linear_regression)


#1e
X <- as.matrix(wine[, -1]) # Predictors
Y <- wine$Quality # Response variable
# Create a function to perform LOOCV with Ridge regression
ridge_regression_loocv <- function(X, Y) {
  n <- length(Y)
  mse_values <- numeric(n)
  for (i in 1:n) {
    # Create training and test data for LOOCV
    X_train <- X[-i, ]
    Y_train <- Y[-i]
    X_test <- X[i, , drop = FALSE]
    Y_test <- Y[i]
    # Fit Ridge regression with different lambda values
    lambdas <- 10ˆseq(-6, 6, length = 100)
    ridge_fit <- cv.glmnet(X_train, Y_train, alpha = 0, lambda = lambdas)
    # Find the lambda with the minimum cross-validated MSE
    best_lambda <- ridge_fit$lambda.min
    # Fit the Ridge regression model with the best lambda on the entire training set
    ridge_model <- glmnet(X_train, Y_train, alpha = 0, lambda = best_lambda)
    # Predict on the test data
    Y_pred <- predict(ridge_model, s = best_lambda, newx = X_test)
    # Calculate the MSE for the test point
    mse_values[i] <- mean((Y_test - Y_pred)ˆ2)

  }
  return(mean(mse_values))
}
# Calculate LOOCV MSE with Ridge regression
loocv_mse_ridge <- ridge_regression_loocv(X, Y)
loocv_mse_ridge

#1f
lasso_regression_loocv <- function(X, Y) {
  n <- length(Y)
  mse_values <- numeric(n)
  for (i in 1:n) {
    # Create training and test data for LOOCV
    X_train <- X[-i, ]
    Y_train <- Y[-i]
    X_test <- X[i, , drop = FALSE]
    Y_test <- Y[i]
    # Fit Lasso regression with different lambda values
    lambdas <- 10ˆseq(-6, 6, length = 100)
    lasso_fit <- cv.glmnet(X_train, Y_train, alpha = 1, lambda = lambdas)
    # Find the lambda with the minimum cross-validated MSE
    best_lambda <- lasso_fit$lambda.min
    # Fit the Lasso regression model with the best lambda on the entire training set
    lasso_model <- glmnet(X_train, Y_train, alpha = 1, lambda = best_lambda)
    # Predict on the test data
    Y_pred <- predict(lasso_model, s = best_lambda, newx = X_test)
    # Calculate the MSE for the test point
    mse_values[i] <- mean((Y_test - Y_pred)ˆ2)
  }
  return(mean(mse_values))
}
# Calculate LOOCV MSE with Lasso regression
loocv_mse_lasso <- lasso_regression_loocv(X, Y)


# 1g


# Create a data frame to store the test MSEs
test_mse_df <- data.frame(
  Model = c("Linear Regression", "Best Subset Selection", "Forward Stepwise Selection"),
  Test_MSE = c(loocv_mse, loocv_mse_best_model, loocv_mse_best_forward_stepwise_model)
)

# Print the summary
print(test_mse_df)

# QUESTION 2a

library(readr)
library(caret)
library(bestglm)

data <- read_csv('diabetes.csv')

splitIndex <- createDataPartition(data$Outcome, p = 0.7, list = FALSE, times = 1)
train_data <- data[splitIndex, ]
test_data <- data[-splitIndex, ]
# Now, create a data frame 'train_data' with both X and y
train_data <- (train_data)[,-8]
test_data <- (test_data)[,-8]

logistic_model_a <- glm(train_data$Outcome ~ ., data = train_data, family = binomial)
y_pred_a <- predict(logistic_model_a, newdata = test_data, type = "response")
test_error_a <- 1 - sum((y_pred_a >= 0.5) == test_data$Outcome) / nrow(test_data)
cat("Test error (Log - All Predictors):", test_error_a, "\n")

# b
best_model_b <- bestglm(train_data, IC = "AIC", family = binomial)

selected_predictors_b <- best_model_b$BestModel
#selected_predictors_b
best_model_b

logistic_model_b <- glm(train_data$Outcome ~ ., data =
                          (train_data)[,-train_data$SkinThickness], family = binomial)
y_pred_b <- predict(logistic_model_b, newdata = test_data, type = "response")
test_error_b <- 1 - sum((y_pred_b >= 0.5) == test_data$Outcome) / nrow(test_data)
cat("Test error (Best-Subset Selection - AIC):", test_error_b, "\n")

#c
best_model_c <- bestglm(train_data, IC = "AIC", family = binomial, method = "exhaustive")
best_model_c

logistic_model_c <- glm(train_data$Outcome ~ ., data = train_data[, -train_data$SkinThickness], family =y_pred_c <- predict(logistic_model_c, newdata = test_data, type = "response")
                        test_error_c <- 1 - sum((y_pred_c >= 0.5) == test_data$Outcome) / nrow(test_data)
                        cat("Test error (Forward Stepwise Selection - AIC):", test_error_c, "\n")
                        
#d
best_model_d <- bestglm(train_data, IC = "AIC", family = binomial, method = "exhaustive")

logistic_model_d <- glm(train_data$Outcome ~ ., train_data[, -train_data$SkinThickness], family = binomy_pred_d <- predict(logistic_model_d, newdata = test_data, type = "response")
                        test_error_d <- 1 - sum((y_pred_d >= 0.5) == test_data$Outcome) / nrow(test_data)
                        cat("Test error (Backward Stepwise Selection - AIC):", test_error_d, "\n")
                        
#e
alpha_values <- 0.1
lambda_values <- 10ˆseq(10, -2, by = -1)
                        # Prepare the data
                        X_train <- as.matrix(train_data[, -8]) # Exclude the Outcome column
                        Y_train <- train_data$Outcome
                        # Fit a ridge classification model with cross-validated lambda
                        ridge_model_e <- cv.glmnet(X_train, Y_train, alpha = alpha_values, lambda = lambda_values)
                        # Find the best lambda from cross-validation
                        best_lambda <- ridge_model_e$lambda.min
                        # Fit the final ridge classification model with the best lambda
                        final_ridge_model <- glmnet(X_train, Y_train, alpha = alpha_values, lambda = best_lambda)
                        # Predict on the test data
                        X_test <- as.matrix(test_data[, -8]) # Exclude the Outcome column
                        Y_test <- test_data$Outcome
                        y_pred_e <- predict(final_ridge_model, s = best_lambda, newx = X_test, type = "response")
                        # Calculate the test error rate
                        test_error_e <- 1 - sum((y_pred_e >= 0.5) == Y_test) / length(Y_test)
                        cat("Test error (Ridge classification):", test_error_e, "\n")
                        


#f

alpha_values <- 0.1
lambda_values <- 10ˆseq(10, -2, by = -1)

# Prepare the data
X_train <- as.matrix(train_data[, -8]) # Exclude the Outcome column
Y_train <- train_data$Outcome

# Fit a Lasso classification model with cross-validated lambda
lasso_model_f <- cv.glmnet(X_train, Y_train, alpha = alpha_values, lambda = lambda_values, family = "binomial")
# Find the best lambda from cross-validation

best_lambda <- lasso_model_f$lambda.min
# Fit the final Lasso classification model with the best lambda
final_lasso_model <- glmnet(X_train, Y_train, alpha = alpha_values, lambda = best_lambda, family = "binomial")
# Predict on the test data
X_test <- as.matrix(test_data[, -8]) # Exclude the Outcome column
Y_test <- test_data$Outcome
 y_pred_f <- predict(final_lasso_model, s = best_lambda, newx = X_test, type = "response")
 
# Calculate the test error rate
test_error_f <- 1 - sum((y_pred_f >= 0.5) == Y_test) / length(Y_test)
cat("Test error (Lasso classification):", test_error_f, "\n")
                                                   
#g

summary_data <- data.frame(
  Method = c('Best-Subset Selection (AIC)', 'Forward Stepwise Selection (AIC)',
             'Backward Stepwise Selection (AIC)'),
  Test_Error_Rate = c(test_error_b, test_error_c, test_error_d 
  )
)
print(summary_data)











