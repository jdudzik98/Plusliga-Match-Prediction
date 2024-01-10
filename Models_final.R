# Libraries and data load -------------------------------------------------
library(readr)
library(dplyr)
library(DataExplorer)
library(lmtest)
library(caret)
library(pROC)
library(randomForest)
library(corrplot)
library(glmnet)
library(ggplot2)
library(e1071)
library(LiblineaR)
library(here)
library(gridExtra)
library(pander)

# Read the data
df <- read_csv("Plusliga_data_for_model.csv", 
               col_types = cols(Year = col_character()))
df_numeric <- df[, sapply(df, is.numeric)]

plot_density(df_numeric)
summary(df)


# Data transformations ----------------------------------------------------


# Switch character columns to numeric
df <- df %>%
  mutate(across(c(Sets_ratio_table_host, Sets_ratio_table_guest, Points_ratio_table_host, Points_ratio_table_guest), ~ as.numeric(gsub(",", ".", .))))

# Handling NAs
df$Spectators <- replace(df$Spectators, is.na(df$Spectators), 0)
df$Relative_spectators <- replace(df$Relative_spectators, !is.finite(df$Relative_spectators), NA)
df$Relative_spectators <- replace(df$Relative_spectators, is.na(df$Relative_spectators), mean(df$Relative_spectators, na.rm = T))

df <- df %>%
  mutate(Match_time_of_season = case_when(
    Phase %in% c("play-off", "play-out") ~ "play-off",
    !is.na(Round_original) & Round_original < 6 ~ "Start of the season",
    !is.na(Round_original) & Round_original < 20 ~ "Mid-season",
    !is.na(Round_original) ~ "End of the season",
    TRUE ~ "other"
  ))

df <- df %>%
  select(-Phase, -Round_original)

df <- df %>%
  mutate(Time_Category = factor(Time_Category),
         Year = factor(Year),
         Match_time_of_season = factor(Match_time_of_season))

# Plotting variables ------------------------------------------------------


# Printing histograms

# Convert selected columns to factors
df[, c("Winner", "Time_Category", "Year", "Match_time_of_season", "Set_number")] <- lapply(df[, c("Winner", "Time_Category", "Year", "Match_time_of_season", "Set_number")], factor)
# Rename levels if necessary
levels(df$Winner) <- make.names(levels(df$Winner))

# Alternatively, you can convert the response variable to a factor and allow R to automatically create valid variable names:
df$Winner <- as.factor(df$Winner)

# Replace all the null values with 0s as we will now operate on differences
df <- replace(df, is.na(df), 0)

# Calculate the differences:
host_guest_cols <- sort(colnames(df)[grep("host|guest", colnames(df))])
other_cols <- df %>% select(-all_of(host_guest_cols))
df2 <- df[, c(names(other_cols))]
# Subtract guest columns from host columns and create new columns in df2

df2$diff_Current_position_table = df$Current_position_table_host - df$Current_position_table_guest
df2$diff_Matches_ratio_last_5 = df$Matches_ratio_last_5_host - df$Matches_ratio_last_5_guest
df2$diff_Season_points_to_matches = df$Season_points_to_matches_host - df$Season_points_to_matches_guest
df2$diff_Sets_ratio_table = df$Sets_ratio_table_host - df$Sets_ratio_table_guest
df2$diff_Points_ratio_table = df$Points_ratio_table_host - df$Points_ratio_table_guest
df2$diff_Current_serve_effectiveness = df$Current_host_serve_effectiveness - df$Current_guest_serve_effectiveness
df2$diff_Current_positive_reception_ratio = df$Current_host_positive_reception_ratio - df$Current_guest_positive_reception_ratio
df2$diff_Current_perfect_reception_ratio = df$Current_host_perfect_reception_ratio - df$Current_guest_perfect_reception_ratio
df2$diff_Current_negative_reception_ratio = df$Current_host_negative_reception_ratio - df$Current_guest_negative_reception_ratio
df2$diff_Current_attack_accuracy = df$Current_host_attack_accuracy - df$Current_guest_attack_accuracy
df2$diff_Current_attack_effectiveness = df$Current_host_attack_effectiveness - df$Current_guest_attack_effectiveness
df2$diff_Current_timeouts = df$Current_timeouts_host - df$Current_timeouts_guest
df2$diff_Current_challenges = df$Current_challenges_host - df$Current_challenges_guest


# Create a data frame with each MatchID and the majority class in that match
match_summary <- df2 %>%
  group_by(MatchID) %>%
  summarise(MajorityClass = names(which.max(table(Winner))))

# Perform stratified sampling on the match_summary
set.seed(396596)
match_ids_split <- createDataPartition(match_summary$MajorityClass, p = 0.7, list = FALSE, times = 1)

# Use these splits to create your training and testing sets
train_data <- df2 %>% filter(MatchID %in% match_summary$MatchID[match_ids_split])
test_data <- df2 %>% filter(MatchID %in% match_summary$MatchID[-match_ids_split])

par(mfrow= c(1,1))
# Correlation matrix:
cor_matrix <- cor(train_data[,c(5,6,8:14,16:28)])


corrplot(cor_matrix, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 100, 
         addCoef.col = "black", number.cex = 0.1, 
         diag = FALSE)
# Function to generate histograms for factors
library(ggplot2)

library(ggplot2)
library(gridExtra)

# Function to generate bar plots for factors
plot_bar <- function(factor_var, col_name) {
  ggplot(df, aes(x = factor_var)) +
    geom_bar(fill = "steelblue") +
    labs(title = paste("Bar plot of", col_name)) +
    theme_minimal()
}

# Subset factor columns and get column names
factor_columns <- names(df)[sapply(df, is.factor)]
col_names <- names(df)

# Create a list to store bar plots
plots <- lapply(factor_columns, function(col) {
  col_name <- col_names[col]
  plot_bar(df[[col]], col_name)
})

# Arrange plots in a 3x2 grid using grid.arrange
grid_plot <- do.call(grid.arrange, c(plots, ncol = 2))

# Convert the grid plot to markdown format
markdown_output <- paste0("```{r}\n", "print(grid_plot)\n```")

# Logistic Regression -----------------------------------------------------

# Modelling
formula <- Winner ~ Year + Time_Category + Spectators + Relative_spectators + diff_Current_position_table +
  diff_Matches_ratio_last_5 + diff_Season_points_to_matches + diff_Sets_ratio_table + diff_Points_ratio_table +
  Set_number + Current_point_difference + Current_set_difference + Max_point_difference_throughout_set +
  Min_point_difference_throughout_set + Max_point_difference_throughout_match + Min_point_difference_throughout_match +
  Running_net_crossings_average + diff_Current_serve_effectiveness + diff_Current_positive_reception_ratio +
  diff_Current_perfect_reception_ratio + diff_Current_negative_reception_ratio + diff_Current_attack_accuracy +
  diff_Current_attack_effectiveness + diff_Current_timeouts + diff_Current_challenges + Match_time_of_season

# 
# logit <- glm(formula,
#              # here we define type of the model
#              family =  binomial(link = "logit"),
#              data = train_data)
# logit %>% saveRDS(here("outputs", "logit.rds"))
logit <- readRDS("outputs/logit.rds")
summary(logit)

lrtest(logit)

logit_fitted <- predict.glm(logit,
                             type = "response")
ROC.train.logit <- roc(as.numeric(train_data$Winner == "TRUE."), 
                        logit_fitted)
logit_test_fitted <- predict.glm(logit, test_data, type = "response")

ROC.test.logit <- roc(as.numeric(test_data$Winner == "TRUE."), 
                       logit_test_fitted)
table(train_data$Winner,
      ifelse(logit_fitted > 0.5, # condition
             "TRUE.", # what returned if condition TRUE
             "FALSE.")) # what returned if condition FALSE

confusionMatrix(data = as.factor(ifelse(logit_fitted > 0.5, # condition
                                        "TRUE.", # what returned if condition TRUE
                                        "FALSE.")), 
                reference = as.factor(train_data$Winner))


# Compute ECE
compute_ece <- function(predicted_probs,  n_bins = 20) {
  
  observed_labels <- ifelse(test_data$Winner == "TRUE.", 1, 0)
  
  # Create bins for the predicted probabilities
  bins <- cut(predicted_probs, breaks = seq(0, 1, length.out = n_bins + 1), include.lowest = TRUE, labels = FALSE)
  
  # Compute the average predicted probability in each bin
  bin_avgs <- tapply(predicted_probs, bins, mean)
  
  # Compute the observed frequency in each bin
  bin_true <- tapply(observed_labels, bins, mean, default = 0)
  
  # Compute the absolute difference between the predicted probability and observed frequency in each bin
  bin_diffs <- abs(bin_avgs - bin_true)
  
  # Compute the number of instances in each bin
  bin_counts <- table(bins)
  
  # Compute ECE
  ece <- sum(bin_diffs * bin_counts) / length(predicted_probs)
  
  return(ece)
}

# Random Forest -----------------------------------------------------------

ctrl_cv5 <- trainControl(method = "cv", 
                          number =    5,
                          classProbs = T)

# set.seed(396596)
# randomforest4 <- randomForest(formula, 
#                               data = train_data,
#                               ntree = 2000,
#                               mtry = 3,
#                               maxnodes = 300,
#                               trControl = ctrl_cv10,
#                               nodesize = 1000)


#randomforest4 %>% saveRDS(here("outputs", "randomforest4.rds"))
randomforest4 <- readRDS("outputs/randomforest4.rds")


pred.train.randomforest4 <- predict(randomforest4, 
                                    train_data, 
                                    type = "prob")[,2]

pred.test.randomforest4 <- predict(randomforest4, 
                                   test_data, 
                                   type = "prob")[,2]

# XGB Model ---------------------------------------------------------------


ctrl_cv5 <- trainControl(method = "repeatedcv", repeats = 5,
                         classProbs = TRUE,
                         summaryFunction = twoClassSummary)


# Train the XGBoost model
parameters_xgb2 <- expand.grid(nrounds = c(500),
                               max_depth = c(2),
                               eta = c(0.1),
                               gamma = 0.2, 
                               colsample_bytree = c(0.7),
                               min_child_weight = c(1000), 
                               subsample = 0.3)


# #Train the XGBoost model
# set.seed(396596)
# xgb2 <- train(formula,
#               data = train_data,
#               method = "xgbTree",
#               trControl = ctrl_cv5,
#               tuneGrid  = parameters_xgb2)
#xgb2 %>% saveRDS(here("outputs", "xgb2.rds"))
xgb2 <- readRDS("outputs/xgb2.rds")


pred.train.xgb2 <- predict(xgb2, 
                           train_data, 
                           type = "prob")[,2]

pred.test.xgb2 <- predict(xgb2, 
                          test_data, 
                          type = "prob")[,2]




# SVM ---------------------------------------------------------------------

# define target and predictors
target <- train_data$Winner  # replace with your target variable name
predictors <- train_data[, -c(1,2)]
# Convert factors to dummy variables
predictors_dummy <- model.matrix(~ . - 1, data = predictors)

# Convert the predictors to a data.frame
predictors_dummy_df <- as.data.frame(predictors_dummy)

# Predict on test data
predictors_test <- test_data[, -c(1,2)]
predictors_test_dummy <- model.matrix(~ . - 1, data = predictors_test)

predictors_test_dummy <- predictors_test_dummy[, colnames(predictors_dummy)]


# set.seed(396596)
# #Train the model with probability estimates
# svm_model_prob <- svm(as.factor(target) ~ .,
#                       data = predictors_dummy_df,
#                       type = "C-classification",
#                       kernel = "linear",
#                       cost = 0.1,
#                       probability = TRUE)
# 
# svm_model_prob %>% saveRDS(here("outputs", "svm_model_prob.rds"))

svm_model_prob <- readRDS("outputs/svm_model_prob.rds")

# Convert the test predictors to a data.frame
predictors_test_dummy_df <- as.data.frame(predictors_test_dummy)

# Predict probabilities on the test data
SVM_linear_fitted_train <- predict(svm_model_prob, 
                             newdata = predictors_dummy_df, 
                             probability = TRUE)

# Predict probabilities on the test data
SVM_linear_fitted_test <- predict(svm_model_prob, 
                             newdata = predictors_test_dummy_df, 
                             probability = TRUE)

# Extract probabilities
SVM_linear_fitted_prob_train <- attr(SVM_linear_fitted_train, "probabilities")
SVM_linear_fitted_prob_test <- attr(SVM_linear_fitted_test, "probabilities")

confusionMatrix(data = as.factor(ifelse(SVM_linear_fitted_prob_test[,1] >0.5, "TRUE.", "FALSE.")), reference = test_data$Winner)
ROC.test.svm_linear  <- roc(test_data$Winner == "TRUE.", 
                            SVM_linear_fitted_prob_test[,1])
# LASSO -------------------------------------------------------------------


# Set up grid
parameters <- expand.grid(lambda = c(0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000))

set.seed(396596)
#Fit models and perform cross-validation
# lasso_model <- cv.glmnet(x = model.matrix(formula, train_data),
#                          y = train_data$Winner,
#                          family = "binomial",
#                          alpha = 1,
#                          type.measure = "auc",
#                          nfolds = 5,
#                          lambda = parameters$lambda)
# lasso_model %>% saveRDS(here("outputs", "lasso_model.rds"))
lasso_model <- readRDS("outputs/lasso_model.rds")
best_lasso <- which.max(lasso_model$cvm)

# Make predictions on train data
lasso_pred_train <- predict(lasso_model, newx = model.matrix(formula, train_data),
                      s = lasso_model$lambda[best_lasso], type = "response")

# Make predictions on test data
lasso_pred_test <- predict(lasso_model, newx = model.matrix(formula, test_data),
                      s = lasso_model$lambda[best_lasso], type = "response")

compute_ece(lasso_pred_test)
auc_lasso <- auc(test_data$Winner, lasso_pred_test)

ROC.test.Lasso  <- roc(as.numeric(test_data$Winner == "TRUE." ), 
                       lasso_pred_test)

confusionMatrix(data = as.factor(ifelse(lasso_pred_test >0.5, "TRUE.", "FALSE.")), reference = test_data$Winner)


# Models comparison -------------------------------------------------------


accuracy <- function(preds, actuals){
  result = sum(ifelse(preds > 0.5, "TRUE.", "FALSE.") == actuals) / length(actuals)
}

comparison_df <- data.frame(
  Models = c("Logit", 
             "Random Forest", 
             "eXtreme Gradient Boosting", 
             "Support Vector Machines", 
             "Lasso"),
  ECE = c(compute_ece(logit_test_fitted),
          compute_ece(pred.test.randomforest4),
          compute_ece(pred.test.xgb2),
          compute_ece(SVM_linear_fitted_prob_test[,1]),
          compute_ece(lasso_pred_test)),
  Accuracy_test = c(accuracy(logit_test_fitted, test_data$Winner),
                    accuracy(pred.test.randomforest4, test_data$Winner),
                    accuracy(pred.test.xgb2, test_data$Winner),
                    accuracy(SVM_linear_fitted_prob_test[,1], test_data$Winner),
                    accuracy(lasso_pred_test, test_data$Winner)),
  difference = c(accuracy(logit_fitted, train_data$Winner)-accuracy(logit_test_fitted, test_data$Winner),
                 accuracy(pred.train.randomforest4, train_data$Winner)-accuracy(pred.test.randomforest4, test_data$Winner),
                 accuracy(pred.train.xgb2, train_data$Winner)-accuracy(pred.test.xgb2, test_data$Winner),
                 accuracy(SVM_linear_fitted_prob_train[,1], train_data$Winner)-accuracy(SVM_linear_fitted_prob_test[,1], test_data$Winner),
                 accuracy(lasso_pred_train, train_data$Winner)-accuracy(lasso_pred_test, test_data$Winner))
)



# Create a pander table
pander(comparison_df)

# Validations -------------------------------------------------------------

validation_graph <- test_data[,c(1,2,4,7)]
# count duplicates in test_data


validation_graph$logit_pred <- logit_test_fitted
validation_graph$rf_pred <- pred.test.randomforest4
validation_graph$xgb_pred <- pred.test.xgb2
validation_graph$svm_pred <- SVM_linear_fitted_prob_test[,1]
validation_graph$lasso_pred <- lasso_pred_test[,"s1"]


# create the bucket column

real_bucket_middle_points <- function(input, n_buckets) {
  bucket_width = 1 / n_buckets
  buckets <- cut(input, seq(0, 1, bucket_width), include.lowest = TRUE)
  
  # Create a data frame with your input and its corresponding bucket
  data <- data.frame(input = input, bucket = buckets)
  
  # Group by bucket and calculate the mean
  bucket_means <- aggregate(input ~ bucket, data, mean)
  
  # Create a named vector to use for matching
  bucket_mean_vec <- setNames(bucket_means$input, bucket_means$bucket)
  
  # Match the bucket mean to each data point's bucket
  mean_input <- bucket_mean_vec[as.character(buckets)]
  
  return(mean_input)
}


validation_graph$logit_bucket_middle_point <- real_bucket_middle_points(validation_graph$logit_pred, 20)
validation_graph$rf_bucket_middle_point <- real_bucket_middle_points(validation_graph$rf_pred, 20)
validation_graph$xgb_bucket_middle_point <- real_bucket_middle_points(validation_graph$xgb_pred, 20)
validation_graph$svm_bucket_middle_point <- real_bucket_middle_points(validation_graph$svm_pred, 20)
validation_graph$lasso_bucket_middle_point <- real_bucket_middle_points(validation_graph$lasso_pred, 20)



# Probability buckets vs actual wins --------------------------------------

logit_calibration <- validation_graph %>%
  group_by(logit_bucket_middle_point) %>%
  summarize(true_logit_percentage = mean(Winner == "TRUE.")) %>%
  ggplot(aes(x = logit_bucket_middle_point, y = true_logit_percentage)) +
  geom_point(size = 3, shape = 21, fill = "white", color = "blue") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  labs(x = "Expected probability", y = "Fraction of positives",
       title = "LOGIT model predictions vs truly won") +
  theme(plot.title = element_text(hjust = 0.5, face="bold", size=14, color="black"),
        axis.title.x = element_text(face="bold", size=12, color="black"),
        axis.title.y = element_text(face="bold", size=12, color="black"),
        axis.text = element_text(size=12),
        legend.position = "none")

rf_calibration <- validation_graph %>%
  group_by(rf_bucket_middle_point) %>%
  summarize(true_rf_percentage = mean(Winner == "TRUE.")) %>%
  ggplot(aes(x = rf_bucket_middle_point, y = true_rf_percentage)) +
  geom_point(size = 3, shape = 21, fill = "white", color = "red") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  labs(x = "Expected probability", y = "Fraction of positives",
       title = "Random Forest model predictions vs truly won") +
  theme(plot.title = element_text(hjust = 0.5, face="bold", size=14, color="black"),
        axis.title.x = element_text(face="bold", size=12, color="black"),
        axis.title.y = element_text(face="bold", size=12, color="black"),
        axis.text = element_text(size=12),
        legend.position = "none")

xgb_calibration <- validation_graph %>%
  group_by(xgb_bucket_middle_point) %>%
  summarize(true_xgb_percentage = mean(Winner == "TRUE.")) %>%
  ggplot(aes(x = xgb_bucket_middle_point, y = true_xgb_percentage)) +
  geom_point(size = 3, shape = 21, fill = "white", color = "purple") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  labs(x = "Expected probability", y = "Fraction of positives")+
  theme(plot.title = element_text(hjust = 0.5, face="bold", size=14, color="black"),
        axis.title.x = element_text(face="bold", size=12, color="black"),
        axis.title.y = element_text(face="bold", size=12, color="black"),
        axis.text = element_text(size=12),
        legend.position = "none")

svm_calibration <- validation_graph %>%
  group_by(svm_bucket_middle_point) %>%
  summarize(true_svm_percentage = mean(Winner == "TRUE.") ) %>%
  ggplot(aes(x = svm_bucket_middle_point, y = true_svm_percentage)) +
  geom_point(size = 3, shape = 21, fill = "white", color = "magenta") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  labs(x = "Expected probability", y = "Fraction of positives",
       title = "SVM model predictions vs truly won") +
  theme(plot.title = element_text(hjust = 0.5, face="bold", size=14, color="black"),
        axis.title.x = element_text(face="bold", size=12, color="black"),
        axis.title.y = element_text(face="bold", size=12, color="black"),
        axis.text = element_text(size=12),
        legend.position = "none")

lasso_calibration <- validation_graph %>%
  group_by(lasso_bucket_middle_point) %>%
  summarize(true_lasso_percentage = mean(Winner == "TRUE.")) %>%
  ggplot(aes(x = lasso_bucket_middle_point, y = true_lasso_percentage)) +
  geom_point(size = 3, shape = 21, fill = "white", color = "dark green") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  labs(x = "Expected probability", y = "Fraction of positives") +
  theme(plot.title = element_text(hjust = 0.5, face="bold", size=14, color="black"),
        axis.title.x = element_text(face="bold", size=12, color="black"),
        axis.title.y = element_text(face="bold", size=12, color="black"),
        axis.text = element_text(size=12),
        legend.position = "none")


# arrange in a grid
grid.arrange(grobs = list(xgb_calibration, lasso_calibration), ncol = 2) 




validation_graph %>%
  group_by(lasso_bucket_middle_point) %>%
  summarize(true_logit_percentage = mean(Winner == "TRUE.") * 100,
            weight = n())


# Create match time buckets
validation_graph <- validation_graph %>% group_by(MatchID) %>% mutate(row_count = row_number())

validation_graph <- validation_graph %>%
  group_by(MatchID) %>%
  mutate(row_count = row_number(),
         max_count = max(row_count),
         percent_match = row_count/max_count) %>%
  ungroup()

validation_graph %>%
  filter(!(max_count %in% c(39, 43, 96))) %>%
  summarise(min_max_count = min(max_count),
            max_max_count = max(max_count))

  validation_graph[validation_graph$max_count == 117,1]

validation_graph$time_bucket <- real_bucket_middle_points(validation_graph$percent_match, 100)
# Accuracy over relative time ---------------------------------------------


# time bucket charts
logit_relative_time <- validation_graph %>%
  mutate(prediction = factor(ifelse(logit_pred > 0.5, # condition
                                    "TRUE.", # what returned if condition TRUE
                                    "FALSE.")),
         prediction_success = ifelse(prediction == Winner, 1,0)) %>%
  group_by(time_bucket) %>%
  summarize(accuracy = sum(prediction_success) / n()) %>%
  ggplot(aes(x = time_bucket, y = accuracy)) +
  geom_point(size = 3, shape = 21, fill = "white", color = "blue")  +
  geom_smooth(color = "blue", linewidth = 0.1)+
  labs(x = "Match progress bucket (%)", y = "Predictions accuracy", 
       title = "LOGIT predictions accuracy over relative time") +
  theme(plot.title = element_text(hjust = 0.5, face="bold", size=14, color="black"),
        axis.title.x = element_text(face="bold", size=12, color="black"),
        axis.title.y = element_text(face="bold", size=12, color="black"),
        axis.text = element_text(size=12),
        legend.position = "none")

rf_relative_time <- validation_graph %>%
  mutate(prediction = factor(ifelse(rf_pred > 0.5, # condition
                                    "TRUE.", # what returned if condition TRUE
                                    "FALSE.")),
         prediction_success = ifelse(prediction == Winner, 1,0)) %>%
  group_by(time_bucket) %>%
  summarize(accuracy = sum(prediction_success) / n()) %>%
  ggplot(aes(x = time_bucket, y = accuracy)) +
  geom_point(size = 3, shape = 21, fill = "white", color = "red") +
  geom_smooth(color = "red", linewidth = 0.1)+
  labs(x = "Match progress bucket (%)", y = "Predictions accuracy", 
       title = "Random Forest predictions accuracy over relative time") +
  theme(plot.title = element_text(hjust = 0.5, face="bold", size=14, color="black"),
        axis.title.x = element_text(face="bold", size=12, color="black"),
        axis.title.y = element_text(face="bold", size=12, color="black"),
        axis.text = element_text(size=12),
        legend.position = "none")

xgb_relative_time <- validation_graph %>%
  mutate(prediction = factor(ifelse(xgb_pred > 0.5, # condition
                                    "TRUE.", # what returned if condition TRUE
                                    "FALSE.")),
         prediction_success = ifelse(prediction == Winner, 1,0)) %>%
  group_by(time_bucket) %>%
  summarize(accuracy = sum(prediction_success) / n()) %>%
  ggplot(aes(x = time_bucket, y = accuracy)) +
  geom_point(size = 3, shape = 21, fill = "white", color = "purple") +
  geom_smooth(color = "purple", linewidth = 0.1)+
  labs(x = "Match progress bucket (%)", y = "Predictions accuracy", 
       title = "XGBoost predictions accuracy over relative time") +
  theme(plot.title = element_text(hjust = 0.5, face="bold", size=14, color="black"),
        axis.title.x = element_text(face="bold", size=12, color="black"),
        axis.title.y = element_text(face="bold", size=12, color="black"),
        axis.text = element_text(size=12),
        legend.position = "none")


svm_relative_time <- validation_graph %>%
  mutate(prediction = factor(ifelse(svm_pred > 0.5, # condition
                                    "TRUE.", # what returned if condition TRUE
                                    "FALSE.")),
         prediction_success = ifelse(prediction == Winner, 1,0)) %>%
  group_by(time_bucket) %>%
  summarize(accuracy = sum(prediction_success) / n()) %>%
  ggplot(aes(x = time_bucket, y = accuracy)) +
  geom_point(size = 3, shape = 21, fill = "white", color = "magenta") +
  geom_smooth(color = "magenta", linewidth = 0.1)+
  labs(x = "Match progress bucket (%)", y = "Predictions accuracy", 
       title = "SVM predictions accuracy over relative time") +
  theme(plot.title = element_text(hjust = 0.5, face="bold", size=14, color="black"),
        axis.title.x = element_text(face="bold", size=12, color="black"),
        axis.title.y = element_text(face="bold", size=12, color="black"),
        axis.text = element_text(size=12),
        legend.position = "none")

lasso_relative_time <- validation_graph %>%
  mutate(prediction = factor(ifelse(lasso_pred > 0.5, # condition
                                    "TRUE.", # what returned if condition TRUE
                                    "FALSE.")),
         prediction_success = ifelse(prediction == Winner, 1,0)) %>%
  group_by(time_bucket) %>%
  summarize(accuracy = sum(prediction_success) / n()) %>%
  ggplot(aes(x = time_bucket, y = accuracy)) +
  geom_point(size = 3, shape = 21, fill = "white", color = "dark green") +
  geom_smooth(color = "dark green", linewidth = 0.1)+
  labs(x = "Match progress bucket (%)", y = "Predictions accuracy", 
       title = "Lasso predictions accuracy over relative time") +
  theme(plot.title = element_text(hjust = 0.5, face="bold", size=14, color="black"),
        axis.title.x = element_text(face="bold", size=12, color="black"),
        axis.title.y = element_text(face="bold", size=12, color="black"),
        axis.text = element_text(size=12),
        legend.position = "none")

grid.arrange(grobs = list(logit_relative_time, xgb_relative_time, svm_relative_time, lasso_relative_time), ncol = 2)

# combined accuracy over relative time ------------------------------------


# Bind the data frames into one
all_models_data <- bind_rows(
  validation_graph %>%
    mutate(model = "logit", 
           prediction = ifelse(logit_pred > 0.5, "TRUE.", "FALSE."),
           prediction_success = ifelse(prediction == Winner, 1, 0)),
  
  validation_graph %>%
    mutate(model = "rf", 
           prediction = ifelse(rf_pred > 0.5, "TRUE.", "FALSE."),
           prediction_success = ifelse(prediction == Winner, 1, 0)),
  
  validation_graph %>%
    mutate(model = "xgb", 
           prediction = ifelse(xgb_pred > 0.5, "TRUE.", "FALSE."),
           prediction_success = ifelse(prediction == Winner, 1, 0)),
  
  validation_graph %>%
    mutate(model = "svm", 
           prediction = ifelse(svm_pred > 0.5, "TRUE.", "FALSE."),
           prediction_success = ifelse(prediction == Winner, 1, 0)),
  
  validation_graph %>%
    mutate(model = "lasso", 
           prediction = ifelse(lasso_pred > 0.5, "TRUE.", "FALSE."),
           prediction_success = ifelse(prediction == Winner, 1, 0))
)

# Compute accuracy by model and time bucket
accuracy_data <- all_models_data %>%
  group_by(model, time_bucket) %>%
  summarize(accuracy = sum(prediction_success) / n(), .groups = "drop")

# Compute accuracy by model and time bucket, excluding RF model
accuracy_data <- all_models_data %>%
  filter(model != "rf") %>%
  group_by(model, time_bucket) %>%
  summarize(accuracy = sum(prediction_success) / n(), .groups = "drop")

# Plot
ggplot(accuracy_data, aes(x = time_bucket, y = accuracy, color = model)) +
  # geom_point(size = 3, shape = 21, fill = "white") + # Remove this line to remove points
  geom_smooth(se = FALSE) +
  labs(x = "Match progress bucket (%)", y = "Predictions accuracy", 
       title = "Model predictions accuracy over relative time") +
  theme(plot.title = element_text(hjust = 0.5, face="bold", size=14, color="black"),
        axis.title.x = element_text(face="bold", size=12, color="black"),
        axis.title.y = element_text(face="bold", size=12, color="black"),
        axis.text = element_text(size=12)) +
  scale_color_manual(values = c("logit" = "blue", "xgb" = "purple", "svm" = "magenta", "lasso" = "dark green")) +
  guides(color = guide_legend(title = "Model"))


# Accuracy over time ------------------------------------------------------


set_mean_rows <- data.frame()

for (i in 1:5) {
  temp_result <- validation_graph %>%
    group_by(MatchID) %>%
    filter(Set_number == i) %>%
    summarise(max_row_count = max(row_count)) %>%
    summarise(mean_max_row_count = mean(max_row_count))
  
  temp_result$set_number <- i
  set_mean_rows <- rbind(set_mean_rows, temp_result)
}
set_mean_rows$prev_mean_max_row_count <- lag(set_mean_rows$mean_max_row_count)
set_mean_rows$label_position <- (set_mean_rows$mean_max_row_count + set_mean_rows$prev_mean_max_row_count) / 2
set_mean_rows$label_position[1] <- set_mean_rows$mean_max_row_count[1] / 2


logit_acc_over_time <- validation_graph %>%
  mutate(prediction = factor(ifelse(logit_pred > 0.5, # condition
                                    "TRUE.", # what returned if condition TRUE
                                    "FALSE.")),
         prediction_success = ifelse(prediction == Winner, 1,0)) %>%
  group_by(row_count) %>%
  summarize(accuracy = sum(prediction_success) / n()) %>%
  mutate(cutoff = quantile(row_count, 0.9)) %>%
  filter(row_count <= cutoff) %>%
  ggplot(aes(x = row_count, y = accuracy)) +
  geom_point(size = 2, shape = 21, fill = "white", color = "black") +
  geom_smooth(color = "blue") +
  ylim(0.6, 1) +
  labs(x = "Point number", y = "Predictions accuracy", 
       title = "LOGIT predictions accuracy over time") +
  theme(plot.title = element_text(hjust = 0.5, face="bold", size=14, color="black"),
        axis.title.x = element_text(face="bold", size=12, color="black"),
        axis.title.y = element_text(face="bold", size=12, color="black"),
        axis.text = element_text(size=12),
        legend.position = "none")



rf_acc_over_time <- validation_graph %>%
  mutate(prediction = factor(ifelse(rf_pred > 0.5, # condition
                                    "TRUE.", # what returned if condition TRUE
                                    "FALSE.")),
         prediction_success = ifelse(prediction == Winner, 1,0)) %>%
  group_by(row_count) %>%
  summarize(accuracy = sum(prediction_success) / n()) %>%
  mutate(cutoff = quantile(row_count, 0.9)) %>%
  filter(row_count <= cutoff) %>%
  ggplot(aes(x = row_count, y = accuracy)) +
  geom_point(size = 2, shape = 21, fill = "white", color = "black") +
  geom_smooth(color = "red") +
  ylim(0.6, 1) +
  labs(x = "Point number", y = "Predictions accuracy", 
       title = "Random Forest predictions accuracy over time") +
  theme(plot.title = element_text(hjust = 0.5, face="bold", size=14, color="black"),
        axis.title.x = element_text(face="bold", size=12, color="black"),
        axis.title.y = element_text(face="bold", size=12, color="black"),
        axis.text = element_text(size=12),
        legend.position = "none")

xgb_acc_over_time <- validation_graph %>%
  mutate(prediction = factor(ifelse(xgb_pred > 0.5, # condition
                                    "TRUE.", # what returned if condition TRUE
                                    "FALSE.")),
         prediction_success = ifelse(prediction == Winner, 1,0)) %>%
  group_by(row_count) %>%
  summarize(accuracy = sum(prediction_success) / n()) %>%
  mutate(cutoff = quantile(row_count, 0.9)) %>%
  filter(row_count <= cutoff) %>%
  ggplot(aes(x = row_count, y = accuracy)) +
  geom_point(size = 2, shape = 21, fill = "white", color = "black") +
  geom_smooth(color = "purple") +
  ylim(0.6, 1) +
  labs(x = "Point number", y = "Predictions accuracy", 
       title = "XGBoost predictions accuracy over time") +
  theme(plot.title = element_text(hjust = 0.5, face="bold", size=14, color="black"),
        axis.title.x = element_text(face="bold", size=12, color="black"),
        axis.title.y = element_text(face="bold", size=12, color="black"),
        axis.text = element_text(size=12),
        legend.position = "none")

svm_acc_over_time <- validation_graph %>%
  mutate(prediction = factor(ifelse(svm_pred > 0.5, # condition
                                    "TRUE.", # what returned if condition TRUE
                                    "FALSE.")),
         prediction_success = ifelse(prediction == Winner, 1,0)) %>%
  group_by(row_count) %>%
  summarize(accuracy = sum(prediction_success) / n()) %>%
  mutate(cutoff = quantile(row_count, 0.9)) %>%
  filter(row_count <= cutoff) %>%
  ggplot(aes(x = row_count, y = accuracy)) +
  geom_point(size = 2, shape = 21, fill = "white", color = "black") +
  geom_smooth(color = "magenta") +
  ylim(0.6, 1) +
  labs(x = "Point number", y = "Predictions accuracy", 
       title = "SVM predictions accuracy over time") +
  theme(plot.title = element_text(hjust = 0.5, face="bold", size=14, color="black"),
        axis.title.x = element_text(face="bold", size=12, color="black"),
        axis.title.y = element_text(face="bold", size=12, color="black"),
        axis.text = element_text(size=12),
        legend.position = "none")

lasso_acc_over_time <- validation_graph %>%
  mutate(prediction = factor(ifelse(lasso_pred > 0.5, # condition
                                    "TRUE.", # what returned if condition TRUE
                                    "FALSE.")),
         prediction_success = ifelse(prediction == Winner, 1,0)) %>%
  group_by(row_count) %>%
  summarize(accuracy = sum(prediction_success) / n()) %>%
  mutate(cutoff = quantile(row_count, 0.9)) %>%
  filter(row_count <= cutoff) %>%
  ggplot(aes(x = row_count, y = accuracy)) +
  geom_point(size = 2, shape = 21, fill = "white", color = "black") +
  geom_smooth(color = "dark green") +
  ylim(0.6, 1) +
  labs(x = "Point number", y = "Predictions accuracy", 
       title = "Lasso predictions accuracy over time") +
  theme(plot.title = element_text(hjust = 0.5, face="bold", size=14, color="black"),
        axis.title.x = element_text(face="bold", size=12, color="black"),
        axis.title.y = element_text(face="bold", size=12, color="black"),
        axis.text = element_text(size=12),
        legend.position = "none")

add_lines_and_labels <- function(plot) {
  for (i in 1:nrow(set_mean_rows)) {
    plot <- plot + 
      geom_vline(xintercept = set_mean_rows$mean_max_row_count[i], linetype = "dashed", color = "blue") +
      annotate("text", x = set_mean_rows$label_position[i], y = Inf, label = paste("Set", set_mean_rows$set_number[i]), vjust = 1.05, color = "black")
  }
  return(plot)
}

logit_acc_over_time <- add_lines_and_labels(logit_acc_over_time)
rf_acc_over_time <- add_lines_and_labels(rf_acc_over_time)
xgb_acc_over_time <- add_lines_and_labels(xgb_acc_over_time)
svm_acc_over_time <- add_lines_and_labels(svm_acc_over_time)
lasso_acc_over_time <- add_lines_and_labels(lasso_acc_over_time)



# list of plots
plot_list <- list(logit_acc_over_time, xgb_acc_over_time, svm_acc_over_time, lasso_acc_over_time)

# arrange in a grid
grid.arrange(grobs = plot_list, ncol = 2)  # 2 column layout


# Combined accuracy over time ---------------------------------------------

# First calculate the prediction success and accuracy for each model in separate dataframes
logit_acc_df <- validation_graph %>%
  mutate(prediction = factor(ifelse(logit_pred > 0.5, "TRUE.", "FALSE.")),
         prediction_success = ifelse(prediction == Winner, 1, 0)) %>%
  group_by(row_count) %>%
  summarize(accuracy = sum(prediction_success) / n()) %>%
  mutate(cutoff = quantile(row_count, 0.9),
         model = 'Logit') %>%
  filter(row_count <= cutoff)

xgb_acc_df <- validation_graph %>%
  mutate(prediction = factor(ifelse(xgb_pred > 0.5, "TRUE.", "FALSE.")),
         prediction_success = ifelse(prediction == Winner, 1, 0)) %>%
  group_by(row_count) %>%
  summarize(accuracy = sum(prediction_success) / n()) %>%
  mutate(cutoff = quantile(row_count, 0.9),
         model = 'XGBoost') %>%
  filter(row_count <= cutoff)

svm_acc_df <- validation_graph %>%
  mutate(prediction = factor(ifelse(svm_pred > 0.5, "TRUE.", "FALSE.")),
         prediction_success = ifelse(prediction == Winner, 1, 0)) %>%
  group_by(row_count) %>%
  summarize(accuracy = sum(prediction_success) / n()) %>%
  mutate(cutoff = quantile(row_count, 0.9),
         model = 'SVM') %>%
  filter(row_count <= cutoff)

lasso_acc_df <- validation_graph %>%
  mutate(prediction = factor(ifelse(lasso_pred > 0.5, "TRUE.", "FALSE.")),
         prediction_success = ifelse(prediction == Winner, 1, 0)) %>%
  group_by(row_count) %>%
  summarize(accuracy = sum(prediction_success) / n()) %>%
  mutate(cutoff = quantile(row_count, 0.9),
         model = 'Lasso') %>%
  filter(row_count <= cutoff)

# Combine all dataframes into one
acc_df <- rbind(logit_acc_df, xgb_acc_df, svm_acc_df, lasso_acc_df)

# Plot combined data
acc_over_time <- ggplot(acc_df, aes(x = row_count, y = accuracy, color = model)) +
  geom_smooth(se = FALSE) +
  ylim(0.6, 1) +
  labs(x = "Point number", y = "Predictions accuracy", 
       title = "Predictions accuracy over time") +
  theme(plot.title = element_text(hjust = 0.5, face="bold", size=14, color="black"),
        axis.title.x = element_text(face="bold", size=12, color="black"),
        axis.title.y = element_text(face="bold", size=12, color="black"),
        axis.text = element_text(size=12))

print(acc_over_time)



# Accuracy at first point of each set -------------------------------------

set_mean_rows <- data.frame()

for (i in 1:5) {
  temp_result <- validation_graph %>%
    group_by(MatchID) %>%
    filter(Set_number == i) %>%
    summarise(max_row_count = max(row_count)) %>%
    summarise(mean_max_row_count = mean(max_row_count))
  
  temp_result$set_number <- i
  set_mean_rows <- rbind(set_mean_rows, temp_result)
}
set_mean_rows$prev_mean_max_row_count <- lag(set_mean_rows$mean_max_row_count)
set_mean_rows$label_position <- (set_mean_rows$mean_max_row_count + set_mean_rows$prev_mean_max_row_count) / 2
set_mean_rows$label_position[1] <- set_mean_rows$mean_max_row_count[1] / 2


Sets_calibration <- validation_graph %>%
    group_by(MatchID, Set_number) %>%
    #filter(Set_number == 1) %>%
    summarise(lasso = first(lasso_pred),
              point = first(row_count),
              last = max(row_count),
              Winner = first(Winner))

Sets_calibration$lasso_bucket_middle_point <- real_bucket_middle_points(Sets_calibration$lasso, 5)            

Sets_calibration %>%
  group_by(lasso_bucket_middle_point, Set_number) %>%
  summarize(true_lasso_percentage = mean(Winner == "TRUE.")) %>%
  ggplot(aes(x = lasso_bucket_middle_point, y = true_lasso_percentage, color = Set_number)) +
  geom_point(size = 3, shape = 21, fill = "white") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  labs(x = "Expected probability", y = "Fraction of positives") +
  theme(plot.title = element_text(hjust = 0.5, face="bold", size=14, color="black"),
        axis.title.x = element_text(face="bold", size=12, color="black"),
        axis.title.y = element_text(face="bold", size=12, color="black"),
        axis.text = element_text(size=12),
        legend.position = "none")

# PRE ---------------------------------------------------------------------

train_data_pre <- train_data %>% dplyr::group_by(MatchID) %>% dplyr::slice(1)

test_data_pre <- test_data %>% group_by(MatchID) %>% dplyr::slice(1)

formula_pre <- Winner ~ Year + Time_Category + diff_Current_position_table +
  diff_Matches_ratio_last_5 + diff_Season_points_to_matches + diff_Sets_ratio_table + diff_Points_ratio_table +
  Match_time_of_season

# Compute ECE
compute_ece_pre <- function(predicted_probs,  n_bins = 20) {
  
  observed_labels <- ifelse(test_data_pre$Winner == "TRUE.", 1, 0)
  
  # Create bins for the predicted probabilities
  bins <- cut(predicted_probs, breaks = seq(0, 1, length.out = n_bins + 1), include.lowest = TRUE, labels = FALSE)
  
  # Compute the average predicted probability in each bin
  bin_avgs <- tapply(predicted_probs, bins, mean)
  
  # Compute the observed frequency in each bin
  bin_true <- tapply(observed_labels, bins, mean, default = 0)
  
  # Compute the absolute difference between the predicted probability and observed frequency in each bin
  bin_diffs <- abs(bin_avgs - bin_true)
  
  # Compute the number of instances in each bin
  bin_counts <- table(bins)
  
  # Compute ECE
  ece <- sum(bin_diffs * bin_counts) / length(predicted_probs)
  
  return(ece)
}
# Logit with pre only -----------------------------------------------------

logit_pre <- glm(formula_pre,
                 # here we define type of the model
                 family =  binomial(link = "logit"),
                 data = train_data_pre)
logit_pre %>% saveRDS(here("outputs", "logit_pre.rds"))
logit_pre <- readRDS("outputs/logit_pre.rds")
summary(logit_pre)
lrtest(logit_pre)
logit_pre_fitted <- predict.glm(logit_pre,
                                type = "response")

logit_pre_test_fitted <- predict.glm(logit_pre, test_data_pre, type = "response")

confusionMatrix(data = as.factor(ifelse(logit_pre_fitted > 0.5, # condition
                                        "TRUE.", # what returned if condition TRUE
                                        "FALSE.")), 
                reference = as.factor(train_data_pre$Winner))

confusionMatrix(data = as.factor(ifelse(logit_pre_test_fitted > 0.5, # condition
                                        "TRUE.", # what returned if condition TRUE
                                        "FALSE.")), 
                reference = as.factor(test_data_pre$Winner))

# RF pre ------------------------------------------------------------------

ctrl_cv5 <- trainControl(method = "cv", 
                         number =    5,
                         classProbs = T)




# randomforest_pre <- randomForest(formula_pre,
#                                  data = train_data_pre,
#                                  ntree = 300,
#                                  mtry = 2,
#                                  maxnodes = 6,
#                                  trControl = ctrl_cv5,
#                                  nodesize = 300)
# randomforest4_pre %>% saveRDS(here("outputs", "randomforest_pre.rds"))
 randomforest4_pre <- readRDS("outputs/randomforest_pre.rds")
pred.train.randomforest4_pre <- predict(randomforest_pre, 
                                        train_data_pre, 
                                        type = "prob")[,2]


pred.test.randomforest4_pre <- predict(randomforest_pre, 
                                       test_data_pre, 
                                       type = "prob")[,2]


# XGB pre -----------------------------------------------------------------


# define target and predictors
target_pre <- train_data_pre$Winner  # replace with your target variable name
target_pre <-as.numeric(target_pre) - 1
test_target_pre <- test_data_pre$Winner 
test_target_pre <- as.numeric(test_target_pre) -1 
predictors_pre <- train_data_pre[, -c(1,2)]
# Convert factors to dummy variables
predictors_dummy_pre <- model.matrix(~ . - 1, data = predictors_pre)

# Predict on test data
predictors_test_pre <- test_data_pre[, -c(1,2)]
predictors_test_dummy_pre <- model.matrix(~ . - 1, data = predictors_test_pre)

predictors_test_dummy_pre <- predictors_test_dummy_pre[, colnames(predictors_dummy_pre)]

# Convert the predictors to a data.frame
predictors_dummy_df_pre <- as.data.frame(predictors_dummy_pre)
results_df_pre <- data.frame()


# Train the XGBoost model

parameters_xgb2_pre <- expand.grid(nrounds = c(100),
                                   max_depth = c(4),
                                   eta = c(0.1),
                                   gamma = 5, 
                                   colsample_bytree = c(0.5),
                                   min_child_weight = c(10), 
                                   subsample = 0.7)

# set.seed(396596)
# xgb2_pre <- train(formula_pre,
#                   data = train_data_pre,
#                   method = "xgbTree",
#                   trControl = ctrl_cv5,
#                   tuneGrid  = parameters_xgb2_pre)

#xgb2_pre %>% saveRDS(here("outputs", "xgb2_pre.rds"))
xgb2_pre <- readRDS("outputs/xgb2_pre.rds")



pred.train.xgb2_pre <- predict(xgb2_pre, 
                               train_data_pre, 
                               type = "prob")[,2]

pred.test.xgb2_pre <- predict(xgb2_pre, 
                              test_data_pre, 
                              type = "prob")[,2]



# SVM pre -----------------------------------------------------------------

# define target and predictors
target_pre <- train_data_pre$Winner  # replace with your target variable name
predictors_pre <- train_data_pre[, -c(1,2)]
# Convert factors to dummy variables
predictors_dummy_pre <- model.matrix(~ . - 1, data = predictors_pre)

# Convert the predictors to a data.frame
predictors_dummy_df_pre <- as.data.frame(predictors_dummy_pre)

# Predict on test data
predictors_test_pre <- test_data_pre[, -c(1,2)]
predictors_test_dummy_pre <- model.matrix(~ . - 1, data = predictors_test_pre)

predictors_test_dummy_pre <- predictors_test_dummy_pre[, colnames(predictors_dummy_pre)]


# set.seed(396596)
#Train the model with probability estimates
# svm_model_prob_pre <- svm(as.factor(target_pre) ~ .,
#                           data = predictors_dummy_df_pre,
#                           type = "C-classification",
#                           kernel = "linear",
#                           cost = 0.1,
#                           probability = TRUE)

# svm_model_prob_pre %>% saveRDS(here("outputs", "svm_model_prob_pre.rds"))

svm_model_prob_pre <- readRDS("outputs/svm_model_prob_pre.rds")

# Convert the test predictors to a data.frame
predictors_test_dummy_df_pre <- as.data.frame(predictors_test_dummy_pre)

# Predict probabilities on the test data
SVM_linear_fitted_train_pre <- predict(svm_model_prob_pre, 
                                       newdata = predictors_dummy_df_pre, 
                                       probability = TRUE)

# Predict probabilities on the test data
SVM_linear_fitted_test_pre <- predict(svm_model_prob_pre, 
                                      newdata = predictors_test_dummy_df_pre, 
                                      probability = TRUE)

# Extract probabilities
SVM_linear_fitted_prob_train_pre <- attr(SVM_linear_fitted_train_pre, "probabilities")
SVM_linear_fitted_prob_test_pre <- attr(SVM_linear_fitted_test_pre, "probabilities")

confusionMatrix(data = as.factor(ifelse(SVM_linear_fitted_prob_test_pre[,2] >0.5, "TRUE.", "FALSE.")), reference = test_data_pre$Winner)

# LASSO pre -------------------------------------------------------------------


# Set up grid
parameters <- expand.grid(lambda = c(0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000))

# set.seed(396596)
# #Fit models and perform cross-validation
# lasso_model_pre <- cv.glmnet(x = model.matrix(formula_pre, train_data_pre),
#                              y = train_data_pre$Winner,
#                              family = "binomial",
#                              alpha = 1,
#                              type.measure = "auc",
#                              nfolds = 5,
#                              lambda = parameters$lambda)
# lasso_model_pre %>% saveRDS(here("outputs", "lasso_model_pre.rds"))
lasso_model_pre <- readRDS("outputs/lasso_model_pre.rds")
best_lasso_pre <- which.max(lasso_model_pre$cvm)

# Make predictions on train data
lasso_pred_train_pre <- predict(lasso_model_pre, newx = model.matrix(formula_pre, train_data_pre),
                                s = lasso_model_pre$lambda[best_lasso_pre], type = "response")

# Make predictions on test data
lasso_pred_test_pre <- predict(lasso_model_pre, newx = model.matrix(formula_pre, test_data_pre),
                               s = lasso_model_pre$lambda[best_lasso_pre], type = "response")

compute_ece_pre(lasso_pred_test_pre)


confusionMatrix(data = as.factor(ifelse(lasso_pred_test_pre >0.5, "TRUE.", "FALSE.")), reference = test_data_pre$Winner)

# Models PRE comparison -------------------------------------------------------


accuracy <- function(preds, actuals){
  result = sum(ifelse(preds > 0.5, "TRUE.", "FALSE.") == actuals) / length(actuals)
}

comparison_df_pre <- data.frame(
  Models = c("Logit PRE", 
             "Random Forest PRE", 
             "eXtreme Gradient Boosting PRE", 
             "Support Vector Machines PRE", 
             "Lasso PRE"),
  ECE = c(compute_ece_pre(logit_pre_test_fitted),
          compute_ece_pre(pred.test.randomforest4_pre),
          compute_ece_pre(pred.test.xgb2_pre),
          compute_ece_pre(SVM_linear_fitted_prob_test_pre[,2]),
          compute_ece_pre(lasso_pred_test_pre)),
  Accuracy_test = c(accuracy(logit_pre_test_fitted, test_data_pre$Winner),
                    accuracy(pred.test.randomforest4_pre, test_data_pre$Winner),
                    accuracy(pred.test.xgb2_pre, test_data_pre$Winner),
                    accuracy(SVM_linear_fitted_prob_test_pre[,2], test_data_pre$Winner),
                    accuracy(lasso_pred_test_pre, test_data_pre$Winner)),
  difference = c(accuracy(logit_pre_fitted, train_data_pre$Winner)-accuracy(logit_pre_test_fitted, test_data_pre$Winner),
                 accuracy(pred.train.randomforest4_pre, train_data_pre$Winner)-accuracy(pred.test.randomforest4_pre, test_data_pre$Winner),
                 accuracy(pred.train.xgb2_pre, train_data_pre$Winner)-accuracy(pred.test.xgb2_pre, test_data_pre$Winner),
                 accuracy(SVM_linear_fitted_prob_train_pre[,2], train_data_pre$Winner)-accuracy(SVM_linear_fitted_prob_test_pre[,2], test_data_pre$Winner),
                 accuracy(lasso_pred_train_pre, train_data_pre$Winner)-accuracy(lasso_pred_test_pre, test_data_pre$Winner))
)



# Create a pander table
pander(comparison_df_pre)
