library(readr)
library(dplyr)
library(DataExplorer)
library(lmtest)
library(caret)

# Read the data
df <- read_csv("Plusliga_data_for_model.csv", 
                                    col_types = cols(Year = col_character()))


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

# Printing histograms

plot_histogram(df)

summary(df)

###### JUST FOR NOW
# Subset train_data to only include complete rows
df_complete <- df[complete.cases(df), ]


# Set seed for reproducibility
set.seed(396596)
match_ids <- unique(df_complete$MatchID)

# Split MatchIDs into training and testing sets
match_ids_split <- sample(match_ids, size = floor(0.8 * length(match_ids)), replace = FALSE)
train_data <- df_complete %>% filter(MatchID %in% match_ids_split)
test_data <- df_complete %>% filter(!MatchID %in% match_ids_split)





# Modelling
formula <- Winner~ Year+ Time_Category+ Spectators+ Relative_spectators+ Matches_ratio_last_5_host + Matches_ratio_last_5_guest+ Season_points_to_matches_host+ 
  Season_points_to_matches_guest+ Current_position_table_host+ Current_position_table_guest+ Sets_ratio_table_host+ Sets_ratio_table_guest+ Points_ratio_table_host+ 
  Set_number+ Current_point_difference+ Current_set_difference+ Max_point_difference_throughout_set+ Min_point_difference_throughout_set+ 
  Max_point_difference_throughout_match+ Min_point_difference_throughout_match+Running_net_crossings_average+ Current_host_serve_effectiveness+
  Current_guest_serve_effectiveness+ Current_host_positive_reception_ratio+ Current_guest_positive_reception_ratio + Current_host_perfect_reception_ratio+ 
  Current_guest_perfect_reception_ratio+ Current_host_negative_reception_ratio+ Current_guest_negative_reception_ratio+ Current_host_attack_accuracy+
  Current_guest_attack_accuracy+ Current_host_attack_effectiveness+ Current_guest_attack_effectiveness+ Current_timeouts_host+ Current_timeouts_guest+ 
  Current_challenges_host+ Current_challenges_guest+ Match_time_of_season

logit <- glm(formula,
                    # here we define type of the model
                    family =  binomial(link = "logit"),
                    data = train_data)
summary(logit)
lrtest(logit)
# We cannot reject the null hypothesis, so the model is not insignificant


# one can calculate predicted probabilities 
# of "success" (here the second level of the dependent 
# variable) by adding type="response"

logit_fitted <- predict.glm(logit,
                         type = "response")

table(train_data$Winner,
      ifelse(logit_fitted > 0.5, # condition
             TRUE, # what returned if condition TRUE
             FALSE)) # what returned if condition FALSE

confusionMatrix(data = as.factor(logit_fitted > 0.5), 
                reference = as.factor(train_data$Winner))


logit_test_fitted <- predict.glm(logit, test_data, type = "response")


table(test_data$Winner,
      ifelse(logit_test_fitted > 0.5, # condition
             TRUE, # what returned if condition TRUE
             FALSE)) # what returned if condition FALSE

confusionMatrix(data = as.factor(logit_test_fitted > 0.5), 
                reference = as.factor(test_data$Winner))
