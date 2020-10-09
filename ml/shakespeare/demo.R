library(sparklyr)

sc <- spark_connect(master = "local", version = "3.0.0")
# source: https://www.kaggle.com/kingburrito666/shakespeare-plays
sdf <- spark_read_csv(sc, path = "./Shakespeare_data.csv")

sdf %>% print(n = 10L)

lines_sdf <- sdf %>%
  dplyr::filter(Play == "Romeo and Juliet") %>%
  dplyr::rename(Character = Player, Line = PlayerLine) %>%
  dplyr::filter(!is.na(Character))

print(lines_sdf %>% sdf_nrow())

datasets <- lines_sdf %>% sdf_random_split(train = 0.7, test = 0.3)
print(datasets$train)
print(datasets$test)

pipeline <- ml_pipeline(sc) %>%
  ft_tokenizer(input_col = "Line", output_col = "words") %>%
  ft_hashing_tf(input_col = "words", output_col = "features", num_features = 2L^15) %>%
  # Assign each character name a unique index number
  ft_string_indexer(input_col = "Character", output_col = "CharacterIndex", handle_invalid = "keep") %>%
  # And then use that index number as the label for prediction
  ml_random_forest_classifier(features_col = "features", label_col = "CharacterIndex", num_trees = 40L, max_depth = 5L)
  # ml_logistic_regression(features_col = "features", label_col = "CharacterIndex")
  # ml_decision_tree_classifier(features_col = "features", label_col = "CharacterIndex", max_depth = 8L, max_bins = 40L)
  # ml_naive_bayes(features_col = "features", label_col = "CharacterIndex")

model <- pipeline %>% ml_fit(datasets$train)
predictions <- model %>% ml_transform(datasets$test) %>%
  dplyr::select(CharacterIndex, prediction, probability) %>%
  collect()

print(sum(predictions$CharacterIndex == predictions$prediction))
print(sum(predictions$CharacterIndex != predictions$prediction))
