library(sparklyr)

sc <- spark_connect(master = "local", version = "3.0.0")
# source: https://www.kaggle.com/kingburrito666/shakespeare-plays
sdf <- spark_read_csv(sc, path = "./Shakespeare_data.csv")

sdf %>% print(n = 10L)

lines_sdf <- sdf %>%
  dplyr::filter(Play == "Hamlet") %>%
  dplyr::rename(Character = Player, Line = PlayerLine) %>%
  dplyr::filter(!is.na(Character)) %>%
  dplyr::mutate(ActScene = substring_index(ActSceneLine, '.', 2)) %>%
  dplyr::group_by(PlayerLinenumber, ActScene) %>%
  dplyr::summarize(Character = first_value(Character), Line = aggregate(collect_list(Line), "", ~ concat_ws(' ', .x, .y)))

transformed_sdf <- lines_sdf %>%
  dplyr::mutate(Terms = array("")) %>%
  dplyr::compute()

print(lines_sdf %>% sdf_nrow())

datasets <- lines_sdf %>% sdf_random_split(train = 0.7, test = 0.3)
print(datasets$train)
print(datasets$test)

pipeline <- ml_pipeline(sc) %>%
  # And then do some feature engineering
  ft_tokenizer(input_col = "Line", output_col = "Terms") %>%
  ft_dplyr_transformer(transformed_sdf %>% dplyr::mutate(num_terms = size(Terms))) %>%
  ft_hashing_tf(input_col = "Terms", output_col = "term_freq", num_features = 2L^15) %>%
  ft_stop_words_remover(input_col = "Terms", output_col = "interesting_terms", stop_words = ml_default_stop_words(sc, "english")) %>%
  ft_hashing_tf(input_col = "interesting_terms", output_col = "interesting_term_freq", num_features = 2L^15) %>%
  ft_ngram(input_col = "Terms", output_col = "bigrams", n = 2L) %>%
  ft_hashing_tf(input_col = "bigrams", output_col = "bigram_freq", num_features = 2L^15) %>%
  ft_ngram(input_col = "interesting_terms", output_col = "interesting_bigrams", n = 2L) %>%
  ft_hashing_tf(input_col = "interesting_bigrams", output_col = "interesting_bigram_freq", num_features = 2L^15) %>%
  ft_vector_assembler(input_cols = c("term_freq", "bigram_freq", "interesting_term_freq", "interesting_bigram_freq", "num_terms"), output_col = "features") %>%
  # Assign each character name a unique index number
  ft_string_indexer(input_col = "Character", output_col = "CharacterIndex", handle_invalid = "keep") %>%
  # Now choose a ML model. Choose wisely.
  ml_random_forest_classifier(features_col = "features", label_col = "CharacterIndex", num_trees = 100L, max_depth = 10L, prediction_col = "prediction")
  # ml_logistic_regression(features_col = "features", label_col = "CharacterIndex")
  # ml_decision_tree_classifier(features_col = "features", label_col = "CharacterIndex", max_depth = 8L, max_bins = 40L)
  # ml_naive_bayes(features_col = "features", label_col = "CharacterIndex")

model <- pipeline %>% ml_fit(datasets$train)
predictions <- model %>% ml_transform(datasets$test) %>%
  dplyr::select(CharacterIndex, prediction, probability) %>%
  collect()

print(sum(predictions$CharacterIndex == predictions$prediction)) # total number of correct predictions
print(sum(predictions$CharacterIndex != predictions$prediction)) # total number of incorrect predictions
