library(tidyverse)
library(tidytext)
library(stopwords)
library(SnowballC)
library(hunspell)
library(spacyr) #spacy_install() necessário na primeira vez
library(tidymodels)
library(textrecipes)
library(discrim)
library(themis)
library(doFuture)


#Processamento paralelo
all_cores <- parallel::detectCores(logical = FALSE)
registerDoFuture()
cl <- makeCluster(all_cores-1)
plan(cluster, workers = cl)

spacyr::spacy_initialize(model = "pt_core_news_lg", entity = FALSE)


#==== Split n Fold ====
split = initial_split(base, strata = top_codificacao)

train = training(split)
test = testing(split)

folds = vfold_cv(train, v = 5, strata = top_codificacao)


#==== Recipe ====
stopword_source <- function(values = c("snowball", "stopwords-iso", "nltk")) {
  new_qual_param(
    type = "character",
    values = values,
    # By default, the first value is selected as default. We'll specify that to
    # make it clear.
    default = "snowball",
    label = c(stopword_source = "stopword_source")
  )
}


recipe = recipe(top_codificacao ~ original + EIXO, data = train) %>% 
         step_dummy(EIXO) %>% 
         step_tokenize(original) %>%
         step_stopwords(original, stopword_source = "snowball", language = "pt") %>% 
         step_ngram(original,num_tokens = tune(), min_num_tokens = 1) %>% 
         step_tfidf(original)


#==== Especificação ====
lasso_spec = multinom_reg(penalty = tune(), mixture = 1) %>%
  set_mode("classification") %>%
  set_engine("glmnet")


#==== Workflows ====
lasso_wf = workflow() %>% 
            add_recipe(recipe) %>% 
            add_model(lasso_spec)


# #==== Fits Simples ====
# lasso_fit = fit_resamples(
#   lasso_wf,
#   folds,
#   control = control_resamples(save_pred = TRUE)
# )
# 
# collect_metrics(lasso_fit)
# 
# conf_mat_resampled(lasso_fit, tidy = FALSE) %>%
#   autoplot(type = "heatmap")
# 
# 
# 
#==== Grid e Tune ====
lasso_grid = grid_regular(
  penalty(),  num_tokens(range = c(1,6)),
  levels = c(penalty = 20, num_tokens = 3)
)

lasso_tune <- tune_grid(
  lasso_wf,
  folds,
  grid = lasso_grid,
  control = control_resamples(save_pred = TRUE)
)


autoplot(lasso_tune) +
  labs(
    title = "Lasso model performance across regularization penalties",
    subtitle = "Performance metrics can be used to identity the best penalty"
  )

show_best(lasso_tune, "accuracy")


conf_mat_resampled(lasso_tune, parameters = lasso_tune %>% select_best("accuracy"), tidy = FALSE) %>%
  autoplot(type = "heatmap")

chosen = lasso_tune %>% select_best("accuracy")


#==== Estimação final ====
final_wf <- finalize_workflow(lasso_wf, chosen)

final <- last_fit(final_wf, split)
collect_metrics(final)

collect_predictions(final) %>%
  conf_mat(truth = top_codificacao, estimate = .pred_class) %>%
  autoplot(type = "heatmap")
