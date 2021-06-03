library(tidymodels)
library(tidyverse)
library(magrittr)
library(skimr)
library(kknn)
library(stacks)
library(glmnet)
library(xgboost)
library(kernlab)
library(keras)
library(ranger)
library(purrr) ; library(magrittr) ; library(MASS)

train <- read_csv("train.csv") %>% 
  janitor::clean_names()
test <- read_csv( "test.csv") %>%
  janitor::clean_names()

train %<>%
  mutate_if(is.character, as.factor) %>% 
  mutate(credit = factor(credit))
test %<>%
  mutate_if(is.character, as.factor)

credit_recipe <- train %>% 
  recipe(credit ~ .) %>% 
  step_mutate(yrs_birth = -ceiling(days_birth/365),
              yrs_employed = -ceiling(days_employed/365),
              perincome = income_total / family_size,
              adult_income = (family_size - child_num) * income_total,
              begin_month = -begin_month) %>% 
  step_rm(index, days_birth, work_phone, phone, email) %>%
  step_unknown(occyp_type) %>% 
  step_zv(all_predictors()) %>% 
  step_integer(all_nominal(), -all_outcomes()) %>% 
  step_center(all_predictors(), -all_outcomes()) %>% 
  prep(training = train)

train2 <- juice(credit_recipe)
test2 <- bake(credit_recipe, new_data= test)

set.seed(2021)

validation_split <- vfold_cv(train2, v = 5, strata = credit)


model_spec <- rand_forest(mtry = 3,
                          min_n = 5,
                          trees = 1000) %>% 
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

rf_workflow <- workflow() %>%
  add_model(model_spec) %>% 
  add_formula(credit ~ .)


library(tictoc)
tic()
rf_fit_vfold <- 
  rf_workflow %>% 
  fit_resamples(credit ~ .,
                data = train2,
                resamples = validation_split,
                grid = 2,
                metrics = metric_set(accuracy,roc_auc),
                control = control_stack_resamples())
toc()


library(tictoc)

mlr_spec <- multinom_reg(penalty = tune(),
                         mixture = tune()) %>% 
  set_engine("glmnet") %>% 
  set_mode("classification")


param_grid <- grid_latin_hypercube(penalty(),
                                   mixture(),
                                   size = 3)



mlr_wflow <- 
  workflow() %>% 
  add_model(mlr_spec) %>%
  add_formula(credit ~ .)

tic()
mlr_fit_vfold <-tune_grid(
  mlr_wflow,
  data = train2,
  resamples = validation_split,
  grid = param_grid,
  control = control_stack_resamples(),
  metrics = metric_set(accuracy,roc_auc))
toc()



credit_stacking <- 
  stacks() %>% 
  add_candidates(rf_fit_vfold) %>% 
  add_candidates(mlr_fit_vfold) %>%


credit_stacking %<>% 
  blend_predictions() %>% 
  fit_members()

result <- predict(credit_stacking, test2, type = "prob")


submission <- read_csv("sample_submission.csv")
sub_col <- names(submission)
submission <- bind_cols(submission$index, result)
names(submission) <- sub_col
write.csv(submission, row.names = FALSE,
          "stacking_rf_mlr_son.csv")