library(tidymodels)
library(tidyverse)
library(magrittr)
library(tictoc)
library(skimr)
library(kknn)
library(stacks)
library(glmnet)
library(xgboost)
library(kernlab)
library(keras)
library(ranger)
library(purrr) ; library(magrittr) ; library(MASS)


########################################################################################################
#############################################      data       ##########################################
########################################################################################################

train <- read_csv("train.csv") %>% 
  janitor::clean_names()
test <- read_csv( "test.csv") %>%
  janitor::clean_names()

#train data의 문자형(character data)를 요인(factor)으로 바꿔주기 & 
#예측하고자 하는 credit을 요인(factor)으로 설정하기
train %<>%
  mutate_if(is.character, as.factor) %>% 
  mutate(credit = factor(credit))


#test data의 문자형 데이터를 factor로 바꿔주기
test %<>%
  mutate_if(is.character, as.factor)

#<take>#데이터 탐색
train %>% skim()



#전처리
credit_recipe <- train %>% 
  recipe(credit ~ .) %>%  #credit을 y로 하는 모형으로(credit외의 변수를 예측하는데 쓸것임)
  step_mutate(yrs_birth = -ceiling(days_birth/365), #나이
              yrs_employed = -ceiling(days_employed/365), #근무기간
              perincome = income_total / family_size, #인당소득
              adult_income = (family_size - child_num) * income_total, #성인소득
              begin_month = -begin_month) %>% #신용카드를 발급한지 몇달 됐니
  step_rm(index, days_birth, work_phone, phone, email) %>%  #데이터 제거
  step_unknown(occyp_type) %>% #먼지 모르겠음
  step_zv(all_predictors()) %>% #분산이 0인 변수를 제거
  step_integer(all_nominal(), -all_outcomes()) %>% #먼지 모르겠음
  step_center(all_predictors(), -all_outcomes()) %>% #평균을 0으로 만들기
  prep(training = train) #training으로 'train'data를 사용한다


train2 <- juice(credit_recipe)
test2 <- bake(credit_recipe, new_data= test)

set.seed(2021)

validation_split <- vfold_cv(train2, v = 5, strata = credit)


#################################################################################################
#################################         random forest        ##################################
#################################################################################################

#################################         rf 모수 추정         ##################################

tune_spec <- rand_forest(mtry = tune(),
                         min_n = tune(),
                         trees = 1000) %>% 
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

param_grid <- grid_random(finalize(mtry(), x = train2[,-1]), min_n(), size = 100)

workflow <- workflow() %>%
  add_model(tune_spec) %>% 
  add_formula(credit ~ .)

library(doParallel)
Cluster <- makeCluster(detectCores() - 1)
registerDoParallel(Cluster)

library(tictoc)
tic()
tune_result <- workflow %>% 
  tune_grid(validation_split,
            grid = param_grid,
            control = control_stack_resamples(),
            metrics = metric_set(mn_log_loss))
toc()

tune_best <- tune_result %>% select_best(metric = "mn_log_loss")
tune_best$mtry
tune_best$min_n

### mn_log_loss 기준 tunebest -> mtry : 3, min_n : 11
### acc_roc 기준 tunebest -> mtry : , min_n : 
### 시간이 걸려서 나중에 계산해 보길,,,


#################################       rf vfold 계산      ##########################################



model_spec <- rand_forest(mtry = 3,
                          min_n = 11,
                          trees = 1000) %>% 
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

rf_workflow <- workflow() %>%
  add_model(model_spec) %>% 
  add_formula(credit ~ .)

tic()
rf_fit_vfold <- 
  rf_workflow %>% 
  fit_resamples(credit ~ .,
                data = train2,
                resamples = validation_split,
                metrics = metric_set(mn_log_loss),
                control = control_stack_resamples())
toc()


######################################################################################################
##########################################         mlr          ######################################
######################################################################################################

########################################## mlr  모수 추정   ##########################################

mlr_spec <- multinom_reg(penalty = tune(),
                         mixture = tune()) %>% 
  set_engine("glmnet") %>% 
  set_mode("classification")


param_grid <- grid_latin_hypercube(penalty(), mixture(),  size = 100)


mlr_wflow <- 
  workflow() %>% 
  add_model(mlr_spec) %>%
  add_formula(credit ~ .)

tic()
tune_result <- mlr_wflow %>% 
  tune_grid(validation_split,
            grid = param_grid,
            control = control_stack_resamples(),
            metrics = metric_set(mn_log_loss))
toc()

mn_log_tune_best <- tune_result %>% select_best(metric = "mn_log_loss")
mn_log_tune_best#$penalty
mn_log_tune_best#$mixture


acc_roc_tune_best <- tune_result %>% select_best(metric = "accuracy","roc_auc")
acc_roc_tune_best$penalty
acc_roc_tune_best$mixture

### accuracy,roc_auc 기준 : penalty-> 9.333026e-09, mixture-> 0.03507123 1300초 걸림
### mn_log_loss 기준 : penalty-> 2.832072e-08, mixture-> 0.01048237 1200초 걸림
### mn_log_loss 기준 : take 0.0000000109  0.0375

####################################    mlr vfold 계산  #########################################


mlr_spec <- multinom_reg(penalty = 0.0000000109,
                         mixture = 0.0375) %>% 
  set_engine("glmnet") %>% 
  set_mode("classification")

mlr_wflow <- 
  workflow() %>% 
  add_model(mlr_spec) %>%
  add_formula(credit ~ .)


tic()
mlr_fit_vfold <- 
  mlr_wflow %>% 
  fit_resamples(credit ~ .,
                data = train2,
                resamples = validation_split,
                metrics = metric_set(mn_log_loss),
                control = control_stack_resamples())
toc()


#######################################################################################################
########################################         stacking           ###################################
#######################################################################################################


credit_stacking <- 
  stacks() %>% 
  add_candidates(rf_fit_vfold) %>% 
  add_candidates(mlr_fit_vfold)

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
