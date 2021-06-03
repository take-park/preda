library(tidymodels)
library(tidyverse)
library(magrittr)
library(tictoc)
library(skimr)
library(kknn)
library(stacks)
library(glmnet)
library(kernlab)
library(keras)
library(ranger)
library(purrr) ; library(magrittr) ; library(MASS)

#keras를 돌리기 위해서 설치필요, 아나콘다 & 텐서플로우
#install.packages("tensorflow")
#require(tensorflow)
#install.packages("curl")
#install_tensorflow()

#######################################################################################################
#############################################      data       #########################################
#######################################################################################################

train <- read.csv("train.csv") %>% 
  janitor::clean_names()
test <- read.csv("test.csv") %>%
  janitor::clean_names()

#train data의 문자형(character data)를 요인(factor)으로 바꿔주기 & 
#예측하고자 하는 credit을 요인(factor)으로 설정하기
train %<>%
  mutate_if(is.character, as.factor) %>% 
  mutate(credit = factor(credit))


#test data의 문자형 데이터를 factor로 바꿔주기
test %<>%
  mutate_if(is.character, as.factor)


#전처리
credit_recipe <- train %>% 
  recipe(credit ~ .) %>%  #credit을 y로 하는 모형으로(credit외의 변수를 예측하는데 쓸것임) 
  step_mutate(yrs_birth = -ceiling(days_birth/365), #나이 
              yrs_employed = -ceiling(days_employed/365), #근무기간
              perincome = income_total / family_size, #인당소득
              adult_income = (family_size - child_num) * income_total, #성인소득
              begin_month = -begin_month) %>% #신용카드를 발급한지 몇달 됐니
  step_rm(index, days_birth, work_phone, phone, email) %>%  #데이터 제거
  step_unknown(occyp_type) %>% #결측값에 "unknown" factor 할당 
  step_zv(all_predictors()) %>% #분산이 0인 변수를 제거
  step_integer(all_nominal(), -all_outcomes()) %>% 
  #nominal data(순서 없이 분류된, eg 독어,영어,일어)에 각각 유닉한 정수를 부여
  step_center(all_predictors(), -all_outcomes()) %>% #평균을 0으로 만들기
  prep(training = train) #training으로 'train'data를 사용한다


train2 <- juice(credit_recipe)
test2 <- bake(credit_recipe, new_data= test)

set.seed(2021)

validation_split <- vfold_cv(train2, v = 5, strata = credit)


#######################################################################################################
#################################         random forest        ########################################
#######################################################################################################

#################################         rf 모수 추정         ########################################

                                      ## 발표자 : 백한별 ##

tune_spec <- rand_forest(mtry = tune(),
                         min_n = tune(),
                         trees = 1000) %>% 
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

set.seed(2021)
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
tune_result$.metrics

tune_best <- tune_result %>% select_best(metric = "mn_log_loss")
tune_best$mtry
tune_best$min_n

### mn_log_loss 기준 tunebest -> mtry : 3, min_n : 11
### mn_log_loss 기준   [take] -> mtry : 2, min_n : 4


#################################       rf vfold 계산      ##########################################



model_spec <- rand_forest(mtry = 2,
                          min_n = 4,
                          trees = 1000) %>% 
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

rf_workflow <- workflow() %>%
  add_model(model_spec) %>% 
  add_formula(credit ~ .)

set.seed(2021) #필요!#
param_grid <- grid_random(finalize(mtry(), x = train2[,-1]), min_n(), size = 100)

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

                                        ## 발표자 : 박태익 ##

mlr_spec <- multinom_reg(penalty = tune(),
                         mixture = tune()) %>% 
  set_engine("glmnet") %>% 
  set_mode("classification")

set.seed(2021)
mlr_grid <- grid_latin_hypercube(penalty(), mixture(),  size = 1000)
mlr_grid

mlr_wflow <- 
  workflow() %>% 
  add_model(mlr_spec) %>%
  add_formula(credit ~ .)
tic()
tune_result <- mlr_wflow %>% 
  tune_grid(validation_split,
            grid = mlr_grid,
            control = control_stack_resamples(),
            metrics = metric_set(mn_log_loss))
toc()

tune_result %>%
  collect_metrics()

tune_result %>%
  show_best()


tune_result %>%
  collect_metrics() %>%
  filter(mean < 0.86246)


tune_result %>%
  collect_metrics() %>%
  filter(mean < 0.86246) %>%
  ggplot(aes(penalty, mean, color = .metric)) +
  geom_line(size = 1.5) +
  scale_x_log10() +
  theme(legend.position = "none") +
  labs(title = "mn_log_loss")

mn_log_tune_best <- tune_result %>% select_best(metric = "mn_log_loss")
mn_log_tune_best$penalty
mn_log_tune_best$mixture

### mn_log_loss 기준 : penalty-> 0.00548831, mixture-> 0.0993676 1500초

#########################################    mlr vfold 계산  ##########################################


mlr_spec <- multinom_reg(penalty = 0.00548831,
                         mixture = 0.0993676) %>%
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

mlr_fit_vfold$.predictions
mlr_fit_vfold$.notes
mlr_fit_vfold$splits
mlr_fit_vfold$id
mlr_fit_vfold$.metrics
mlr_fit_vfold
summary(mlr_fit_vfold)


#### coefficient 값을 어떻게 확인하지?

coefficients(mlr_fit_vfold)
coef(mlr_fit_vfold)
coef(mlr_fit_vfold, s = 0.00548831)
coef.glmnet(mlr_fit_vfold, type = "coefficients")

# options(max.print = 10)
# lasso_fit %>% 
#   tidy() %>% 
#   filter(estimate > 0.001)

########################################################################################################
#                                       Deep Learning                                                  #
########################################################################################################
#                                                                                                      #
#################### keras를 사용하기 위해선 아나콘다와 텐서플로우를 install해야한다 ###################
#                                                                                                      #
########################################################################################################

                                        ## 발표자 : 손성만 ##

nn_spec   <- mlp(epochs = 100, 
                dropout = 0.2, activation = "linear",hidden_units = 5) %>% 
              set_engine("keras") %>%
              set_mode("classification")

#epochs -> 데이터를 몇 번 학습할지
#dropout -> 추정된 모수들중 몇 프로를 떨굴지
#activation -> 각 layer간의 함수 설정
#hidden_units -> input layer와 output layer 사이의 층 갯수
#662개의 데이터가 각 epochs 씩, 100번 학습되고 이것이 vfold갯수인 5개 만큼 발생된다#

nn_wflow <- 
  workflow() %>% 
  add_model(nn_spec) %>%
  add_formula(credit ~ .)

library(tictoc)
tic()
nn_fit_vfold <- 
  nn_wflow %>% 
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
  add_candidates(mlr_fit_vfold) %>%
  add_candidates(nn_fit_vfold)

credit_stacking %<>% 
  blend_predictions() %>% 
  fit_members()

result <- predict(credit_stacking, test2, type = "prob")

submission <- read_csv("sample_submission.csv")
sub_col <- names(submission)
submission <- bind_cols(submission$index, result)
names(submission) <- sub_col
write.csv(submission, row.names = FALSE,
          "stacking_rf_mlr_nn_son.csv")
