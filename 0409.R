#### library

library(tidymodels)
library(tidyverse)
library(magrittr)
library(skimr)
library(knitr)


#### 1교시 :  CART ------------------------
# CART = classification And Regression Tree
# 장점 1) 설명이 쉽다. 2) 사람의 의사결정과 비슷
#      3) catagorical data를 다루기 쉬움(dummy coding 불필요)
#      4) 노말라이즈 같은 모노톤 트랜스폼에 영향을 덜 받음
#      5) robust to outliers
# 단점 1) 예측성능이 그렇게 좋진 않다.
#      2) 바운더리 값에 민감

X <- -10:10
y <- X^2

s<- -2
group1_index <- which(X < s)
X[group1_index]
X[-group1_index]
RSS <- 
  group1 <- 
  