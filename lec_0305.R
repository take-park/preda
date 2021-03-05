library(tidyverse)

devtools::install_github("allisonhorst/palmerpenguins")
library(palmerpenguins)

library("palmerpenguins")

head(penguins)
summary(penguins)
structure(penguins)

a <- 1

a

myf <- function(){
  a <- 5
  a
}

myf()

## 데이터 탐색방법
summary(penguins)
table(penguins$species)
glimpse(penguins)
##

plot(penguins$species,penguins$island)

## pipe operator 연습
1:10 %>% sum()
c(1, 3, 4) %in% 2:10

penguins %>% select()
