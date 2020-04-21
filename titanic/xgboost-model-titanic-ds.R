setwd("~/Desktop/git_projects/titanic-ml/titanic")

library(doSNOW)
library(caret)

train_set <- read.csv("xgboost_train_data.csv", header = TRUE)
test_set <- read.csv("test_submit_data.csv", header = TRUE)


####
#Prepare the features for the model
####

features <- c('survived', 'pclass', 'new.title', 'ticket.party.size', 'avg.fare')

#Train set

train_set_xg <- train_set[,features]
train_set_xg$pclass <- as.factor(train_set_xg$pclass)
train_set_xg$survived<- as.factor(train_set_xg$survived)
dummy.vars <- dummyVars(~ ., data = train_set_xg[, -1])
train.dummy <- predict(dummy.vars, train_set_xg[, -1])
#View(train.dummy)

train.dummy <- cbind(train.dummy[,], survived = train_set_xg[,'survived'])

#Test set

test_set_xg <- train_set[,features]
test_set_xg$pclass <- as.factor(test_set_xg$pclass)
dummy.vars <- dummyVars(~ ., data = test_set_xg[, -1])
test.dummy <- predict(dummy.vars, test_set_xg[, -1])
#View(train.dummy)



#Train control (10-fold CV, repeated 10 times; grid search for 'perfect' hyperparameters)
train.control <- trainControl(method = "repeatedcv",
                              number = 10,
                              repeats = 3,
                              search = "grid")


#Combination of parameters (University of Kaggle)
tune.grid <- expand.grid(eta = c(0.05, 0.075, 0.1),
                         nrounds = c(50, 75, 100),
                         max_depth = 6:8,
                         min_child_weight = c(2.0, 2.25, 2.5),
                         colsample_bytree = c(0.3, 0.4, 0.5),
                         gamma = 0,
                         subsample = 1)


####
#Training part
####

cl <- makeCluster(6, type = "SOCK")

registerDoSNOW(cl)

xgboost_train <- train(survived ~ .,
                       data = train.dummy,
                       method = "xgbTree", tuneGrid = tune.grid, trControl = train.control)

stopCluster(cl)

xgboost_train

saveRDS(xgboost_train, "xgboost_model_1.rds")
xg_boost_model <- readRDS("xgboost_model_1.rds")


xg_boost_preds <- predict(xg_boost_model, test.dummy)
str(xg_boost_preds)

submit_df <- data.frame(PassengerId = rep(892:1309), Survived = xg_boost_preds)
View(xg_boost_preds)

write.csv(submit_df, file = "XGBOOST_SUB_20200420_1.csv", row.names = FALSE)