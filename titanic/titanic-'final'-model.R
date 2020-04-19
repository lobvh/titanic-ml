
setwd("~/Desktop/git_projects/titanic-ml/titanic")

library(caret)
library(doSNOW)
library(rpart)
library(rpart.plot)
library(randomForest)
library(stringr)

#Get the data
train_data <- "train.csv"
test_data <- "test.csv"

#Load the data
train_set <- read.csv(train_data, header = TRUE) #We want the column names to be in the header, so header = TRUE
test_set <- read.csv(test_data, header = TRUE)

#Make combined dataset
survived <- rep("None", nrow(test_set))
test_set_with_survived <- data.frame(survived, test_set) 
data_combined <- rbind(train_set, test_set_with_survived) 

#Convert pclass and survived as pclass
data_combined$survived <- as.factor(data_combined$survived)
data_combined$pclass <- as.factor(data_combined$pclass)

####
#We will use whole dataset but different features for each Random Forest algorithm. 
#Note that on combined dataset labels are "1" "0" and "None", where in train dataset we already have preprocessed ones and zeroes. 
####

RF_LABELS <- as.factor(train_set$survived) #Made it upper-case since it's constant. 

set.seed(2348)
cv_10_folds <- createMultiFolds(RF_LABELS, k = 10, times = 10)

# Set up caret's trainControl object per above.
ctrl_1 <- trainControl(method = "repeatedcv", number = 10, repeats = 10,
                       index = cv_10_folds)


####
#Past titles
####

extractTitle <- function(name) {
  name <- as.character(name)
  
  if (length(grep("Miss.", name)) > 0) {
    return ("Miss.")
  } else if (length(grep("Master.", name)) > 0) {
    return ("Master.")
  } else if (length(grep("Mrs.", name)) > 0) {
    return ("Mrs.")
  } else if (length(grep("Mr.", name)) > 0) {
    return ("Mr.")
  } else {
    return ("Other")
  }
}

titles <- NULL
for (i in 1:nrow(data_combined)) {
  titles <- c(titles, extractTitle(data_combined[i,"name"]))
}

data_combined$title <- as.factor(titles)

#####
#New Titles
#####

name.splits <- str_split(data_combined$name, ",")
name.splits <- str_split(sapply(name.splits, "[", 2), " ")
titles <- sapply(name.splits, "[", 2)


# Re-map titles to be more exact
titles[titles %in% c("Dona.", "the")] <- "Lady."
titles[titles %in% c("Ms.", "Mlle.")] <- "Miss."
titles[titles == "Mme."] <- "Mrs."
titles[titles %in% c("Jonkheer.", "Don.")] <- "Sir."
titles[titles %in% c("Col.", "Capt.", "Major.")] <- "Officer"
table(titles)

# Make title a factor
data_combined$new.title <- as.factor(titles)

####
#Collapse titles based on visual analysis
####
indexes <- which(data_combined$new.title == "Lady.")
data_combined$new.title[indexes] <- "Mrs."

indexes <- which(data_combined$new.title == "Dr." | 
                   data_combined$new.title == "Rev." |
                   data_combined$new.title == "Sir." |
                   data_combined$new.title == "Officer")
data_combined$new.title[indexes] <- "Mr."


#####
#Ticket party size and Average fare feature
#####

ticket.party.size <- rep(0, nrow(data_combined))
avg.fare <- rep(0.0, nrow(data_combined))
tickets <- unique(data_combined$ticket)

for (i in 1:length(tickets)) {
  current.ticket <- tickets[i]
  party.indexes <- which(data_combined$ticket == current.ticket)
  current.avg.fare <- data_combined[party.indexes[1], "fare"] / length(party.indexes)
  
  for (k in 1:length(party.indexes)) {
    ticket.party.size[party.indexes[k]] <- length(party.indexes)
    avg.fare[party.indexes[k]] <- current.avg.fare
  }
}

data_combined$ticket.party.size <- ticket.party.size
data_combined$avg.fare <- avg.fare


####
#Decision tree training function:
####

rpart.cv <- function(seed, training, labels, ctrl) {
  cl <- makeCluster(6, type = "SOCK")
  registerDoSNOW(cl)
  
  set.seed(seed)
  
  # Leverage formula interface for training
  rpart.cv <- train(x = training, y = labels, method = "rpart", tuneLength = 30, 
                    trControl = ctrl)
  
  #Shutdown cluster
  stopCluster(cl)
  
  return (rpart.cv)
}







  ##############################################################################################################
  #                                            MODELING                                                        #
  ##############################################################################################################


####
#Decision tree// Just to confirm that I loaded everything the same!
####


features <- c("pclass", "new.title", "ticket.party.size", "avg.fare")
rpart.train.4 <- data_combined[1:891, features]
rpart.4.cv.1 <- rpart.cv(3242, rpart.train.4, RF_LABELS, ctrl_1)
rpart.4.cv.1

prp(rpart.4.cv.1$finalModel, type = 0, extra = 1, under = TRUE)

###
#Let's check Kaggle score:
###


test_submit_data <- data_combined[892:1309, features]

rpart_preds <- predict(rpart.4.cv.1$finalModel, test_submit_data, type = "class")

submit_df <- data.frame(PassengerId = rep(892:1309), Survived = rpart_preds)

write.csv(submit_df, file = "RPART_SUB_20200419_1.csv", row.names = FALSE)


########################################################################################

####
#Random Forest
####

features <- c("pclass", "new.title", "ticket.party.size", "avg.fare")
rf_train_data <- data_combined[1:891, features]

set.seed(1234)
rf_temp <- randomForest(x = rf_train_data, y = RF_LABELS, ntree = 1000)
rf_temp

test_submit_data <- data_combined[892:1309, features]

rf_preds <- predict(rf_temp, test_submit_data)
table(rf_preds)

# Write out a CSV file for submission to Kaggle
submit_df <- data.frame(PassengerId = rep(892:1309), Survived = rf_preds)

write.csv(submit_df, file = "RF_SUB_20200419_1.csv", row.names = FALSE)

#=================================================================
# XGBoost
#=================================================================

####
#10 folds CV repeated 3 times, with the gird search that tries all the permutations of tuning parameters in order to find the best model.
####

features <- c("pclass", "new.title", "ticket.party.size", "avg.fare")
xgboost_train_data <- data_combined[1:891, features]

train.control <- trainControl(method = "repeatedcv",
                              number = 10,
                              repeats = 3,
                              search = "grid")

####
#These are found on the Internet. I know how in an essence XGBoost works, but I like the way professionals think.
####

tune.grid <- expand.grid(eta = c(0.05, 0.075, 0.1),
                         nrounds = c(50, 75, 100),
                         max_depth = 6:8,
                         min_child_weight = c(2.0, 2.25, 2.5),
                         colsample_bytree = c(0.3, 0.4, 0.5),
                         gamma = 0,
                         subsample = 1)


cl <- makeCluster(6, type = "SOCK")

registerDoSNOW(cl)

xgboost_train <- train(x = xgboost_train_data, y = RF_LABELS, method = "xgbTree", tuneGrid = tune.grid, trControl = train.control)

stopCluster(cl)


preds <- predict(caret.cv, titanic.test)