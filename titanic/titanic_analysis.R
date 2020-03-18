
#Setting default working directory every new session.

setwd("~/Desktop/git_projects/titanic-ml/titanic")

#Get the data
train_data <- "train.csv"
test_data <- "test.csv"

#Load the data
train_set <- read.csv(train_data, header = TRUE) #We want the column names to be in the header, so header = TRUE
test_set <- read.csv(test_data, header = TRUE)

####
#Now, we need to combine both train and test set. One of the reasons we do that is because when we do some "feature transformations"
#(converting it into factor etc.) it has to be consistent both on the train and the test data. We want the same representations!
#Second most important thing: "We don't want to mess with the original data. Make copy of it and do things on it, so we can keep them as
#a reference! Later on, you will learn that for example Neural Networks (NN) accept vectors as inputs, and that you need to do some
#data preparation before puting it into NN (for example one_hot_encoding). Logically, if you do one-hot-encoding on training data, 
#then when you do testing on test_data it's expected to make it one-hot-encoded too! If NN learned representations using one-hot-encoded vectors,
#it would be bad if you put integer vectors. 
####

####
#In order to combine two dataframes in R both of them must have same number of columns (aka features).
#Survived is missing from the test_set, so we are gonna make it and populate it with the string (chr datatype in R!) "None",
#since those need to be predicted! For now we don't have that information, hence "None". 
####

#Make combined dataset
survived <- rep("None", nrow(test_set))
test_set_with_survived <- data.frame(survived, test_set) 
data_combined <- rbind(train_set, test_set_with_survived) 

####
#It's imporant that you put 'survived' in front of 'test_set' because that is the order in which dataframe will be made respectively.
#In this case first column will be 'survived' and then you will have the rest of columns of test_set.
#Conversly, if you do data.frame(test_set, survived) you will first have all the columns from 'test_set' and then 'survived' column.
#Since the column order in train_set is survived, "rest_of_the_columns" we will stick with the first case!
####

#Let's see what kind of data types we are delaing with according to R
str(data_combined)

####
#Here is another example why we made data_combined. train_data has feature survived which has values 0 or 1, test_data has "None".
#Calling str() on train_data will say that 'survived' variable is an integer type, and for test_data it would say char type.
#We need to get both. 
####

####
#Some of the feature datatypes might be converted just by looking at the metadata (data provided by competition). You see, it "doesn't"
#make too much sense to treat 'pclass' feature as an integer value. Intuitevly it should be categorical. You are not gonna do any multiplication on it, or so.
#You know that 'money' shouldn't be categorical, or for example 'age' because there is whole spectrum of them. 
#Survived should be a factor mainly because of the ML purposes. You can treat them as vectors somwhere along the line, that represent 0 and 1
#respectively. That's why I'm converting them as factors now. 'name' feature seems fishy to be categorical, but for now these are all the fixes.
#During the analysis part I will decide along the way which ones to convert to it's proper datatype. And ofcourse, as you learn more about ML and NN, and understand
#a bit how data preprocessing works for each kind of variables it will be tempting to do at this very stage all the preprocessing.
#But for now, we are purely exploring it. If we find out something about 'names' that will spark our interest when we further investigate it
#at that point we will convert it. For now, every preprocessing is in 'initial phase', for the purposes of data analysis!
####

#Convert pclass and survived as pclass
data_combined$survived <- as.factor(data_combined$survived)
data_combined$pclass <- as.factor(data_combined$pclass)

####
#We will start the analysis with the 'big picture' (aka 'overall') and then get into nitty-gritty details as we go further.
#We will do this methodically, looking at each feature separately, one by one.
#We will do this process iteratively (high level overview of features first, then more and more as we go further.)
#We will also keep in the back of our minds the idea that we are concerning with 'classification problem'.
#Moreover, we will be concerned to 'who survived, who doesn't', and look each feature relative to that. Each feature will be analysed with that idea.
#Every idea around features 'should' be at least centered around that idea.
####

#We will start with the 'survival' feature. What can one do with that? Well, we can see gross survival rate!
table(data_combined$survived)

#$$$$What can we get out of it?
#We know that most of the folks on Titanic didn't survived, that is why the data is skewed towards 'perish'.
#It is being said that ML algos tend to base their decisions on the side to which the data is skewed. 
#I think this is 'statistics part' and that it have some sense (at least in extreme case where it would be 0:890 1:1!)
#I think we have to find meaningful features, in order to help ML model to make meaningful decisions based on who will survive and who will not.
#$$$$

#Second feature = 'pclass'. What can we do with it in terms of "who survived who doesn't".

####
#Let's first look at the number of passangers per class.
####

table(data_combined$pclass)

#$$$$What can we get out of it?
#As expected, there are much more 'poor' folks. Thing that is quite interesting is that we should expect more folks in second class
#than in first class. 
####

####
#From now on, whole idea of analysis will be converging around 'plots' and survival rates against plots for each feature we encounter.
#I found it also to be nice idea. You can only get too far with table() and numbers, but one gains more insights through visual understanding.
#Again, each variable and dedicated visualisation will be thought in a context of 'survival'. That is a good way of attacking this particular problem!
#We will start from the variable 'pclass' and then each new variable will be analysied in visual context.
####

library(ggplot2)






# - - - pclass - - - 


#---> Hypothesis: Rich folks have better survival rates

####
#Since we only have survival rates for training data, that is what we will be concerned only. 
#There are three ways you can do this: 
# 1) By using data_combined first 891 rows 
# 2) By using train_set and converting each variable 'on the fly' which are not coverted previous time
#    (on the fly in order to not ruin philosophy 'Don't touch orginal train and test set.')
# 3) Mix of previous two.
# Since we have everything preprocessed in data_combined, and first 891 rows equal to train_set we will stick to 1)
####

ggplot(data_combined[1:891,], aes(x = pclass, fill = survived)) +
  geom_bar(width=0.5) + 
  xlab("Class") +
  ylab("Total number of people per class") +
  labs(fill = "Survived")

# ---> Hypotheses: Confirmed, because:
#No matter how much money you had only certain types of people can get into first and second class.
#That is what research told me. On the other hand, if we want to prove that we would need to investigate 
#and define 'high status' for that era. For example 'if someone was engineer in that era he is rich, and goes into first class'
#There are no poor engineers, except me at the moment... These are just the possibilities! Some engineers "could be" stuck in third class!
#If we presume that only rich folks can get into first and second class, just by looking at this graph we can deduce that, indeed,
#rich folks have better survival rate. So the hypothesis is confirmed. 

####
#Speaking relatively by class the first class folks survived the most. Close to that number is number of survived
#in third class. But relatively speaking even if they had same amount of survived in third class thats bad.
#It seems that pclass is relatively important to our decision making. Determining the pattern who survived who didn't.
#As always, does 'pclass' is a good feature for predicting survival rates we will see further. 
#Don't get hooked to it, until we prove alternative hypothesis!
#Just not to be confused 2/3 of first class, 1/2 of second class and about 1/4 of third class survived.
#It is the relative percent that counts, not relative number of survived. That is what this graph is conveying.
####







# - - - name - - - 




#Let's examine the first few names just to see what it's composed of, and to get a little sense of it.
#It doesn't matter which part of the dataset we will use here, we just want to see the structure, but since I'm using
#it from the train dataset, it should be again noted that I have to convert variables "on the fly" ("not touching original test and training sets")
#We don't know why is it a factor, or categorical variable. Might be some nice easter egg made purposely by Kaggle team!

head(as.character(train_set$name))

#####
#We will presume that it has structure like this, since Kaggle doesn't have any metadata about that:
# 1) "[Second Name], [Title] [First Name] [Middle Name]" for males
# 2) "[Husbands Second Name], [Title] [Husbands First and Middle Name] ([First and Middlle Name] [Maidens name]"

#####
#Now that we have some high overview of data let's see what kind of anomalies may emerge with the name variable.
#First thing that pops out is: "Are there duplicate names?". 
#One of the problems about having the same names is concerning about one of the rules for training ML model:
#"You test your model on not previously seen data". If we have for example same name John Bradely in test_set and training_set
#model will always predict it right, since it learned everything about John Bradely in training_set. 
#If we have John Bradely in test_set it's breaking the rule we mentioned earlier. 
#There might be the cases where duplicate data in training_data is good for example if there are repeated medical tests.
#But I will deal with that some other times. For this time being I will say it is not good if we encounter them.
####

#####
#Unique will return array of names that are unique by excluding duplicates, so if the length of such array is smaller
#than number_of_observations in data_combined we potentially have duplicates.
#####

length(unique(as.character(data_combined$name)))

#####
#1307<1309 So we can have a situation that those 2 names resemble one of the names (so we have three same names) 
#or each one of them is unique and thus we have duplicates. Remember, we are telling the R that we are finding uniqe rows
#and that measure of the 'uniqueness' is name feature!
#####

####
#Let's find those 2 names and see what kind of situation they resemble (3 "same" names or two duplicates)
#Here, only duplicated needs to be discussed, everything else we've seen before.
#duplicated() searches for duplicate names an returns TRUE on those that have higher subscripts
#For example if we have duplicate name at 5th row and the same name at 16th row it will return TRUE on 16th row. 
####

duplicate_names <- as.character(data_combined[which(duplicated(as.character(data_combined$name))), "name"])

####
#Now we see that those names are actually duplicates. Using duplicated_names we will take a look at those records with each feature.
####


data_combined[which(data_combined$name %in% duplicate_names),]

####
#First of all we will explain how %in% works "make a logical vector such that for each name in data_combined return TRUE
#if that name is in duplicate_names". 
#
#From data analysis perspective we can conclude that overall, those names or should we say "data points" as a whore are not
#duplicates. There are a lot of fun scenarios that could happen in pclass=3 and those people meeting each other out.
#At the end it would be fun to see are those with the same name in the test_set survived or not.
#Maybe that's one of the Kaggle's eastereggs?
#Thing that is a littlleee fishy to me is that both Connolly, Miss. Kate have similar tickets and it might be
#interesting or not if that means that they were close and met each other.
#It might be an endevour to deduce fare size using other variables and I think that Connolly, Miss. Kate might
#help us with that "good time".
#####

####
#Now that we are "sure" (hopefully!) that we ain't have duplicate names we can proceed. 
#Since "title" might be nice feature to engineer let's see if it has any correlation with other features.
#If one of the features has too much missing values, we can use new features as proxies and thus save some time with the imputation.
#It is one of the strategies for handling missing data.
#
#So, let's see if there are any correlations with other variables to see if we are on a good track to extract title as feature.
####

#We gonna import R's stringr library which is helpful for string manipulations
library(stringr)

#Let's look at the misses: 
misses <- data_combined[which(str_detect(data_combined$name, "Miss.")),]
misses[1:5,]

####
#str_detect(string, pattern_in_string) returns a logical vector showing if pattern occured in the string
#which function is "smart enought" to convert those logical table to particular numbers of rows where those occured
####

####
#From this sample few things strike my interest, but when deciding if there are any correlations between that and title
#some of them might be used in aggregate, and some not:
# 1) All of the misses from this sample survived and they are in the third class. Might potentially help us 
#    to understand who actually survived in third class. There aren't many, but not all people is dead!
# 2) There is a bit of a variance in age but in an essence we can conclude they are relatively young. 
#  At this stage we can give a hypothesis that title correlates with age. 
# Since they are "unmarried" sibsp>0 would indicate that either they are traveling with siblings or not.
# If they are widows (some prefer to be called Miss!) we would never know, but in general we can conclude they are 
# unmarried women. 
# parch>0 would indicate that either they are traveling with parents or childrens (if they are widows)
# or parent(s) and child(ren)
# It is not biologically possible that Sandstrom, Miss. Marguerite Rut could have a children thus I conclude that
# parch for her means she is traveling with parent. 
  
summary(misses$age) 

# The mean of 'miss' population is around 21.77 and 75% of misses are 30 years or less, so relatively "young population".
# Remember also that we are making judgement with 50 NA's!
# We can also give an assumption that those are non-married woman (that is what metadata is saying!) but they might
# They might travel with their (boy)friend(s) but we can investigate them only by looking at ticket's etc. 
# That might be helpfull for model to give better predictions. 
##### 

#Let's look at the mrs-es: 
mrses <- data_combined[which(str_detect(data_combined$name, "Mrs.")), ]
mrses[1:5,]

####
#From this sample we can conclude that they have high survival rate, and that they are more distributed toward upper classes 
#2 of them are in first class, and another two at second class. 4/5 are in upper classes. 
#They are relatively "old" in ages, but they are either widows or married (that is also what name feature is confirming!)

summary(mrses$age)

#Their mean is relatively higher than that in misses (36.8), and 75% of them are bellow 46.5 so relatively old!
#Another confirmation that the age might be correlated with title IN GENERAL!

#Nasser, Mrs. Nicholas (Adele Achem) raises an eyebrow but understanding how "eastern people" work (and name sounds quite "easterish")
#nothing surprises me at that age. Maybe she is too young to be married because of data collection being wrong (and indeed it does!)
#Maybe in practise her age is >insert_age_where_you_can_give_birth but this is the "real" data we are concernig with.
#We get what we measure. Thus assumptions are based on data we are provided. 

####
#Since there are many more unique titles, we will be concerned with those that make huge % of data.
#We've done with the females, so next logical step would be to see what's interesting about males!
####

males <- data_combined[which(data_combined$sex == "male"),]
males[1:5,]

#### FACTS
# From this sample we can conclude that 5/5 are dead, 4/5*100% are in the third class
# Mr can be either married or nonmarried man. Nothing about that can be deduced from their age if they are or if they are not,
# but Master is proxy for younger boy. Those are definetly nonmarried males (or we hope so!) IN GENERAL. 
# So, sibsp>1 (aswell as sibsp=0!) could be indicative of Mr. travelling with their spouses IN GENERAL!
# My gosh there is so much variability to handle and think of... 
# About titles and fares correlation, I think we need to use other features to conclude fare not just title.
# Maybe there are patterns that would help us on overall score, but that is digging tooo much into the rabbit hole!
####

####
#Since these are all the facts that concern only small subset of population (aka sample) we will expand it to see
#if we can find some insights. We are gonna extract title feature by creating utility function for doing so, and then
#we gonna use ggplot based on pclass and survived to see if there might be some real-estate in there. 
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

#####
#grep(pattern, vector)
#Returns which indexes in vector have matching patten. Remember, R indexes from 1!
#If there are any, that number would be larger than 0 and we use that information, combined with length to extract title!
#
#grep("Miss.", c("Miss. kelly", "dsddds")) --> 1
#grep("Miss.", "Miss. kelly") <=> grep("Miss.", c("Miss. kelly")) --> 1
#grep("Miss.", "krsds") --> 0
#Be aware of using Mrs. as first else if, and then Mr. because of the way grep() works!
#####

#Let's make a title vector and populate it with particular titles, based on row numbers:

titles <- NULL
for (i in 1:nrow(data_combined)) {
  titles <- c(titles, extractTitle(data_combined[i,"name"]))
}

data_combined$title <- as.factor(titles)

#Here is one of the "whys" we use data_combined. If one finds out that title is "good" feature based on both datasets 
#we can keep it and use it to train and test our ML model. If one finds out it's not good we can exclude it from both
#datasets using one command. 



#Let's gather some facts using our freshly made feature:

ggplot(data_combined[1:891,], aes(x = title, fill = survived)) +
  geom_bar() +
  facet_wrap(~pclass) + 
  ggtitle("Pclass") +
  xlab("Title") +
  ylab("Total Count") +
  labs(fill = "Survived")


######
#That higher class people had more chance of survival is confirmed this time. 
#"Women and children first" holds the water too, especially in the first and second class. 
#
#If one excludes everything except one title and see if there is any pattern, one would see that 
#everything is quite skewed to the right.
#
#For example,if you just focus on Mr. title for three classes you will see that relative survival rate (based on the percent!)
#declines as we progress from first to third class. That same logic applies for each title.
#
#Title "Others" seem to be just in first two classes so might be indicative of folks (titles) who are rich.
#
#Mr.'s in second and third class have relative survival rate almost the same (around 1/8), those in first class have better survival rate!
#This is the thing with analysis: Even if Mr.'s in third class had lower rate of survival we should explain why, and see
#those who survived, can we see using other features (but be careful not to overfit!) to find a trend for survival of folks in third class.


#Fact (and hypothesis!?): Title would be a really good predictor of survival. 







# - - - sex - - - 




####
# From the analysis of 'title' variable (misses, mrses, males etc.) we infered that age and sex (as well as some other features!) 
# seem pretty important, and that they might be correlated with title so we will take a closer look.
###

summary(data_combined$age)
###
#From the whole data_combined dataset (training+set) we see that there are a lot of NA's (around 20%!) and that is a lot of 
# missing values. There are numerus ways one can handle missing data. 
# For the ages one can for example impute missing ages with the (title etc.) mean or (title etc.) median.
# Or make a linear regression model using every other feature except age as an independent variable and age as dependent,
# make prediction's for the missing age and impute NA's with their predicted values, but keeping non-NA's as same. 
# I've seen some authors say that if one put 0 as all NA's than for example neural network eventually learn that those are missing data.
# Probably each ones have different drawbacks. 
#
# What we are gonna do here is run some "tests" to confirm that title could be a good proxy for age, so we are gonna forget about imputing missing values if we
# confirm that for each title it's age distribution relatively generalizes that title.
# And hopefully if we have "check" on each title, we are "safe" to ignore age and thus the imputation of it's missing values.
###

####
#Some conclusions that stick to my head using the summary on age on whole dataset is that people here are relatively young (one can define young however he choses!) .
# 75% of them are 40 and younger, and that 50% is less than 28.
# There are some outliers which could be justified or not based on other features.
# Mean is sliiiiiightly bigger than median which indicates sliiiiiiight skewnes to the right and thus sligghttt tendency
# toward the "bigger age".
# We once mentioned that the more the data is skewed towards something the more the tendency of an model is toward that skew.
####

summary(data_combined[1:891,"age"])

####
#One can also deduce that the training set has almost the same distribution for ages which indicates in order 
#for age distribution on a complete dataset to remain same test set has to be similarly distributed as training set.
#Linear algebra trickery?
#I'm pointing this out because some authors say that it is good/ it saves the day (?) if your training and test data have similar distribution in terms of 
# modeling. Or is it? 
# Here is the summary for the training set: 

summary(data_combined[891:1309, "age"])

# + 

summary(data_combined[1:891,"age"])

# = 

summary(data_combined$age)

###
#Now we are gonna check if we can make 'title' as proxy for age!
###