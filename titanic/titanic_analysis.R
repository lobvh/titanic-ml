
####
#This is based on a Introductory to Data Science with R by David Langer.
#I've used these to teach myself some concepts, and to further extend Dave's work, so it's not pure copy/paste!
###

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

#####
# What's the distribution of females to males across train & test?
####
table(data_combined$sex)

#####
#Some things to point out is that data is skewed to the male spectrum, and if most of the males died and most of the 
#females survived models tend to be in a favour of that. Males are more likely to perish than females so model could
#possibly deduce that "if you are a female you survive, if you are a male you are dead."
####

# Visualize the 3-way relationship of sex, pclass, and survival, compare to analysis of title
ggplot(data_combined[1:891,], aes(x = sex, fill = survived)) +
  geom_bar() +
  facet_wrap(~pclass) + 
  ggtitle("Pclass") +
  xlab("Sex") +
  ylab("Total Count") +
  labs(fill = "Survived")

###
#One can see that as we go from the first class to the third there is a trend in females to survive far more likely than males.
#The odds in first and second class are far better than in the third class. 
#But for the males that rate of survival is (so to say) dramatically decrease and that in percentage second and third class males
#have "same" survival rate. But as we saw from the title feature it gives us a better picture in who survives as a male or female.
###




# - - - age - - - 




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


# Just to be thorough, take a look at survival rates broken out by sex, pclass, and age
ggplot(data_combined[1:891,], aes(x = age, fill = survived)) +
  facet_wrap(~sex + pclass) +
  geom_histogram(binwidth = 10) +
  xlab("Age") +
  ylab("Total Count")

####
#We see in aggregate the confirmation that female tend to have better survival rates than men.
#For the females there is not so much of a pattern: class_1 vs class_2 they tend mostly to survive.
#For the class_3 younger ones tend to have better survival rate, and as we increase the number of age their chance of survival decreases.
#For the males, well class_1 has overall better survival rate than in class_2 and class_3.
#One can infer that younger pals have a good survival rate.
#Taking all this into aggregate (females survive better no matter the age + males who are younger tend to have better survival age)
#we can "confirm" our hypothesis "Women and children first" (all women including the younger ones and young males have good survival rate)
####


###
#Now we are gonna check if we can make 'title' as proxy for age!
###

# Validate that "Master." is a good proxy for male children

#First we gonna subset all the datapoints with title master...
boys <- data_combined[which(data_combined$title == "Master."),]

#... and take summary statistics:
summary(boys$age)

###
#Statistics are in favour of us and pretty much informative that yes, Master could be a good proxy for male children.
#Maximum age for the given distribution is 14.5, minimum is 0.33. 75% are less than 9 years old, so one can conclude
#that there were boys who are relatively young in age. 
###

###
#Reflexively, if this is the distribution for "boys" then everything else in general would be "adult males".
#That's why we are not gonna subset Mr's, but let's take a quick summary to confirm it:
summary(data_combined[which(data_combined$title == "Mr."),]$age)
#In general there is a good separation between males of title Master. and Mr., and we are gonna show it
#using density plots: 

#First we have to 'update' subset of males, since we added new feature 'title'
males_with_title <- data_combined[which(data_combined$sex == "male"),]

ggplot(males_with_title[males_with_title$title != "Other",], aes(x = age, fill = title)) +
  geom_density(alpha = 0.5) +
  xlab("Age") +
  ylab("Total Count")

#One can see a decently clear separation between males of title Master. and Mr.
###

####
#We know that "Miss." is more complicated, let's examine further
#One can use the same variable name 'misses' in a sense of 'updating' existing variable and thus
#saving some memory. In order to be no confusion we will use different name for variable.
####

misses_updated <- data_combined[which(data_combined$title == "Miss."),]
mrses_updated <- data_combined[which(data_combined$title == "Mrs."),]

summary(misses_updated$age)
summary(mrses_updated$age)

###
#One can see that, yes, there is an 'obvious' separation between the titles of Miss-es and Mrs-es.
#As we observed previously Mrs-es are more "older" in general than Miss-es, but one should expect women that are not married passing some age!
#It's nothing unusual, one should've expect that. 
#But the problem would be "how to distinguish female children vs adult non-maried Miss".
#We see that 25% of the Misses are 15 years or less, and it seems that the data is "skewed" which means that large portion of them are older/adult.
#Let's plot two density distributions for Miss-es and Mrs-es.
###

females <- data_combined[which(data_combined$sex == "female"),]

ggplot(females[females$title != "Other",], aes(x = age, fill = title)) +
  geom_density(alpha = 0.5) +
  xlab("Age") +
  ylab("Total Count")

###
#As we expected, there is no clear separation between those two titles. 
#One should expect that there are married women to be younger in age ('blue part in red part'), 
#as well as that there are nonmarried women who are "older" in age.
#One could make many more inferencies from this density plot, but since this is iterative process I'm not gonna infer anything more.
#We see that we somehow need to 'extract children' both from Mrs-es and Miss-es.
#Let's see if other variables could help us with that!
###

####
#Sometimes I think that by understanding our data and feature engineering (aka making something explicit) we are helping not only ourselves,
#but also a ML model so it could make better predictions. Why is then data exploration so boring after all?!
####

###
#Enough for motivational messages. Let's see if pclass could help us better separate at least female children from Miss.
#After all, I forgot to mention that if the "child" is already married, it is no more child. It is an adult. Thats why we don't care for Mrs part of children.
####

ggplot(misses[misses$survived != "None",], aes(x = age, fill = survived)) +
  facet_wrap(~pclass) +
  geom_histogram(binwidth = 5) +
  ggtitle("Age for 'Miss.' by Pclass") + 
  xlab("Age") +
  ylab("Total Count")

####
#Mainly younger Misses (< or = 20 y.o) are from second and third class. 
#Since we have title of Master that is at max 14.5y.o. we will generalize "female children" as so!
####

####
#One of the ways we can distinguish or so to say separate female children from the rest is by using heuristic:
#"Children don't travel alone, even at this day and age. Especially 18 or less."
#Let's sift and create misses_alone variable:

misses_alone <- misses_updated[which(misses_updated$sibsp == 0 & misses_updated$parch == 0),]
summary(misses_alone$age)
length(which(misses_alone$age <= 14.5))

#We see that 25% of them are 21y.o. or less, and the rest of them is relatively old!
#Since male children aka Master is assumed to be 14.5y.o. or less we will use same heuristic to distinguish female children.
#Since there is only 4 of them we can conclude that rest of the misses are children. 
#We might be wrong, but not that wrong!

misses_rest <- misses_updated[which(misses_updated$sibsp != 0 & misses_updated$parch != 0),]

#We will use "helping-variable" in order to help us make density plot where we could separate misses_alone
#and misses_rest. 

misses_ph <- rep("Alone", nrow(misses_alone))
misses_alone_1 <- data.frame(misses_alone, misses_ph)

misses_ph <- rep("Not Alone", nrow(misses_rest))
misses_rest_1 <- data.frame(misses_rest, misses_ph)

misses_updated_1 <- rbind(misses_alone_1, misses_rest_1)

ggplot(misses_updated_1, aes(x = age, fill = as.factor(misses_ph))) +
  geom_density(alpha = 0.5) +
  xlab("Age") +
  ylab("Total Count") +
  labs(fill = "Traveling:") +
  ggtitle("Misses separation")

####
#If one focuses only on the part 20y.o. and less one can see that our misses_alone is a decent heuristic
#for separating younger women. Good proportion of graph indicates that there are many younger women who are not traveling alone!
#Yes, there are other misses who could travel with their friends etc. but I will stick to this. 
###



# - - - sibsp - - - 


#####
#Move on to the sibsp variable, summarize the variable
summary(data_combined$sibsp)

####
#One can see that median value is 0 which means that 50% of the passangers are traveling without sibling(s) or spouse
#that mean is so close to the mean which means (pun intended) tendency towards 0 and that the maximum value is 8.
#Nothing so special in my opinion, and can't conclude any meaningful stuff from this yet.
#Maybe we should treat it as a factor if there is a reasonable sense number of distinct values for sibsp. 
#That would imensly help us to make some visualizations.
####
length(unique(data_combined$sibsp))
#7 is of a reasonable size so we will indeed treat it as factor.

data_combined$sibsp <- as.factor(data_combined$sibsp)

####
#From here we can make many graphs and get many different analysis of what seems resonable from an analysis perspective.
#One can make analysis relative to any or collection of features. 
#I don't want to get into rabbit hole concluding that for example maybe it doesn't seem reasonable to look sibsp relative to name in terms of survival rate.
#Maybe it will help us to extract some features (combination of name and sibsp variable) in the future, but for now I will spare myself of that.
#I think for example that fare and sibsp have some good correlation and including both features would be redundant. 
#Since title feature encompasses both sex and age maybe it makes more sense to watch it relative to that, and maybe segment it ("facet wrap it") relative to pclass. 

ggplot(data_combined[1:891,], aes(x = sibsp, fill = survived)) +
  geom_bar() +
  facet_wrap(~pclass + title) + 
  ggtitle("Pclass, Title") +
  xlab("SibSp") +
  ylab("Total Count") +
  ylim(0,300) +
  labs(fill = "Survived")

####
#There are predominantly people who travel with 0 or 1 sibsp. That's what summary on sibsp confirmed!
#Those who travel without sibling or spouse always have the best survival rate. 
#Women and children first holds water here. 
#Rich folks survived the most holds the water too.
#I don't see too much signal here. Might come back for revision!
####


####
#Not gonna get into too much detail of why it might be intuitive to treat parch as factor. 
#At least for the sake of visualization it is possible to make some conclusions if parch is factor. 
###

data_combined$parch <- as.factor(data_combined$parch)
ggplot(data_combined[1:891,], aes(x = parch, fill = survived)) +
  geom_bar() +
  facet_wrap(~pclass + title) + 
  ggtitle("Pclass, Title") +
  xlab("ParCh") +
  ylab("Total Count") +
  ylim(0,300) +
  labs(fill = "Survived")

####
#If you switch in R between plots Pclass, Title vs Sibsp and Pclass, Title vs Parch you can intuit that
#graphs are pretty similar! I'm gonna argue (I keep switching between them while I'm typing this!) that
#survival rates for Mr. is the same in first class when sibs=0, but it's quite worse for the ones in second and third class
#even if data here is a bit skewed toward parch = 0. 
#Speaking relatively to sibsp = 0 one can see that number of those who survived is the same in parch = 0,
#but those who perished is increased! 
#Maybe someone can conclude something out of it. 
#I might be wrong, but I think making model on parch will yield worse results for males than by using sibsp. 
####


###
#But maybe if we combine them we will get a feature that is much more expresive! 
#Combination of those we will call 'family-size'
###

temp_sibsp <- c(train_set$sibsp, test_set$sibsp) # Since we made
temp_parch <- c(train_set$parch, test_set$parch) #              these two to be factors, and now we need numbers!
data_combined$family_size <- as.factor(temp_sibsp + temp_parch + 1) #If parch and sibsp are factors...

####
#Again, won't argue too much why it might lead us astray to have some connections between name variable for example and new feature we engineered. 
####

ggplot(data_combined[1:891,], aes(x = family_size, fill = survived)) +
  geom_bar() +
  facet_wrap(~pclass + title) + 
  ggtitle("Pclass, Title") +
  xlab("family.size") +
  ylab("Total Count") +
  ylim(0,300) +
  labs(fill = "Survived")

###
#It's much informative in term of "survival trend" among the classes and titles, and maybe even obvious that
#if you have large family and it might be problem for you to keep everyone around, thus the chances of survival are bad.
###



# - - - ticket - - - 

#####
#Just before we start I want to emphasise something called overfiting. 
#I would be precise and say "overfiting on training data" since we are building our model based on training data.
#We could make a model that could be 100% precise, for example "if you are that and that name, you survive/perish" and build a model on top of that.
#On top of 'name' variables. When we imput some new data (test data) and there is no data with that name in the training set we are ruined.
#That is why we need a model that GENERALIZES on all training data. So when you see those graphs that map each data point precisely, that is the sign of overfitting. 
#####


####
#On to the ticket variable. As Dave pointed out, one should get used to write str() for each variable whenever we start analyzing it.
#It's a shorthand for using str() on a combined dataset.
####
str(data_combined$ticket)

####
#Based on the huge number of levels ticket really isn't a factor variable it is a string, so let's convert it.
#Then we will show first 50 of them. We could use 'any number'. It's for a sake of giving us a bigger picture aka we will better see pattern
#if there are relatively high number of data points, especially for this kind of variable. 
####
data_combined$ticket <- as.character(data_combined$ticket)
data_combined$ticket[1:50]

####
#There's no immediately apparent structure in the data, let's see if we can find some. I saw on the internet that those indicate something but won't delve deep into that.
#We'll start with taking a look at just the first char for each.
#It's easy for visualization if there are relatively few (we can make them as factor!)
#It's not necessarily correct way of extracting feature, or that one should expect something out of it. 
#This is some 'basic heuristic'. 
####

ticket_first_char <- ifelse(data_combined$ticket == "", " ", substr(data_combined$ticket,1,1))
unique(ticket_first_char)

####
#OK, we can make a factor for analysis purposes and visualize
###

data_combined$ticket_first_char <- as.factor(ticket_first_char)


# First, a high-level plot of the data
ggplot(data_combined[1:891,], aes(x = ticket_first_char, fill = survived)) +
  geom_bar() +
  ggtitle("Survivability by ticket.first.char") +
  xlab("ticket.first.char") +
  ylab("Total Count") +
  ylim(0,350) +
  labs(fill = "Survived")



####
#One might conclude from the plot that since we know that those tickets that start with 1, 2 and 3 might be indicators of pclass.
#But there are more people in first class that in second:
table(data_combined$pclass)
#Maybe those from 4, 5, etc. could be also grouped into first, second and third class in order to confirm previous hypothesis?
#It's quite a mixed bag of survivability, and we hate overfiting... Might be predictive, but let's drill a bit more!
####

ggplot(data_combined[1:891,], aes(x = ticket_first_char, fill = survived)) +
  geom_bar() +
  facet_wrap(~pclass) + 
  ggtitle("Pclass") +
  xlab("ticket.first.char") +
  ylab("Total Count") +
  ylim(0,300) +
  labs(fill = "Survived")

###
#We can see that majority tickets in each class start with the same number as pclass they are in. 
#If we put that stack of P on top of 1 in the first class we will have "that amount of people" which
#further implies that our previous hypothesis is true! That majority of imbalance for first class is "hidding" in the P?
#There is W in each class, and very few of them. Might indicate some kind of people that are indeed passengers, but maybe some workers or something?
#There might be some signal here, but I will keep it in back of my head.
###

####
# Lastly, see if we get a pattern when using combination of pclass & title
###

ggplot(data_combined[1:891,], aes(x = ticket_first_char, fill = survived)) +
  geom_bar() +
  facet_wrap(~pclass + title) + 
  ggtitle("Pclass, Title") +
  xlab("ticket.first.char") +
  ylab("Total Count") +
  ylim(0,200) +
  labs(fill = "Survived")

###
#Again, this is based just on a frist letter of ticket. Everything matches with our intuition about survivability, but I don't
#see some patterns that stick out here. Maybe thorough investigation of ticket feature will be more fruitful.
###






# - - - fare - - - 



####
#Next up - the fares Titanic passengers paid
####

str(data_combined$fare)
summary(data_combined$fare)
length(unique(data_combined$fare))


###
#We can't make it a factor... Too much instances. 
###

###High level overview
ggplot(data_combined, aes(x = fare)) +
  geom_histogram(binwidth = 5) +
  ggtitle("Combined Fare Distribution") +
  xlab("Fare") +
  ylab("Total Count") +
  ylim(0,200)

####
#There are some folks that haven't paid anything? That could be interesting. 
#We see that distribution is skewed towards higher end. We should expect that, because majority of folks that were on Titanic were in third class.
#aka low fares. We confirmed that with median<mean, and also median = 14.454. 
#There is an outlier up there where fare>500, but further investigation of fare variable will tell us why is that so.
###


####
#Now that we have high level overview, let's see if it's predictive in some sense:
####

ggplot(data_combined[1:891,], aes(x = fare, fill = survived)) +
  geom_histogram(binwidth = 5) +
  facet_wrap(~pclass + title) + 
  ggtitle("Pclass, Title") +
  xlab("fare") +
  ylab("Total Count") +
  ylim(0,50) + 
  labs(fill = "Survived")


####
#One would expect that rich folks survived more, and indeed that is so. I don't see anything that will cause more than overfitting here, because
#everytihing matches our previous intuitions in terms of survivability. I will leave fare for now, but it might help me to feature engineer something using it.
####


                ################################################################################################
                #                               EXPLANATORY DATA ANALYSIS                                      #
                ################################################################################################


####
#
#This is the part where we check our intuitions about features, and also test if our feature engineering is worthwhile.
#I think "all" of the ML algos have explicit or implicit way of providing us with the feature importance. 
#We need something that is fast (for classification problems!), effective and simple to interpret. 
#Without worrying too much on hyperparameters. We will leave that fine tuning for the "real" modeling part. 
#
#
#We will use Random Forests here. I wont drill about the algorithm here there is plenty of it on Internet. 
#In an essence it uses ensemble method of trees and averages loss on each tree. 
#Each tree gets it bootstraped sample (drawing N samples from training set where N is number of rows in training set).
#By drawing samples with replacement some of the rows won't be sampled. Those are colled out-of-bag. 
#Evaluate model on samples that you drawed, test the model on non-sampled (out-of-bag) ones.
#What confuses most people is when you are evaluating loss on that particular tree OOB sample is HIDDING it's labels.
#You get the predicted ones, then you compare it with the real labels and calculate loss.
#Draw samples -> Make model (tree) on them -> Test the model (tree) on OOB by hiding it's labels -> Compare real labels of OOB with predicted ones -> Calculate loss
#Then you repeat that proces for each tree and compute the average loss. 
#Now watch out, since you are drawing samples RANDOMLY with replacement some of the OOB's from the first tree won't be OOB on some other trees. 
#That's why validation on test data is way more accurate of overall accuracy than OOB since each data point from test set has been put
#through EACH TREE in Random Forest, where some of the OOB samples are not tested on all trees. 
#There is also (I guess!) a high probability that EACH training sample will be picked in some of the OOB's at least once, so that's why in confusion matrix you will get score for each row.
#
#That's a high-overview enough to get you started. I'm not an expert on RF's...
#
####



###
#Let's import the randomForest library.
###

library(randomForest)


####
#We will use whole dataset but different features for each Random Forest algorithm. 
#Note that on combined dataset labels are "1" "0" and "None", where in train dataset we already have preprocessed ones and zeroes. 
####

RF_LABELS <- as.factor(train_set$survived) #Made it upper-case since it's constant. 

####
#We said that pclass and title are by far the best predictors. By using idea of Occam's Razor we will train our model just on two features.
#set.seed() is used for reproducability. Let's just say that it helps drawing the same samples for trees each time we run algo, 
#and that everything that it's varying are the features we use to train RF model.
#
#We set importance to TRUE in order to make feature importance explicit, and ntree is... well I wont philosophise too much about that.
#500 trees is by default. Probably no matter how big your training data is or how many features you use after 500 trees accuracy pretty much plateus. 
#And developers decide to make it 500 by default.
#I think the OOB will be a bit accurate when we use more trees since there is high probability that your OOB samples would be
#introudced in more trees, but that's just intuition. It might be wrong. And, since RFs are fast few more trees wont ruin time metric here. 
####


rf_train_1 <- data_combined[1:891, c("pclass", "title")]

set.seed(1234)
random_forest_1 <- randomForest(x = rf_train_1, y = RF_LABELS, importance = TRUE, ntree = 1000)
random_forest_1
varImpPlot(random_forest_1) #This one is for PLOTing feature (VARiable) IMPortance

#####
#So our OOB accuracy is 20.99% which is not bad. Don't get into a trap thinking that this will match Kaggle accuracy.
#Put a test set into this model and get a submission to Kaggle. 
#We see that this model is good at predicting "pesimistic results" since class.error for those who perish is around 0.024, but bad at predicting optimistic ones.
#We need to have in mind that we must minimize both of these errors. Kaggle gives both accuracies the same weight, that is 
#it cares for percentage of good predicted for those who perish, and for those who survive.
#Since (here) upper part of confusion matrix represents Predicted values, and horizontal line represents actuall values class.error is calculated as
#13/536+13 and 174/174+168, which is respectively false positive rate, and true positive rate (sensitivity/recall).
#Intuitevely, if we want to improve overall class error one of the strategy would be to put as much 13s into 536, and 174s into 168s. 
#That comes with a price! I think those kind of manipulations are what ROC is preaching: "sacrifising something for something else".
#####

####
#We saw that sibsp, parch and family size might be predictable let's see.
###


####
#Train a Random Forest using pclass, title, & sibsp
###

rf_train_2 <- data_combined[1:891, c("pclass", "title", "sibsp")]

set.seed(1234)
random_forest_2 <- randomForest(x = rf_train_2, y = RF_LABELS, importance = TRUE, ntree = 1000)
random_forest_2
varImpPlot(random_forest_2)

####
#Not only that OOB improved, but also overall error rate! It droped on predicting pessimistic results, but "vastly" improved on predicting ones who survived.
#Better than random_forest_1. 
#One of the things I should've mentioned is that by drawing plots of variable importance you see which ones are good, and which ones don't add too much signal. 
#There is no point of comparing two features, but it seems more meaningful for more than two!
####

####
#Train a Random Forest using pclass, title, & parch
####


rf_train_3 <- data_combined[1:891, c("pclass", "title", "parch")]

set.seed(1234)
random_forest_3 <- randomForest(x = rf_train_3, y = RF_LABELS, importance = TRUE, ntree = 1000)
random_forest_3
varImpPlot(random_forest_3)

###
#One can see that this one has slightly worse OOB than that on sibsp. I would argue that sibsp and parch are similar in nature, and that some
#values of sibsp is making RF model better. Intuitevely, maybe there is something in the number of siblings or spouses that gives you higher chance of determining surivalism 
#then by having parent or children. Maybe there is some advantage you can think of: "children are hard to get together especially if you have more of them, parent's are older and thus have less velocity to get to the upper decks"
#but something grinds my gears since I don't know if random forest intuits this the same way. I'll definetly put some numbers and expect from random forest meaning of life.
#Would it be a happy choice? Satre doesn't thinks so... 
###

#####
#Train a Random Forest using pclass, title, sibsp, parch. 
#Maybe when we use them in aggregate the RF will intuit groups and see that those with bigger families tend not to survive?
#####

rf_train_4 <- data_combined[1:891, c("pclass", "title", "sibsp", "parch")]

set.seed(1234)
random_forest_4 <- randomForest(x = rf_train_4, y = RF_LABELS, importance = TRUE, ntree = 1000)
random_forest_4
varImpPlot(random_forest_4)

####
#Well, seems like our intuition that ML algos pick humane intuitions validate here. The OOB is by far the lowest, and class errors are good. 
#Since ML algos can "maybe" intuit some things, we will assume that it needs aggregation of sibsp and parch to get a better feeling for "familyness".
#There is also that part "traveling all alone". Maybe algo doesn't understand what is the total family size.
####

####
#Train a Random Forest using pclass, title, & family.size
####

rf_train_5 <- data_combined[1:891, c("pclass", "title", "family_size")]

set.seed(1234)
random_forest_5 <- randomForest(x = rf_train_5, y = RF_LABELS, importance = TRUE, ntree = 1000)
random_forest_5
varImpPlot(random_forest_5)

####
#So this gives even better results! We explicitly engineered feature family_size and we see that it helps.
#Now we will see if that is so, let's also add parch and sibsp separately, and then parch and sibsp together. 
####


####
#Train a Random Forest using pclass, title, sibsp, & family_size
####
rf_train_6 <- data_combined[1:891, c("pclass", "title", "sibsp", "family_size")]

set.seed(1234)
random_forest_6 <- randomForest(x = rf_train_6, y = RF_LABELS, importance = TRUE, ntree = 1000)
random_forest_6
varImpPlot(random_forest_6)

####
#It got a little worst, but as our intuition is that sibsp is more predictive than parch, let's see if that holds the water!
####


#####
#Train a Random Forest using pclass, title, parch, & family_size
#####

rf_train_7 <- data_combined[1:891, c("pclass", "title", "parch", "family_size")]

set.seed(1234)
random_forest_7 <- randomForest(x = rf_train_7, y = RF_LABELS, importance = TRUE, ntree = 1000)
random_forest_7
varImpPlot(random_forest_7)

####
#Yeah, that might be true.
#Lastly:
####


#####
#Train a Random Forest using pclass, title, parch, sibsp & family_size
#####

rf_train_8 <- data_combined[1:891, c("pclass", "title", "sibsp", "parch", "family_size")]

set.seed(1234)
random_forest_8 <- randomForest(x = rf_train_8, y = RF_LABELS, importance = TRUE, ntree = 1000)
random_forest_8
varImpPlot(random_forest_8)

####
#Maybe our intuition that RF catches the same one is not necessarily good, and we should prefer simpler model.
#So far the best results were yielded using random_forest_5.
####





            ################################################################################################
            #                                      CROSS VALIDATION                                        #
            ################################################################################################


####
#So, the main idea of this part will be to explain a bit what CV is, and why do we leverage it. 
#In an essence it gives almost "real" estimate of what one should expect in terms of accuracy of model. 
#Here we will prove that OOB is not indeed our true estimate of accuracy. We explained why way up there.
#Let's put our submission to Kaggle:

test_submit_dataframe <- data_combined[892:1309, c("pclass", "title", "family_size")] 

####
#These are the features we used to train random_forest_5. Now, we will use our RF model to predict values on never before seen data.
#How good are those predictions will be based on Kaggles estimate.
####


random_forest_5_predictions <- predict(random_forest_5, test_submit_dataframe)
table(random_forest_5_predictions)

####
#As we pointed out elsewhere, we might expect such result to "predict" more of those who perished. 
#We are just checking how much it predicted. 
####

####
#Write out a CSV file for submission to Kaggle. Needs to be same format as Kaggle's.
#If you view in R submit_dataframe you will see there are "extra" row namings, hence row.names = FALSE.
####

submit_dataframe <- data.frame(PassengerId = rep(892:1309), Survived = random_forest_5_predictions)
write.csv(submit_dataframe, file = "RF_SUB_20200904_1.csv", row.names = FALSE)


####
#It's pretty obvious how to upload your submission to Kaggle, so I'm gonna skip that part.
#My submission score is 0.79425 but the OOB predicts that we should score 0.8182. We overfitted the training data.
####

####
#Here, caret package uses stratified cross validation.
#The main idea is a methodology of spliting the data, and iteratively training and testing on all training dataset.
#Everyone suggestes that 10 fold cross validation repeated 10 times is first step to any ML modeling. 
#We will start with that. Instead of constantly puting submissions to Kaggle and checking score, and the fact you only have 5 sumbissions per day
#we need to find some methodology to calibrate our results localy. We use CV for that. The main idea is to calibrate our score of random_forest_5 
#to the point of Kaggle's prediction for this score, that is 0.79426. Everything else is shooting in dark. 
#
#As Dave puts:
#"When Kaggle competitions close the final results are calculated using a private data set that is different than either
#the training and test data that is provided publicly. Given that you submit only your predictions to the Kaggle web site, 
#Kaggle cannot know what features you used to build your model. While I've never researched how exactly Kaggle does this, there has to be a model(s) that they use that translate the patterns of your model's performance on the test data set to performance on the private data."
####

####
#To leverage the CV methodology we will use doSnow for parallel processing, and caret for creating folds.
####


library(caret)
library(doSNOW)


####
#We will start with the 10-fold-stratified-CV, repeated 10 times. To put it simply: "Shuffle the training set and make me 10 stratified folds, each fold containing 1/10 of the training data.
#Stratified means that each fold is ballanced and to have "same number" of those who survived and those who perished.
#Imagine that you have 9 folds for training where the 10th one is only with those who survived.
#If you are a bit towards math you will see that 9 separate folds with same percentage of survived/perished will in aggregation yield same percentage.
#Why should 10th one be with the same percentage? Well, that one which is used in the holdout set will be included in the training in the next 9 folds, so we have to keep it the same percentage.
#Enough philosophy. If you are curious you can search the Internet for what satisfies your inner "I want to understand."
#I just want to mention that stratification is used for making sure that there are no "hidden features in data", thats why we need to balance it. 
#Repeated 10 times will induce even more randomization aka "trying all the possible ways of making test and training data, to ensure higher accuracy"
####


set.seed(2348)
cv_10_folds <- createMultiFolds(RF_LABELS, k = 10, times = 10)

#Check stratification
table(RF_LABELS)
342 / 549

table(RF_LABELS[cv_10_folds[[33]]])
308 / 494

# Set up caret's trainControl object per above.
ctrl_1 <- trainControl(method = "repeatedcv", number = 10, repeats = 10,
                       index = cv_10_folds)


####
#At this stage check my detecting_cores.r file I've included to check how many aviable cores and threads you have to use.
#I'm not responsible for any damage.
###

###
#Create cluster of child processes, and "register" them
###
cluster_1 <- makeCluster(6, type = "SOCK")
registerDoSNOW(cluster_1)

###
#Set seed for reproducability and train RF algorithm on the same dataset, with the same features as random_forest_5 via using cross-validation
###
set.seed(34324)
rf_5_cv_1 <- train(x = rf_train_5, y = RF_LABELS, method = "rf", tuneLength = 3,
                   ntree = 1000, trControl = ctrl_1)

###
#Shutdown cluster after it finishes 
###
stopCluster(cluster_1)

###
#Check out results 
###

rf_5_cv_1

###
#We see that the best accuracy is using mtry=2. that is 0.8105179. Our random_forest_5 accuracy is 81.82, and our Kaggle accuracy
#is 0.79425. So, our CV score is a bit more optimistic, and we see that RF is also overfitting. 
#We are training on 90% of data and evaluating our test on 10% of data. Maybe the problem is in having too much data for our RF, aka RF is more prone to overfit then on less data.
#That beign said, we will try to train our RF on less data using 5-fold CV repeated 10 times. 
#Same code again, I won't drain it. 
###


####
#5-folds cross-validation
####
set.seed(5983)
cv_5_folds <- createMultiFolds(RF_LABELS, k = 5, times = 10)

ctrl_2 <- trainControl(method = "repeatedcv", number = 5, repeats = 10,
                       index = cv_5_folds)


cluster_1 <- makeCluster(6, type = "SOCK")
registerDoSNOW(cluster_1)

set.seed(89472)
rf_5_cv_2 <- train(x = rf_train_5, y = RF_LABELS, method = "rf", tuneLength = 3,
                   ntree = 1000, trControl = ctrl_2)

stopCluster(cluster_1)


rf_5_cv_2

###
#Now our best accuracy is 0.8134652 which is quite "worse" then those obtained from 10-fold CV.
#We expected it to be more pessimistic. So finally, we will use the heuristic which is used by some Kagglers:
#Make such CV where folds mimics percentage of the training vs test set on complete dataset. 
#Here is around 1/3, that is 2/3 is training set and 1/3 is test set. So, we will do 3-fold cross validation.
###


####
#3-folds cross-validation
####
set.seed(37596)
cv_3_folds <- createMultiFolds(RF_LABELS, k = 3, times = 10)

ctrl_3 <- trainControl(method = "repeatedcv", number = 3, repeats = 10,
                       index = cv_3_folds)


cluster_1 <- makeCluster(6, type = "SOCK")
registerDoSNOW(cluster_1)

set.seed(94622)
rf_5_cv_3 <- train(x = rf_train_5, y = RF_LABELS, method = "rf", tuneLength = 3,
                   ntree = 1000, trControl = ctrl_3)

stopCluster(cluster_1)


rf_5_cv_3

#### 
# So far:
# rf_5_cv_10 -> 0.8105179
# rf_5_cv_5 -> 0.8123434
# rf_5_cv_3 -> 0.8139169 
#Dave had the worst score on rf_5_cv_3 and he expected that, I'm gonna stick with the same one also.
#Maybe in this particular case it yields these scores but in future they will be "better". 
#Since I'm using 3-fold CV I will be wrong, or should I say I expect that whenever I have score it will be wrong 0.8139169 - 0.79425 = 0.0196669
#That is, my score minus 0.0196669 should be the Kaggle one. You guess that if one gets score that is less than Kaggles it's the other way around. 
####

####
#Okay, so far we have seen that we might be overfitting our data. It is confirmed also with 10,5 and 3-fold CV, since all of them are smaller than Kaggle's.
#We should turn ourselves into inspecting where the algorithm might overfit. 
#We can't do that with Random Forests since they are only showing us the feature importance, and all the trees in AGGREGATION.
#
#Remember, mtry is a variable which says "try mtry features per tree to split the nodes", which makes some trees use minimum 1 and maximum all features.
#There might be some trees that use all the features (since there are 1000 trees!) "pclass", "title" and "family_size".
#We can train a single tree with all 3 features and inspect what RF is learning after all.
#Don't be overwhelmed if you see that decision tree has better accuracy than our RF algo. That's why RF are used. 
#To train a bunch of trees and average their score, because decision trees are very prone to learning unique characteristics of given training data. 
#So, after all decision trees are easier to understand. 
#Random forests also mitigate variance of each tree trained, and since they are averaging the results of each tree give good estimation of overall accuracy. 
####

####
#To train decesion trees we need rpart, so we are gonna import those libraries.
####

library(rpart)
library(rpart.plot)

####
#This is sort of like 'reverse engineering'. We will see that not only that tree will train on all the features we give it,
#but will also 'drop' ones that don't have too much information, which is better than RF's feature importance.
#Since 'if you repeat same stuff again, you should write a function' holds, we will make one for training trees.
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


####
#Let's grab the 'most promising features' to inspect what got wrong, and what are the potential causes of overfitting. 
####

features <- c("pclass", "title", "family_size")
rpart.train.1 <- data_combined[1:891, features]

# Run CV and check out results
rpart.1.cv.1 <- rpart.cv(94622, rpart.train.1, RF_LABELS, ctrl_3)
rpart.1.cv.1

####
#We obtained the accuracy of 0.8210999 on this model. 
#Compared to random forest model with the same parameters (0.8139169) we are better. But this is expected, since we are training decision tree.

####
#Let's plot the decision tree:
####
prp(rpart.1.cv.1$finalModel, type = 0, extra = 1, under = TRUE)

####
#One can see from the tree where there might be problems. Some of the 'hypotheses' are also valid so things like
#'Have title of Miss, Mr., or Master. not and being in class 3 means that you will survive. 
#But the decision tree intuited that if you are in pclass=3 it depends on your family size weather you survived or not.
#Since family_size = 5,6,8,11 is a bit too specific that might be overfitting, and we should concentrate on that part to improve it.
#Having the title Mr and Other means that you will die around 80% of time. 
#There might be some women in title Other so we will investigate that in order to put everyone in the same basket.
#We know that title of Mr. have better survival rate at first class, so we also might improve that.
#
#So not only that tree's help us seeing where we might overfit, but also where we might improve our model.
#Intuitevely, if we find such subset of features to improve single tree, we are also gonna improve our overall model.
#So, using trees is a bit of a 'improve single, to improve overall'. 
###

################################################################################################
#                             FIXING THE TITLE VARIABLE                                        #
################################################################################################

####
#Parse out last name and title:
#Based on how 'name' variable look we need to find a way to extract the title variable properly.
####
data_combined[1:25, "name"]

name.splits <- str_split(data_combined$name, ",")
name.splits[1]

####
#This might be a nice place to extract last name title, which might be used later. 
####

last.names <- sapply(name.splits, "[", 1)
last.names[1:10]
data.combined$last.name <- last.names

####
#Here we are gonna extract titles. 
####

name.splits <- str_split(sapply(name.splits, "[", 2), " ")
titles <- sapply(name.splits, "[", 2)
unique(titles)

####
#We see that some of the 'new' titles we havent seen might be indicative of nobility, and hence people who are rich.
#Using the experience of Google you can get the idea of why one would aggregate these variables as such:
####

# What's up with a title of 'the'?
data_combined[which(titles == "the"),]

# Re-map titles to be more exact
titles[titles %in% c("Dona.", "the")] <- "Lady."
titles[titles %in% c("Ms.", "Mlle.")] <- "Miss."
titles[titles == "Mme."] <- "Mrs."
titles[titles %in% c("Jonkheer.", "Don.")] <- "Sir."
titles[titles %in% c("Col.", "Capt.", "Major.")] <- "Officer"
table(titles)

# Make title a factor
data_combined$new.title <- as.factor(titles)

# Visualize new version of title
ggplot(data_combined[1:891,], aes(x = new.title, fill = survived)) +
  geom_bar() +
  facet_wrap(~ pclass) + 
  ggtitle("Surival Rates for new.title by pclass")

####
#We see that yeah, these folks wiht title of Dr. etc. are more correlated with those that are in higher classes. 
#We have already decided that Ms, Mile are Miss, and Mme. is Mr. 
#From this plot based on relative survival rate we can put into the same basket other 
#nobility titles. 

# Collapse titles based on visual analysis
indexes <- which(data.combined$new.title == "Lady.")
data_combined$new.title[indexes] <- "Mrs."

indexes <- which(data_combined$new.title == "Dr." | 
                 data_combined$new.title == "Rev." |
                 data_combined$new.title == "Sir." |
                 data_combined$new.title == "Officer")
data_combined$new.title[indexes] <- "Mr."


