
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


moras summary odraditi da vidis gdje je average godina za svaku od mrs i msses pa ako je taj average pomjeren onda jebiga 
znas ko je stariji od zena i zasto predvidjeti da su starje ove ili mladje



