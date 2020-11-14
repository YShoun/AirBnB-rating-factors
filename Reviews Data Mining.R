library(dplyr)
library(tidytext)
library(caret)
library(e1071)
library(pROC)
library(textstem)
library(caTools)
library(quanteda)
library(tm)

# the data is too big we need to take a sample
data1 = read.csv("C:\\Users\\Y.Shoun\\Documents\\Cours\\M2\\S2\\CA683 - Data Analytics and Data Mining\\Continuous_Assesment\\data\\reviewsToR.csv", sep= ",", header = TRUE)
data = data1

# take neutral out
positive <- data[data$sentiment == "positive", ]
negative <- data[data$sentiment == "negative", ]

data <- rbind(positive, negative)
data$sentiment <- factor(data$sentiment)  
str(data)
data <- subset(data, select = -c(polarity))
data <- data[!apply(data == "", 1, all),]

#shuffle the new dataset
set.seed(42)
rows <- sample(nrow(data))
rows <- data[rows, ]
set.seed(123)
rows <- sample(nrow(data))
data <- data[rows, ]


# taking a sample
set.seed(100)
n01 = floor(nrow(data)*0.04) # 0.001
sample = sample.int(nrow(data),n01)
data = data[sample,]

str(data)
data <- subset(data, select = -c(listing_id, id, date, reviewer_id, comments))
data$comments_clean <- as.character (data$comments_clean)
data$sentiment <- as.factor(data$sentiment)


##### TF-IDF on NB #####

# Creating corpus = read friendly in R for text-mining
corpus<-Corpus(VectorSource(data$comments_clean))
corpus.clean<-corpus
corpus.clean

# Getting DTM based on TF-IDF
dtm<- DocumentTermMatrix(corpus.clean, control = list(weighting=weightTfIdf))
dtm # because of 100% sparsity, we retain all the words.

# Splitting train/test at 0,7
df.train<-data[1:1457,]
df.test<-data[1458:2082,]

length(df.test$sentiment)

dtm.train<-dtm[1:1457,]
dtm.test<-dtm[1458:2082,]

corpus.clean.train <- corpus.clean[1:1457]
corpus.clean.test <- corpus.clean[1458:2082]


freq <- findFreqTerms(dtm.train, 5) # only keep word that has a freq of a t least 5
length((freq))

dtm.train.nb <- DocumentTermMatrix(corpus.clean.train, control=list(dictionary = freq))
dim(dtm.train.nb)

dtm.test.nb <- DocumentTermMatrix(corpus.clean.test, control=list(dictionary = freq))
dim(dtm.test.nb)

# Function to convert the word frequencies to yes (presence) and no (absence) labels
convert_count <- function(x) {
  y <- ifelse(x < 0, 1,0)
  y <- factor(y, levels=c(0,1), labels=c("negative_value", "positive_value"))
  y
}

# Apply the convert_count function to get final training and testing DTMs
trainNB <- apply(dtm.train.nb, 2, convert_count)
testNB <- apply(dtm.test.nb, 2, convert_count)


# Train the classifier
# https://cran.r-project.org/web/packages/naivebayes/naivebayes.pdf
# according to the doc, naive_bayes is used to fit Naive Bayes model in which predictors are assumed to be independent within each class label.
classifier <- naiveBayes(trainNB, df.train$sentiment) # X_train, Y_train
pred <- predict(classifier, newdata=testNB)

mean(pred != df.test$sentiment)

library(caret)
conf.mat <- confusionMatrix(pred, df.test$sentiment)
conf.mat

# ROC
roc = roc(pred, as.ordered(df.test$sentiment))
plot(roc, print.auc = TRUE,print.thres = "best")


##### TF-IDF on SVM #####

# GETTING THE CORPUS 
corpus.svm <- VCorpus(VectorSource(data$comments_clean))

# GETTING DTM 

dtm.svm <- DocumentTermMatrix(corpus.svm, control = list(weighting = weightTfIdf)) # create tdm from n-grams
dtm.svm

# REDUCTION OF DTM

sparse <- removeSparseTerms(dtm.svm, 0.999) # because of the high number of words

tweetsSparse <- as.data.frame(as.matrix(sparse))
colnames(tweetsSparse) <- make.names(colnames(tweetsSparse))
tweetsSparse$class <- data$sentiment

# Build a training and testing set.
set.seed(123)
split <- sample.split(tweetsSparse$class, SplitRatio=0.7)
train_data <- subset(tweetsSparse, split==TRUE)
test_data <- subset(tweetsSparse, split==FALSE)

train_data$class<-as.factor(train_data$class)

# tune for the best cost
model.tune <- tune.svm(class ~ ., data=train_data, kernel="linear", cost = 10^(-3:2))

plot(10^(-3:2),model.tune$performance$error,log="x")
lines(10^(-3:2),model.tune$performance$error)

# train model
model.svm <- svm(class~.,data=train_data, kernel='linear',cost=1)
model.svm

#Predict Output
preds.svm <- predict(model.svm,test_data)

# test set error rate
mean(preds.svm != test_data$class)

conf.svm <- confusionMatrix(table(test_data$class, preds.svm))
conf.svm

roc.svm <- roc(preds.svm, as.ordered(test_data$class))
plot(roc.svm, print.auc = TRUE,print.thres = "best")


##### Reoworking the dataset #####
data.reworked = data1


str(data.reworked)
data.reworked <- subset(data.reworked, select = -c(listing_id, id, date, reviewer_id, comments))
data.reworked$comments_clean <- as.character (data.reworked$comments_clean)
data.reworked$sentiment <- as.factor(data.reworked$sentiment)
str(data.reworked)

# because we have way more positive than negative review we need to balance the data
negative <- data.reworked[data.reworked$sentiment == "negative", ]
negative <- negative[negative$polarity < -0.7, ] 
length(negative$sentiment)
negative$sentiment <- factor(negative$sentiment)  

# # take a sample of 421 neutral comments
# neutral <- data.reworked[data.reworked$sentiment == "neutral", ]
# length(neutral$sentiment)
# 
# set.seed(100)
# n.neutral = floor(nrow(neutral)*0.0017)
# sample = sample.int(nrow(neutral),n.neutral)
# neutral = neutral[sample,]

# take a sample of 421 positive comments
positive <- data.reworked[data.reworked$sentiment == "positive", ]
positive <- positive[positive$polarity < 0.7, ]
length(positive$sentiment)

set.seed(100)
n.positive = floor(nrow(positive)*0.004)
sample = sample.int(nrow(positive),n.positive)
positive = positive[sample,]
length(positive$sentiment)
positive$sentiment <- factor(positive$sentiment)  

# merging the 3 tables
# data <- rbind(positive, neutral, negative)
data <- rbind(positive, negative)
data$sentiment <- factor(data$sentiment)  
str(data)
data <- subset(data, select = -c(polarity))
data <- data[!apply(data == "", 1, all),]

#shuffle the new dataset
set.seed(42)
rows <- sample(nrow(data))
rows <- data[rows, ]
set.seed(123)
rows <- sample(nrow(data))
data <- data[rows, ]


##### TF-IDF on NB reworked #####

# Creating corpus = read friendly in R for text-mining
corpus<-Corpus(VectorSource(data$comments_clean))
corpus.clean<-corpus
corpus.clean

# Getting DTM based on TF-IDF
dtm<- DocumentTermMatrix(corpus.clean, control = list(weighting=weightTfIdf))
dtm
rowTotals <- slam::row_sums(dtm)
dtm <- dtm[rowTotals > 0, ]
dtm # because of 100% sparsity, we retain all the words.

# Splitting train/test at 0,7
df.train<-data[1:184,]
df.test<-data[185:263,]

length(df.test$sentiment)

dtm.train<-dtm[1:184,]
dtm.test<-dtm[185:263,]

corpus.clean.train <- corpus.clean[1:184]
corpus.clean.test <- corpus.clean[185:263]

freq <- findFreqTerms(dtm.train, 5) # only keep word that has a freq of a t least 5
length((freq))

dtm.train.nb <- DocumentTermMatrix(corpus.clean.train, control=list(dictionary = freq))
dim(dtm.train.nb)

dtm.test.nb <- DocumentTermMatrix(corpus.clean.test, control=list(dictionary = freq))
dim(dtm.test.nb)

# Function to convert the word frequencies to yes (presence) and no (absence) labels
convert_count <- function(x) {
  y <- ifelse(x < 0, 1,0)
  y <- factor(y, levels=c(0,1), labels=c("negative_value", "positive_value"))
  y
}

# Apply the convert_count function to get final training and testing DTMs
trainNB <- apply(dtm.train.nb, 2, convert_count)
testNB <- apply(dtm.test.nb, 2, convert_count)


# Train the classifier
# https://cran.r-project.org/web/packages/naivebayes/naivebayes.pdf
# according to the doc, naive_bayes is used to fit Naive Bayes model in which predictors are assumed to be independent within each class label.
classifier <- naiveBayes(trainNB, df.train$sentiment) # X_train, Y_train
pred <- predict(classifier, newdata=testNB)

mean(pred != df.test$sentiment)

library(caret)
conf.mat <- confusionMatrix(pred, df.test$sentiment)
conf.mat

# ROC
roc = roc(pred, as.ordered(df.test$sentiment))
plot(roc, print.auc = TRUE,print.thres = "best")

##### TF-IDF on SVM reworked #####

# GETTING THE CORPUS 
corpus.svm <- VCorpus(VectorSource(data$comments_clean))

# GETTING DTM 

dtm.svm <- DocumentTermMatrix(corpus.svm, control = list(weighting = weightTfIdf)) # create tdm from n-grams
dtm.svm

# REDUCTION OF DTM

sparse <- removeSparseTerms(dtm.svm, 0.999) # because of the high number of words

tweetsSparse <- as.data.frame(as.matrix(sparse))
colnames(tweetsSparse) <- make.names(colnames(tweetsSparse))
tweetsSparse$class <- data$sentiment

# Build a training and testing set.
set.seed(123)
split <- sample.split(tweetsSparse$class, SplitRatio=0.7)
train_data <- subset(tweetsSparse, split==TRUE)
test_data <- subset(tweetsSparse, split==FALSE)

train_data$class<-as.factor(train_data$class)

# tune for the best cost
model.tune <- tune.svm(class ~ ., data=train_data, kernel="linear", cost = 10^(-3:2))

plot(10^(-3:2),model.tune$performance$error,log="x")
lines(10^(-3:2),model.tune$performance$error)

# train model
model.svm <- svm(class~.,data=train_data, kernel='linear',cost=1)
model.svm

#Predict Output
preds.svm <- predict(model.svm,test_data)

# test set error rate
mean(preds.svm != test_data$class)

conf.svm <- confusionMatrix(table(test_data$class, preds.svm))
conf.svm

roc.svm <- roc(preds.svm, as.ordered(test_data$class))
plot(roc.svm, print.auc = TRUE,print.thres = "best")

