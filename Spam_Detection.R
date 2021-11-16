# The following analysis aims to buid a machine learning model to detect spam SMS.
# The data set is available in UCI Machine Learning repository.

# Loading Libraries
library(tm)
library(e1071)

# Reading Data
sms_data = read.csv("C:/Users/sachi/Documents/R_datafiles/SMSSpamCollection", sep="\t", header = TRUE, stringsAsFactors = FALSE)
colnames(sms_data)= c("Class", "Messages")

# Building the Corpus
corpus = Corpus(VectorSource(sms_data$Messages))

# Preparing the data by applying transformations
cleaning_sms_data = function(data)
{
  # convert Lowercase
  data = tm_map(data, tolower)
  # Remove stop-words  words like: pronouns, articles, prepositions etc. which do not add much meaning
  data = tm_map(data, removeWords, stopwords("english"))
  # strip whitespace
  data = tm_map(data, stripWhitespace) 
  # Remove Punctuations
  data = tm_map(data, removePunctuation)
  
}
transformed_data = cleaning_sms_data(corpus)


# Building Document Term matrix
dtm_data = DocumentTermMatrix(transformed_data)


# Find frequency terms
new_data = findFreqTerms(dtm_data,lowfreq =  10)

sparse = removeSparseTerms(dtm_data, 0.99)
sparse
sms_sparse <- as.data.frame(data.matrix(sparse))
# correcting the name of sparse data set
colnames(sms_sparse) = make.names(colnames(sms_sparse))

sms_sparse$class = sms_data$Class

sms_sparse$class=as.factor(sms_sparse$class)


# Splitting data into train and test
set.seed(123)
index = createDataPartition(sms_sparse$class, p=0.8,  list = FALSE)
sms_train = sms_sparse[index,]
sms_test = sms_sparse[-index,]

# Building SVM Model
svm_model <- svm(class~., data=sms_train, scale=FALSE, kernel='linear',type="C" )
predict_train=predict(svm_model, sms_train)
pred_linear <- predict(svm_model, sms_test)
train_accuracy = confusionMatrix(as.factor(predict_train), as.factor(sms_train$class))
test_accuracy <- confusionMatrix(as.factor(pred_linear),as.factor(sms_test$class))
train_accuracy
test_accuracy
