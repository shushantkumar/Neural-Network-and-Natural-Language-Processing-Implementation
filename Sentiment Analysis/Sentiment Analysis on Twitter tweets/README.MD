# Sentiment Analysis on Twitter Tweets

Sentiment Analysis on Twitter Tweets using Twitter API<br>

Applying different type of classifiers <br>
* NaiveBayesClassifier<br>
* SklearnClassifier-MultinomialNB<br>
* SklearnClassifier-BernoulliNB<br>
* SklearnClassifier-LogisticRegression<br>
* SklearnClassifier-SGDClassifier<br>
* SklearnClassifier-LinearSVC<br>
* SklearnClassifier-NuSVC<br>
<br>
Then finally creating a vote classifier to vote among all the classifiers and print the majority classification
<br>
Trained classifiers are pickled and stored in pickled_algos/file_name.pickle 
<br>
* sentiment_mod.py is like a final file we will use to train and predict other datas<br>
* twitter_sentiment.py uses sentiment_mod which itself use pickled data to predict on Twitter Tweets<br>