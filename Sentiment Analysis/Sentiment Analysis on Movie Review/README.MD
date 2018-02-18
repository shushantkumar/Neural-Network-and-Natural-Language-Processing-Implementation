# Sentiment Analysis on Movie Reviews

Sentiment Analysis on Movie reviews using dataset from NLTK Corpora<br>

Applying different type of classifiers <br>
* NaiveBayesClassifier<br>
* SklearnClassifier-MultinomialNB<br>
* SklearnClassifier-BernoulliNB<br>
* SklearnClassifier-LogisticRegression<br>
* SklearnClassifier-SGDClassifier<br>
* SklearnClassifier-LinearSVC<br>
* SklearnClassifier-NuSVC
<br>
Then finally creating a vote classifier to vote among all the classifiers and print the majority classification
<br>
Trained classifiers are pickled and stored in pickled_algos/file_name.pickle 
<br>
* classifier_gen.py file executes the above mentioned algo<br>
* sentiment_analysis_movie.py executes the same as classifier_gen.py but its faster and pickles the data<br>
* sentiment_mod.py is like a final file we will use to train and predict other datas<br>

Pickled data present at - https://drive.google.com/open?id=1Lm43pUJrQl4Hn_B1J-r8nrf2KYvjtFXh