# RETWEET: Dataset of Tweets and overall predominant sentiment of their Replies. 

*******************************************************
* RETWEET: Sentiment Analysis of Twitter              *
*           Reply-Threads given the Source            *
*                                                     *
*                                                     *
*             GOLD STANDARD TEST DATASET              *
*             TRAINING RAW DATASET                    *
*             TRAINING AUTOMATICALLY-LABELED DATASET  *
*                                                     *
*                                                     *
* https://kaggle.com/soroosharasteh/retweet/          *
* soroosh.arasteh@fau.de                              *
*                                                     *
*******************************************************

Version 1.0: October 10, 2020


In case you use this dataset, please cite the original paper:

S. Tayebi Arasteh, M. Monajem, V. Christlein, P. Heinrich, A. Nicolaou, H.N. Boldaji, M. Lotfinia, S. Evert. "How Will Your Tweet Be Received? Predicting the Sentiment Polarity of Tweet Replies". Proceedings of the 2021 IEEE 15th International Conference on Semantic Computing (ICSC), Laguna Hills, CA, USA, January 2021.






BIBTEX

	@inproceedings{RETWEET,
	  title = "How Will Your Tweet Be Received? Predicting the Sentiment Polarity of Tweet Replies",
	  author = "Tayebi Arasteh, Soroosh and Monajem, Mehrpad and Christlein, Vincent and
	  Heinrich, Philipp and Nicolaou, Anguelos and Naderi Boldaji, Hamidreza and Lotfinia, Mahshad and Evert, Stefan",
	  booktitle = "Proceedings of the 2021 IEEE 15th International Conference on Semantic Computing (ICSC)",
 	  address = "Laguna Hills, CA, USA",
  	  pages = "370-373",
  	  doi = "10.1109/ICSC50631.2021.00068",
	  url = "https://ieeexplore.ieee.org/document/9364527/",
	  month = "01",       
          year = "2021"
	 }



LIST OF VERSIONS

  v1.0 [10.10.2020]: initial distribution of the data

=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
SUMMARY

WHAT: Message-level Polarity Classification.

GOAL: To predict the predominant sentiment among (potential) first-order replies to a given tweet.

IDEA: Mitigate the problem of lacking labeled training data with treating the unsupervised nature of the problem as a supervised learning case.

APPROACH:

1. Train a tweet classifier.
2. Automatically label the replies using the classifier trained in the first part.
3. Choose a final label representing the general predominant sentiment of the replies of every tweet.


=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

DATA COLLECTION

Unfortunately, Twitter does not permit to directly query for all of the replies to a given tweet. Consequently, to download all of the replies to a tweet, the Search API should be used. However the Search API is limited to 75000 request per hour, which causes the mining and downloading process to be slow.
 
Furthermore, using the Twitter API, there is no possibility of downloading absolute random data. Therefore, we try to make the procedure as random as possible by utilizing two different strategies for data downloading and using them in an intermixed manner.

1. Our first strategy is based on a sample of English tweets obtained by filtering the Twitter stream
 via a list of cultural keywords. This list consists of 147 words that are deemed to play a "pivotal 
role in discussions of culture and society", covering diverse words such as "aesthetics", "environment", 
"feminism", "power", "tourism", or "youth". We extracted all tweets in 2019 that have a minimum of 
20 first-order replies in the dataset. The data come with an obvious caveat: Both the source tweet 
as well as all the replies must contain at least one word from the list of keywords. Therewith, 
it is highly unlikely that the list of replies for any given source is exhaustive, i.e. there 
might be many more first-order replies to the source tweet that are not in the dataset.

2.  As our second approach, we use the GetOldTweets3 library to download all the replies corresponding 
to every tweet. We define few restrictions to add randomization to the process. Firstly, every tweet 
and also every reply should contain at least 20 strings. This is due to the fact that our automatic 
tweet classifier, explsined in the paper, is optimized based on the message-level classification paradigm. 
Therefore, it operates optimal when the input contains at least a sufficient number of words. The second 
constraint is that every tweet should contain at least 20 first-order replies. In order to increase randomness, 
in this strategy, instead of referencing to a list of keywords, we manually choose some keywords, which are 
most likely to include long discussions, such as "Coronavirus" and "football" or the ones, which are most 
likely to include strong opinions such as "birthday", "war", or "racism" in order to account for the easy-to-guess examples.


=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

MANUAL ANNOTATIONS FOR THE RETWEET (TEST GOLD DATASET)

5,015 tweets with their corresponding replies, collected as a combination of the two different collection strategies, 
were given to three different students. Each of them had to read all the replies corresponding to every tweet, 
without observing the original tweet in order to avoid having a prior knowledge, and decide on ONE final 
sentiment for the replies. The assigned sentiment can only be one of the positive, negative, or neutral labels.

Considering the fact that this is a really challenging task for the machine, to prevent human mistakes, we correlated 
the results of the three annotators and only chose the tweets, in which all of the annotators had the same opinion 
on the labels, as the final gold standard test data. Therefore, we finally, ended up with a test set consisting of 
1,519 human labeled tweets, with the labels being the sentiment of the replies of a tweet and not the tweet itself. 

=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

DATASET CONTENTS

Training raw dataset: 34,953 unique tweets in total and individual automatic labels for all of their corresponding replies (1,519,504 total replies). Including,
./RETWEET_data/train_reply_labels_set1.txt
./RETWEET_data/train_reply_labels_set2.txt

Training autamtically-labeled dataset: 34,953 unique tweets and ONE final automatic label (chosen based on the algorithm 1 of our paper) for every tweet. Including,
./RETWEET_data/train_final_label.txt

Gold standard test dataset: 1,519 unique tweets with their manual labels for replies. ONE final label, which states the predominant overall polarity of all its replies, is assigned to every tweet. Including,
./RETWEET_data/test_gold.txt


DATA FORMAT FOR ALL THE FILES

	label<TAB>id

where, "label" can be positive, neutral or negative, corresponding to the overall message-level 
polarity of the replies of the tweet and "id" corresponds to the Twitter unique ID for the tweets.


=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

NOTES

1. Please note that by downloading the Twitter data you agree to abide by the Twitter terms of service (https://twitter.com/tos), 
and in particular you agree not to redistribute the data and to delete tweets that are marked deleted in the future.

2. The "neutral" label in the annotations stands for objective or neutral.

3. The distribution consists of a set of Twitter unique tweet IDs with annotations (overall polarity of replies).
As for data privacy, the texts of the tweets and replies are not distributed. But as all the utilized resources in 
this dataset are taken from public tweets, having the tweet unique IDs, you can download the tweet and its replies.
You can use the Semeval Twitter data downloading script to obtain the corresponding tweets: 
	
	https://github.com/seirasto/twitter_download/

4. The dataset URL:

	https://kaggle.com/soroosharasteh/retweet/ 


=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

LICENSE

The accompanying dataset is released under a Creative Commons Attribution 4.0 International License (https://creativecommons.org/licenses/by/4.0/).


CONTACT:

E-mail: soroosh.arasteh@fau.de


SOURCE CODE:

The official source code of the paper: https://github.com/tayebiarasteh/retweet/


REFERENCES:

Sara Rosenthal, Noura Farra, and Preslav Nakov. SemEval-2017 Task 4: Sentiment Analysis in Twitter. In Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017), pp.502-518, August 2017, Vancouver, Canada.

Tony Bennett, Lawrence Grossberg, and Meaghan Morris. 2005. New Keywords: A Revised Vocabulary of Cultur eand Society. Blackwell, Malden.

Dmitry Mottl and Jefferson Henriquel. Getoldtweets3.

