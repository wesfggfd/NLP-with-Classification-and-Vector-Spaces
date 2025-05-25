# Assignment 1: Logistic Regression
Welcome to week one of this specialization. You will learn about logistic regression. Concretely, you will be implementing logistic regression for sentiment analysis on tweets. Given a tweet, you will decide if it has a positive sentiment or a negative one. Specifically you will: 

* Learn how to extract features for logistic regression given some text
* Implement logistic regression from scratch
* Apply logistic regression on a natural language processing task
* Test using your logistic regression
* Perform error analysis




### Prepare the Data
* The `twitter_samples` contains subsets of five thousand positive_tweets, five thousand negative_tweets, and the full set of 10,000 tweets.  
    * If you used all three datasets, we would introduce duplicates of the positive tweets and negative tweets.  
    * You will select just the five thousand positive tweets and five thousand negative tweets.



```python
# select the set of positive and negative tweets
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

# split the data into two pieces, one for training and one for testing (validation set) 
test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg 
test_x = test_pos + test_neg

# combine positive and negative labels
train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)

# Print the shape train and test sets
print("train_y.shape = " + str(train_y.shape))
print("test_y.shape = " + str(test_y.shape))
```



* Create the frequency dictionary using the imported build_freqs function.  
    * We highly recommend that you open utils.py and read the build_freqs function to understand what it is doing.
    * To view the file directory, go to the menu and click File->Open.

```Python
    for y,tweet in zip(ys, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1
```
* Notice how the outer for loop goes through each tweet, and the inner for loop steps through each word in a tweet.
* The 'freqs' dictionary is the frequency dictionary that's being built. 
* The key is the tuple (word, label), such as ("happy",1) or ("happy",0).  The value stored for each key is the count of how many times the word "happy" was associated with a positive label, or how many times "happy" was associated with a negative label.



```python
# create frequency dictionary
freqs = build_freqs(train_x, train_y)

# check the output
print("type(freqs) = " + str(type(freqs)))
print("len(freqs) = " + str(len(freqs.keys())))
```

### Process Tweet


```python
# test the function below
print('This is an example of a positive tweet: \n', train_x[0])
print('\nThis is an example of the processed version of the tweet: \n', process_tweet(train_x[0]))
```



The given function 'process_tweet' tokenizes the tweet into individual words, removes stop words and applies stemming.


This is an example of a positive tweet: 

 FollowFriday @France_Inte @PKuchly57 @Milipol_Paris for being top engaged members in my community this week :)



This is an example of the processed version of the tweet: 

 ['followfriday', 'top', 'engag', 'member', 'commun', 'week', ':)']



 ##  - Extracting the Features

* Given a list of tweets, extract the features and store them in a matrix. You will extract two features.
    * The first feature is the number of positive words in a tweet.
    * The second feature is the number of negative words in a tweet. 
* Then train your logistic regression classifier on these features.
* Test the classifier on a validation set.

### Exercise - extract_features
Implement the extract_features function. 
* This function takes in a single tweet.
* Process the tweet using the imported `process_tweet` function and save the list of tweet words.
* Loop through each word in the list of processed words
    * For each word, check the 'freqs' dictionary for the count when that word has a positive '1' label. (Check for the key (word, 1.0)
    * Do the same for the count for when the word is associated with the negative label '0'. (Check for the key (word, 0.0).)
 

```python
def extract_features(tweet, freqs, process_tweet=process_tweet):
    '''
    Input: 
        tweet: a string containing one tweet
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
    Output: 
        x: a feature vector of dimension (1,3)
    '''
    # process_tweet tokenizes, stems, and removes stopwords
    word_l = process_tweet(tweet)
    
    # 3 elements for [bias, positive, negative] counts
    x = np.zeros(3) 
    
    # bias term is set to 1
    x[0] = 1 
    
    ### START CODE HERE ###
    
    # loop through each word in the list of words
    for word in word_l:
        if (word,1) in freqs:
          # increment the word count for the positive label 1
          x[1] += freqs[(word,1)]
        if (word,0) in freqs:
          # increment the word count for the negative label 0
          x[2] += freqs[(word,0)]
        
    ### END CODE HERE ###
    
    x = x[None, :]  # adding batch dimension for further processing
    assert(x.shape == (1, 3))
    return x
```


# Assignment 2: Naive Bayes
Welcome to week two of this specialization. You will learn about Naive Bayes. Concretely, you will be using Naive Bayes for sentiment analysis on tweets. Given a tweet, you will decide if it has a positive sentiment or a negative one. Specifically you will: 

* Train a naive bayes model on a sentiment analysis task
* Test using your model
* Compute ratios of positive words to negative words
* Do some error analysis
* Predict on your own tweet

You may already be familiar with Naive Bayes and its justification in terms of conditional probabilities and independence.
* In this week's lectures and assignments we used the ratio of probabilities between positive and negative sentiment.
* This approach gives us simpler formulas for these 2-way classification tasks.



### 1.1 - Implementing your Helper Functions

To help you train your naive bayes model, you will need to compute a dictionary where the keys are a tuple (word, label) and the values are the corresponding frequency.  Note that the labels we'll use here are 1 for positive and 0 for negative.

You will also implement a lookup helper function that takes in the `freqs` dictionary, a word, and a label (1 or 0) and returns the number of times that word and label tuple appears in the collection of tweets.

For example: given a list of tweets `["i am rather excited", "you are rather happy"]` and the label 1, the function will return a dictionary that contains the following key-value pairs:

{
    ("rather", 1): 2,
    ("happi", 1) : 1, 
    ("excit", 1) : 1
}

- Notice how for each word in the given string, the same label 1 is assigned to each word.
- Notice how the words "i" and "am" are not saved, since it was removed by process_tweet because it is a stopword.
- Notice how the word "rather" appears twice in the list of tweets, and so its count value is 2.



```python
ef count_tweets(result, tweets, ys):
    '''
    Input:
        result: a dictionary that will be used to map each pair to its frequency
        tweets: a list of tweets
        ys: a list corresponding to the sentiment of each tweet (either 0 or 1)
    Output:
        result: a dictionary mapping each pair to its frequency
    '''
    ### START CODE HERE ###
    for y, tweet in zip(ys, tweets):
        for word in process_tweet(tweet):
            # define the key, which is the word and label tuple
            pair = (word,y)
        
            # if the key exists in the dictionary, increment the count
            if pair in result:
                result[pair] += 1

            # else, if the key is new, add it to the dictionary and set the count to 1
            else:
                result[pair] = 1
    ### END CODE HERE ###

    return result
```

```python
# Testing your function

result = {}
tweets = ['i am happy', 'i am tricked', 'i am sad', 'i am tired', 'i am tired']
ys = [1, 0, 0, 0, 0]
count_tweets(result, tweets, ys)

#{('happi', 1): 1, ('trick', 0): 1, ('sad', 0): 1, ('tire', 0): 2}
```

## 2 - Train your Model using Naive Bayes

Naive bayes is an algorithm that could be used for sentiment analysis. It takes a short time to train and also has a short prediction time.

#### So how do you train a Naive Bayes classifier?
- The first part of training a naive bayes classifier is to identify the number of classes that you have.
- You will create a probability for each class.
$P(D_{pos})$ is the probability that the document is positive.
$P(D_{neg})$ is the probability that the document is negative.
Use the formulas as follows and store the values in a dictionary:

![](https://latex.codecogs.com/svg.image?&space;P(D_{pos})=\frac{D_{pos}}{D}\tag{1})

![](https://latex.codecogs.com/svg.image?P(D_{neg})=\frac{D_{neg}}{D}\tag{2})

- Where ![](https://latex.codecogs.com/svg.image?D) is the total number of documents, or tweets in this case, ![](https://latex.codecogs.com/svg.image?D_{pos}) is the total number of positive tweets and ![](https://latex.codecogs.com/svg.image?D_{neg}) is the total number of negative tweets.




#### Prior and Logprior

The prior probability represents the underlying probability in the target population that a tweet is positive versus negative.  In other words, if we had no specific information and blindly picked a tweet out of the population set, what is the probability that it will be positive versus that it will be negative? That is the "prior".

The prior is the ratio of the probabilities ![](https://latex.codecogs.com/svg.image?\frac{P(D_{pos})}{P(D_{neg})}).
We can take the log of the prior to rescale it, and we'll call this the logprior

![](https://latex.codecogs.com/svg.image?\text{logprior}=log\left(\frac{P(D_{pos})}{P(D_{neg})}\right)=log\left(\frac{D_{pos}}{D_{neg}}\right)).

Note that  ![](https://latex.codecogs.com/svg.image?&space;log(\frac{A}{B}))   is the same as   ![](https://latex.codecogs.com/svg.image?&space;log(A)-log(B)).   So the logprior can also be calculated as the difference between two logs:

![](https://latex.codecogs.com/svg.image?\text{logprior}=\log(P(D_{pos}))-\log(P(D_{neg}))=\log(D_{pos})-\log(D_{neg})\tag{3})




#### Positive and Negative Probability of a Word
To compute the positive probability and the negative probability for a specific word in the vocabulary, we'll use the following inputs:

- $freq_{pos}$ and $freq_{neg}$ are the frequencies of that specific word in the positive or negative class. In other words, the positive frequency of a word is the number of times the word is counted with the label of 1.
- $N_{pos}$ and $N_{neg}$ are the total number of positive and negative words for all documents (for all tweets), respectively.
- $V$ is the number of unique words in the entire set of documents, for all classes, whether positive or negative.

We'll use these to compute the positive and negative probability for a specific word using this formula:

![](https://latex.codecogs.com/svg.image?P(W_{pos})=\frac{freq_{pos}&plus;1}{N_{pos}&plus;V}\tag{4})

![](https://latex.codecogs.com/svg.image?P(W_{neg})=\frac{freq_{neg}&plus;1}{N_{neg}&plus;V}\tag{5})

Notice that we add the "+1" in the numerator for additive smoothing.  This [wiki article](https://en.wikipedia.org/wiki/Additive_smoothing) explains more about additive smoothing.


#### Log likelihood
To compute the loglikelihood of that very same word, we can implement the following equations:

![](https://latex.codecogs.com/svg.image?\text{loglikelihood}=\log\left(\frac{P(W_{pos})}{P(W_{neg})}\right)\tag{6})






### train_naive_bayes
Given a freqs dictionary, `train_x` (a list of tweets) and a `train_y` (a list of labels for each tweet), implement a naive bayes classifier.

##### Calculate $V$
- You can then compute the number of unique words that appear in the `freqs` dictionary to get your $V$ (you can use the `set` function).

##### Calculate $freq_{pos}$ and $freq_{neg}$
- Using your `freqs` dictionary, you can compute the positive and negative frequency of each word $freq_{pos}$ and $freq_{neg}$.

##### Calculate $N_{pos}$, and $N_{neg}$
- Using `freqs` dictionary, you can also compute the total number of positive words and total number of negative words $N_{pos}$ and $N_{neg}$.

##### Calculate $D$, $D_{pos}$, $D_{neg}$
- Using the `train_y` input list of labels, calculate the number of documents (tweets) ![](https://latex.codecogs.com/svg.image?D), as well as the number of positive documents (tweets) ![](https://latex.codecogs.com/svg.image?D_{pos}) and number of negative documents (tweets) ![](https://latex.codecogs.com/svg.image?D_{neg}).
- Calculate the probability that a document (tweet) is positive $P(D_{pos})$, and the probability that a document (tweet) is negative $P(D_{neg})$

##### Calculate the logprior
- the logprior is $log(D_{pos}) - log(D_{neg})$

##### Calculate log likelihood
- Finally, you can iterate over each word in the vocabulary, use your `lookup` function to get the positive frequencies, $freq_{pos}$, and the negative frequencies, $freq_{neg}$, for that specific word.
- Compute the positive probability of each word $P(W_{pos})$, negative probability of each word $P(W_{neg})$ using equations 4 & 5.

![](https://latex.codecogs.com/svg.image?P(W_{pos})=\frac{freq_{pos}&plus;1}{N_{pos}&plus;V}\tag{4})
![](https://latex.codecogs.com/svg.image?P(W_{neg})=\frac{freq_{neg}&plus;1}{N_{neg}&plus;V}\tag{5})

**Note:** We'll use a dictionary to store the log likelihoods for each word.  The key is the word, the value is the log likelihood of that word).

- You can then compute the loglikelihood: ![](https://latex.codecogs.com/svg.image?\left(\frac{P(W_{pos})}{P(W_{neg})}\right)).



```python
def train_naive_bayes(freqs, train_x, train_y):
    '''
    Input:
        freqs: dictionary from (word, label) to how often the word appears
        train_x: a list of tweets
        train_y: a list of labels corresponding to the tweets (0,1)
    Output:
        logprior: the log prior. (equation 3 above)
        loglikelihood: the log likelihood of you Naive bayes equation. (equation 6 above)
    '''
    loglikelihood = {}
    logprior = 0

    ### START CODE HERE ###

    # calculate V, the number of unique words in the vocabulary
    vocab = set(word for (word, label),count in freqs.items())
    V = len(vocab)   

    # calculate N_pos, N_neg, V_pos, V_neg
    N_pos = N_neg = 0
    for (word, label), count in freqs.items():
        # if the label is positive (greater than zero)
        if label == 1:

            # Increment the number of positive words by the count for this (word, label) pair
            N_pos += count

        # else, the label is negative
        else:

            # increment the number of negative words by the count for this (word,label) pair
            N_neg += count
    
    # Calculate D, the number of documents
    D = len(train_y)

    # Calculate D_pos, the number of positive documents
    D_pos = sum(1 for y in train_y if y == 1)

    # Calculate D_neg, the number of negative documents
    D_neg = D - D_pos

    # Calculate logprior
    logprior = np.log(D_pos) - np.log(D_neg)
     
    # For each word in the vocabulary...
    for word in vocab:
        # get the positive and negative frequency of the word
        freq_pos = freqs.get((word,1),0) + 1
        freq_neg = freqs.get((word,0),0) + 1

        # calculate the probability that each word is positive, and negative
        p_w_pos = freq_pos / (V + D_pos)
        p_w_neg = freq_neg / (V + D_neg)

        # calculate the log likelihood of the word
        loglikelihood[word] = np.log(p_w_pos) - np.log(p_w_neg)

    ### END CODE HERE ### 

    return logprior, loglikelihood
```



## 3 - Test your Naive Bayes

Now that we have the `logprior` and `loglikelihood`, we can test the naive bayes function by making predicting on some tweets!


### naive_bayes_predict
Implement `naive_bayes_predict`.

**Instructions**:
Implement the `naive_bayes_predict` function to make predictions on tweets.
* The function takes in the `tweet`, `logprior`, `loglikelihood`.
* It returns the probability that the tweet belongs to the positive or negative class.
* For each tweet, sum up loglikelihoods of each word in the tweet.
* Also add the logprior to this sum to get the predicted sentiment of that tweet.

![](https://latex.codecogs.com/svg.image?p=logprior&plus;\sum_i^N(loglikelihood_i))

#### Note
Note we calculate the prior from the training data, and that the training data is evenly split between positive and negative labels (4000 positive and 4000 negative tweets).  This means that the ratio of positive to negative 1, and the logprior is 0.

The value of 0.0 means that when we add the logprior to the log likelihood, we're just adding zero to the log likelihood.  However, please remember to include the logprior, because whenever the data is not perfectly balanced, the logprior will be a non-zero value.



```python
def naive_bayes_predict(tweet, logprior, loglikelihood):
    '''
    Input:
        tweet: a string
        logprior: a number
        loglikelihood: a dictionary of words mapping to numbers
    Output:
        p: the sum of all the logliklihoods of each word in the tweet (if found in the dictionary) + logprior (a number)

    '''
    ### START CODE HERE ###
    # process the tweet to get a list of words
    word_l = process_tweet(tweet)

    # initialize probability to zero
    p = 0

    # add the logprior
    p += logprior

    for word in word_l:

        # check if the word exists in the loglikelihood dictionary
        if word in loglikelihood:
            # add the log likelihood of that word to the probability
            p += loglikelihood[word]

    ### END CODE HERE ###

    return p
```


