# STEP 1: rename this file to hw3_sentiment.py

# feel free to include more imports as needed here
# these are the ones that we used for the base model
import numpy as np
import sys
from collections import Counter
import math
from nltk.stem import WordNetLemmatizer 
import spacy

"""
Your name and file comment here: Zaki Kidane
"""


"""
Cite your sources here:
- Source for functions precision, recall, f1 is NB_sckitlearn Notebook Day submission
"""

"""
Implement your functions that are not methods of the Sentiment Analysis class here
"""
def generate_tuples_from_file(training_file_path):
    tuples_list = []
    with open(training_file_path) as lines:
        for line in lines:
            line = line.strip('\n')
            token = tuple(line.split('\t'))
            tuples_list.append(token)
    return tuples_list



def precision(gold_labels, classified_labels):
    true_pos = 0
    false_pos = 0
    false_neg = 0
    for i in range(len(classified_labels)):
        if classified_labels[i] == gold_labels[i]:
                if classified_labels[i] == "1":
                    true_pos+=1
        elif classified_labels[i] == "1":
            false_pos+=1
        else:
            false_neg+=1
    return (true_pos/(true_pos+false_pos))


def recall(gold_labels, classified_labels):
    true_pos = 0
    false_pos = 0
    false_neg = 0
    for i in range(len(classified_labels)):
        if classified_labels[i] == gold_labels[i]:
                if classified_labels[i] == "1":
                    true_pos+=1
        elif classified_labels[i] == "1":
            false_pos+=1
        else:
            false_neg+=1
    return true_pos/(true_pos+false_neg)


def f1(gold_labels, classified_labels):
    prec = precision(gold_labels, classified_labels)
    rec = recall(gold_labels, classified_labels)
    f1 = (2*prec*rec)/(rec+prec)
    return f1


"""
Implement any other non-required functions here
"""
#this generates a report of precision, recall and f1 if the test-data file has a label to compare against
def report(training_file, test_file, model):
    classified_labels = []
    gold_labels = []
    training_tuples = generate_tuples_from_file(training_file)
    test_tuples = generate_tuples_from_file(test_file)
    model.train(training_data)
    for a in test_tuples:
        if len(a) <= 2: #this makes sure the test-data file has a label. If not, it is impossible to generate scores
            return None
        gold_labels.append(a[2])
    for a in test_tuples:
        classified_labels.append(sa.classify(a))
    prec = precision(gold_labels, classified_labels)
    rec = recall(gold_labels, classified_labels)
    f1_result = f1(gold_labels, classified_labels)
    print('precision: ', prec)
    print('recall: ', rec)
    print('f1: ', f1_result)
    

def generateLabelTestData(training_file, test_file): 
    id_class_tuples = []
    training_tuples = generate_tuples_from_file(training_file)
    test_tuples = generate_tuples_from_file(test_file)
    sa = SentimentAnalysis()
    sa.train(training_data)
    for a in test_tuples: 
        id_class_tuples.append((a[0], sa.classify(a)))
    output_file = "label_test_data.txt"
    output_f = open(output_file, "w+")
    for line in id_class_tuples: 
        string = str(line[0])+ ' ' + str(line[1])
        output_f.write(string+'\n')
    output_f.close()
    return None

def generateImprovedLabelTestData(training_file, test_file): 
    id_class_tuples = []
    training_tuples = generate_tuples_from_file(training_file)
    test_tuples = generate_tuples_from_file(test_file)
    improved = SentimentAnalysisImproved()
    improved.train(training_data)
    for a in test_tuples: 
        id_class_tuples.append((a[0], sa.classify(a)))
    output_file = "improved_label_test_data.txt"
    output_f = open(output_file, "w+")
    for line in id_class_tuples: 
        string = str(line[0])+ ' ' + str(line[1])
        output_f.write(string+'\n')
    output_f.close()
    
    




"""
implement your SentimentAnalysis class here
"""
class SentimentAnalysis:


    def __init__(self):
        # do whatever you need to do to set up your class here
        self.features = [] #list of features (w, c)
        self.positive_count = {} # {word: # of + labels for word) i.e. count(w_i, 1)
        self.negative_count = {} # {word: # of - labels for word) i.e. count(w_i, 0)
        self.positive_prob = {} #p(w_i|c=1)
        self.negative_prob = {} #p(w_i|c=0)
        self.pos_n = 0 #count(c=1)
        self.neg_n = 0 #count(c=0) 
        self.prob_pos = 1 #p(c=1)
        self.prob_neg = 1 #p(c=0)

    def train(self, examples):
    #input here is list of tuples generated by generate_tuples_from_file
        for e in examples:

            feat = self.featurize(e)
            if e[2] == '1':
                for f in feat: 
                    self.positive_count[f[0]] = self.positive_count.get(f[0], 0) +1
                    self.negative_count[f[0]] = self.negative_count.get(f[0], 0)
                    self.pos_n += 1
            else:
                for f in feat:
                    self.negative_count[f[0]] = self.negative_count.get(f[0], 0) +1
                    self.positive_count[f[0]] = self.positive_count.get(f[0], 0)
                    self.neg_n +=1

        for k,v in self.positive_count.items():
            self.positive_prob[k] = (v+1)/(self.pos_n + len(self.positive_count))
        for k,v in self.negative_count.items():
            self.negative_prob[k] = (v+1)/(self.neg_n + len(self.negative_count))


        self.prob_pos = self.pos_n/(self.pos_n+self.neg_n)
        self.prob_neg = self.neg_n/(self.pos_n+self.neg_n)
        return None

    def score(self, data):
        data_p = (data[0], data[1], 1)
        data_n = (data[0], data[1], 0)
        feat_p = self.featurize(data_p)
        feat_n = self.featurize(data_n)
        prob_positive = np.log(self.prob_pos)#this is ln(prob_pos) and later, e^prob will be applied
        prob_negative = np.log(self.prob_neg)
        for f in feat_p:
            if f[0] in self.positive_prob: #this makes sure we ignore unseen words
                prob_positive = prob_positive + np.log(self.positive_prob[f[0]])
        for f in feat_n:
            if f[0] in self.positive_prob:#this makes sure we ignore unseen words
                prob_negative = prob_negative + np.log(self.negative_prob[f[0]])
        prob_positive = np.exp(prob_positive)
        prob_negative = np.exp(prob_negative)
        return {"0": prob_negative, "1": prob_positive}

    def classify(self, data):
        score_dict = self.score(data)
        return max(score_dict, key=score_dict.get)

    def featurize(self, data):
        list_features = []
        sentence = data[1]
        sentence_list = sentence.split()
        label = data[2]
        for s in sentence_list:
            list_features.append((s, label))


        return list_features
    
    def __str__(self):
        return "Naive Bayes - bag-of-words baseline"

    
class SentimentAnalysisImproved: 
    def __init__(self):
        # do whatever you need to do to set up your class here
        self.features = [] #list of features (w, c)
        self.positive_count = {} # {word: # of + labels for word) i.e. count(w_i, 1)
        self.negative_count = {} # {word: # of - labels for word) i.e. count(w_i, 0)
        self.positive_prob = {} #p(w_i|c=1)
        self.negative_prob = {} #p(w_i|c=0)
        self.pos_n = 0 #count(c=1) 
        self.neg_n = 0 #count(c=0)
        self.prob_pos = 1 #p(c=1)
        self.prob_neg = 1 #p(c=0)
        self.nlp =spacy.load('en_core_web_sm')

    def train(self, examples):
    #input here is list of tuples generated by generate_tuples_from_file
        for e in examples:

            feat = self.featurize(e)
            if e[2] == '1':
                self.pos_n += 1
                for f in feat:
                    self.positive_count[f[0]] = self.positive_count.get(f[0], 0) +1
                    self.negative_count[f[0]] = self.negative_count.get(f[0], 0)
                    
            else:
                self.neg_n +=1
                for f in feat:
                    self.negative_count[f[0]] = self.negative_count.get(f[0], 0) +1
                    self.positive_count[f[0]] = self.positive_count.get(f[0], 0)
                    

        for k,v in self.positive_count.items():
            self.positive_prob[k] = (v+1)/(self.pos_n + len(self.positive_count))
           
        for k,v in self.negative_count.items():
            self.negative_prob[k] = (v+1)/(self.neg_n + len(self.negative_count))
       


        self.prob_pos = self.pos_n/(self.pos_n+self.neg_n)
        self.prob_neg = self.neg_n/(self.pos_n+self.neg_n)
        return None

    def score(self, data):
        data_p = (data[0], data[1], 1)
        data_n = (data[0], data[1], 0)
        feat_p = self.featurize(data_p)
        feat_n = self.featurize(data_n)
        prob_positive = np.log(self.prob_pos)#this is ln(prob_pos)
        prob_negative = np.log(self.prob_neg)
        for f in feat_p:
            if f[0] in self.positive_prob: #this makes sure we ignore unseen words
                prob_positive = prob_positive + np.log(self.positive_prob[f[0]])
        for f in feat_n:
            if f[0] in self.positive_prob:#this makes sure we ignore unseen words
                prob_negative = prob_negative + np.log(self.negative_prob[f[0]])
        prob_positive = np.exp(prob_positive)
        prob_negative = np.exp(prob_negative)
        return {"0": prob_negative, "1": prob_positive}

    def classify(self, data):
        score_dict = self.score(data)
        return max(score_dict, key=score_dict.get)
    #only this function has been modified in the improved model class, with lemmatizing and lowercasing, both 
    #don't have any effect on the outcome
    def featurize(self, data):
        list_features = []
        sentence = data[1].lower()
        label = data[2]
#         sentence_list = sentence.split()
        #below nltk lemmatizer was commented out for being unsuccessful in improving accuracy
#         lemmatizer = WordNetLemmatizer()
#         lemmatized_list = [lemmatizer.lemmatize(w) for w in sentence_list]
        
#         for s in lemmatized_list:
#             list_features.append((s, label))

        
        doc = self.nlp(sentence)
        #remove stop_words like I or is which are common across all documents
        token_list = [token for token in doc if not token.is_stop]
        #lemmatize
        lemmas = [token.lemma_ for token in token_list]
        for s in lemmas: 
            list_features.append((s, label))
        
        return list_features
    
    def __str__(self):
        return "Naive Bayes - with additional features: lowercase, spacy stop-words and lemmatization"
    



if __name__ == "__main__":
    print("in main")
    if len(sys.argv) != 3:
        print("Usage:", "python hw3_sentiment.py training-file.txt testing-file.txt")
        sys.exit(1)

    training = sys.argv[1]
    testing = sys.argv[2]

    sa = SentimentAnalysis()
    training_data = generate_tuples_from_file(training)
    test_data = generate_tuples_from_file(testing)
    sa.train(training_data)
    print(sa.classify(test_data[2]))

    print(sa)
    
    # the following function generates a file label_test_data.txt with space-separated id and classification
    generateLabelTestData(training, testing)
    
    #the following prints the report for the accuracy scores of the model against the given test file
    report(training, testing, sa)


    improved = SentimentAnalysisImproved()
    improved.train(training_data)
    print(improved.classify(test_data[2]))
    
    #generate file with improved_label_test_data.txt
    generateImprovedLabelTestData(training, testing)
    
    #the following generates report for imprroved sentiment analysis model
    report(training, testing, improved)
    print(improved)
