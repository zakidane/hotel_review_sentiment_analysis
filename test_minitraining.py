import unittest
import hw3_sentiment as hw3

# updated 3/5/2020 to fix ordering issues in tests


class TestSentimentAnalysisBaselineMiniTrain(unittest.TestCase):
    
    def setUp(self):
        #Sets the Training File Path
        # Feel free to edit to reflect where they are on your machine
        self.trainingFilePath="minitrain.txt"
        self.devFilePath="minidev.txt"


    def test_GenerateTuplesFromTrainingFile(self):
        #Tests the tuple generation from the sentences
        sa = hw3.SentimentAnalysis()
        examples = hw3.generate_tuples_from_file(self.trainingFilePath)
        actualExamples = [('ID-2', 'The hotel was not liked by me', '0'), ('ID-3', 'I loved the hotel', '1'), ('ID-1', 'The hotel was great', '1'), ('ID-4', 'I hated the hotel', '0')]
        self.assertListEqual(sorted(actualExamples), sorted(examples))

       
    def test_ScorePositiveExample(self):
        #Tests the Probability Distribution of each class for a positive example
        sa = hw3.SentimentAnalysis()
        examples = hw3.generate_tuples_from_file(self.trainingFilePath)
        #Trains the Naive Bayes Classifier based on the tuples from the training data
        sa.train(examples)
        #Returns a probability distribution of each class for the given test sentence
        sentence = ("ID11:", "I loved the hotel")
        score=sa.score(sentence)
        #P(C|text)=P(I|C)*P(loved|C)*P(the|C)*P(hotel|C),where C is either 0 or 1(Classifier)
        pos = ((1+1)/(8+12))*((1+1)/(8+12))*((1+1)/(8+12))*((2+1)/(8+12))*(2/4)
        neg = ((1+1)/(11+12))*((0+1)/(11+12))*((1+1)/(11+12))*((2+1)/(11+12))*(2/4)
        actualScoreDistribution={'1': pos, '0': neg}
        self.assertAlmostEqual(actualScoreDistribution['0'], score['0'], places=5)
        self.assertAlmostEqual(actualScoreDistribution['1'], score['1'], places=5)
    
  
    def test_ScorePositiveExampleRepeats(self):
        #Tests the Probability Distribution of each class for a positive example
        sa = hw3.SentimentAnalysis()
        examples = hw3.generate_tuples_from_file(self.trainingFilePath)
        #Trains the Naive Bayes Classifier based on the tuples from the training data
        sa.train(examples)
        #Returns a probability distribution of each class for the given test sentence
        sentence = ("ID12", "I loved the hotel loved the hotel")
        score=sa.score(sentence)
        #P(C|text)=P(I|C)*P(loved|C)*P(the|C)*P(hotel|C),where C is either 0 or 1(Classifier)
        pos = ((1+1)/(8+12))*((1+1)/(8+12))*((1+1)/(8+12))*((2+1)/(8+12))*((1+1)/(8+12))*((1+1)/(8+12))*((2+1)/(8+12))*(2/4)
        neg = ((1+1)/(11+12))*((0+1)/(11+12))*((1+1)/(11+12))*((2+1)/(11+12))*((0+1)/(11+12))*((1+1)/(11+12))*((2+1)/(11+12))*(2/4)
        actualScoreDistribution={'1': pos, '0': neg}
        self.assertAlmostEqual(actualScoreDistribution['0'], score['0'], places=5)
        self.assertAlmostEqual(actualScoreDistribution['1'], score['1'], places=5)

    def test_ScorePositiveExampleWithUnkowns(self):
        #Tests the Probability Distribution of each class for a positive example
        sa = hw3.SentimentAnalysis()
        examples = hw3.generate_tuples_from_file(self.trainingFilePath)
        #Trains the Naive Bayes Classifier based on the tuples from the training data
        sa.train(examples)
        #Returns a probability distribution of each class for the given test sentence
        sentence = ("ID13", "I loved the hotel a lot")
        score=sa.score(sentence)
        #P(C|text)=P(I|C)*P(loved|C)*P(the|C)*P(hotel|C)*P(a|C)*P(lot|C)*P(C),where C is either 0 or 1(Classifier)
        pos = ((1+1)/(8+12))*((1+1)/(8+12))*((1+1)/(8+12))*((2+1)/(8+12))*(2/4)
        neg = ((1+1)/(11+12))*((0+1)/(11+12))*((1+1)/(11+12))*((2+1)/(11+12))*(2/4)
        actualScoreDistribution={'1': pos, '0': neg}
        self.assertAlmostEqual(actualScoreDistribution['0'], score['0'], places=5)
        self.assertAlmostEqual(actualScoreDistribution['1'], score['1'], places=5)
        

    def test_ClassifyForPositiveExample(self):
        #Tests the label classified  for the positive test sentence
        sa = hw3.SentimentAnalysis()
        examples = hw3.generate_tuples_from_file(self.trainingFilePath)
        sa.train(examples)
        #Classifies the test sentence based on the probability distribution of each class
        sentence = ("ID14","I loved the hotel a lot")
        label=sa.classify(sentence)
        actualLabel='1'
        self.assertEqual(actualLabel,label)
        


    def test_ScoreForNegativeExample(self):
        #Tests the Probability Distribution of each class for a negative example
        sa = hw3.SentimentAnalysis()
        examples = hw3.generate_tuples_from_file(self.trainingFilePath)
        sa.train(examples)
        sentence = ("ID15", "I hated the hotel")
        score=sa.score(sentence)
         #P(C|text)=P(I|C)*P(hated|C)*P(the|C)*P(hotel|C)*P(C),where C is either 0 or 1(Classifier)
        pos = ((1+1)/(8+12))*((0+1)/(8+12))*((1+1)/(8+12))*((2+1)/(8+12))*(2/4)
        neg = ((1+1)/(11+12))*((1+1)/(11+12))*((1+1)/(11+12))*((2+1)/(11+12))*(2/4)
        actualScoreDistribution={'1': pos, '0': neg}
        self.assertAlmostEqual(actualScoreDistribution['0'], score['0'], places=5)
        self.assertAlmostEqual(actualScoreDistribution['1'], score['1'], places=5)
        

    def test_ClassifyForNegativeExample(self):
        #Tests the label classified  for the negative test sentence
        sa = hw3.SentimentAnalysis()
        examples = hw3.generate_tuples_from_file(self.trainingFilePath)
        sa.train(examples)
        sentence = ("ID16", "I hated the hotel")
        label=sa.classify(sentence)
        actualLabel='0'
        self.assertEqual(actualLabel,label)    


    def test_precision(self):
        gold = [1, 1, 1, 0, 0]
        gold = [str(b) for b in gold]
        classified = [1, 0, 0, 0, 1]
        classified = [str(b) for b in classified]
        self.assertEqual((1 / 2), hw3.precision(gold, classified))


    def test_recall(self):
        gold = [1, 1, 1, 0, 0]
        gold = [str(b) for b in gold]
        classified = [1, 0, 0, 0, 1]
        classified = [str(b) for b in classified]
        self.assertEqual((1 / 3), hw3.recall(gold, classified))

    def test_f1(self):
        gold = [1, 1, 1, 0, 0]
        gold = [str(b) for b in gold]
        classified = [1, 0, 0, 0, 1]
        classified = [str(b) for b in classified]
        p = 1 / 2
        r = 1 / 3
        self.assertEqual((2 * p * r) / (p + r), hw3.f1(gold, classified))
        

if __name__ == "__main__":
    print("Usage: python test_minitraining.py")
    unittest.main()

