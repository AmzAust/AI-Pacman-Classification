# mira.py
# -------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# Mira implementation
import util
PRINT = True

class MiraClassifier:
    """
    Mira classifier.
    
    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False
        self.C = 0.001
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."

        self.features = trainingData[0].keys() # this could be useful for your code later...

        if (self.automaticTuning):
            Cgrid = [0.002, 0.004, 0.008]
        else:
            Cgrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
        """
        This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid,
        then store the weights that give the best accuracy on the validationData.
        
        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        representing a vector of values.
        """
        myCounter = util.Counter()
        gridWeights = util.Counter()
        label = util.Counter()
        mySum = 0    
       
        for c in Cgrid: #Loop through C grid
            for a in range(self.max_iterations): #Return max of iterations
                for b in range(len(trainingData)): #Return training data length
                    for d in self.legalLabels:
                        label[d] = trainingData[b].__mul__(self.weights[d]) #Obtain label
                    if (trainingLabels[b] == label.argMax()):
                        pass
                    else:
                        maxLabel = label.argMax() #Get max label
                        countTwo = util.Counter()
                        tData = trainingData[b]
                        tLabel = trainingLabels[b]
                        weightCalc = (self.weights[maxLabel].__sub__(self.weights[tLabel])).__mul__(tData) + 1.0
                        multData = (2.0 / (tData.__mul__(tData)))
                        calc = weightCalc / multData #Calculate weights
                        minArg = min(c, calc) #Return minimum value
                          
                        for e in trainingData[b].keys():
                            countTwo[e] = minArg * trainingData[b][e]
                        self.weights[label.argMax()].__sub__(countTwo)
                        self.weights[trainingLabels[b]].__radd__(countTwo)
                        
            gridWeights[c] = self.weights #Update counter
            
            for f in range(len(validationData)): #Return length of validation data
                for g in validationLabels:
                    label[g] = validationData[f].__mul__(self.weights[g]) #Obtain label
                if (validationLabels[f] != label.argMax()):
                    pass
                else:
                    mySum += 1 #Add to sum if we are at max label   
            myCounter[c] = mySum #Update counter
            
        self.weights = gridWeights[myCounter.argMax()] #Update weights with highest value
    

    def classify(self, data ):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.
        
        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses