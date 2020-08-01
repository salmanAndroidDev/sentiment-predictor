import numpy as np
import sys

np.random.seed(1)

class SentimentNetwork:
    def __init__(self, raw_data, raw_label, alpha= 0.01, hidden_size= 100, iterations= 3):
        self.alpha = alpha
        self.hidden_size = hidden_size
        self.iterations= iterations
        self.raw_data = raw_data
        self.raw_label= raw_label

        self.input_dataset, self.token, self.word2index, self.vocabs = self.prepare_input_data()
        self.target_dataset = self.get_labels()
        
        self.weights_0_1 = 0.2*np.random.random((len(self.vocabs),hidden_size)) - 0.1
        self.weights_1_2 = 0.2*np.random.random((hidden_size,1)) - 0.1


    # Here is action functions
    def sigmoid(self, x):
        return 1/ (1 + np.exp(-x))
    
    def relu(self, x):
        return (x > 0) * x

    def train(self):
        """ training neural network """
        correct,total = (0,0)
        
        for iter in range(self.iterations):
            for i in range(len(self.input_dataset)-1000):
                x,y = (self.input_dataset[i],self.target_dataset[i])

                layer_1 = self.sigmoid(np.sum(self.weights_0_1[x],axis=0))
                layer_2 = self.sigmoid(np.dot(layer_1,self.weights_1_2)) 

                layer_2_delta = layer_2 - y
                layer_1_delta = layer_2_delta.dot(self.weights_1_2.T)
                
                self.weights_0_1[x] -= layer_1_delta * self.alpha
                self.weights_1_2 -= np.outer(layer_1,layer_2_delta) * self.alpha

                if(np.abs(layer_2_delta) < 0.5):
                    correct += 1
                total += 1
                if(i % 10 == 9):
                    progress = str(i/float(len(self.raw_data)))
                    sys.stdout.write('\rIter:'+str(iter)\
                                    +' Progress:'+progress[2:4]\
                                    +'.'+progress[4:6]\
                                    +'% Training Accuracy:'\
                                    + str(correct/float(total)) + '%')
                    print()

    
    def test(self):
        """Testing accuracy with 1000 sentences"""
        correct,total = (0,0)

        for i in range(len(self.input_dataset)-1000,len(self.input_dataset)):
            x = self.input_dataset[i]
            y = self.target_dataset[i]

            layer_1 = self.sigmoid(np.sum(self.weights_0_1[x],axis=0))
            layer_2 = self.sigmoid(np.dot(layer_1,self.weights_1_2))

            if(np.abs(layer_2 - y) < 0.5):
                correct += 1
            total += 1
        print("Test Accuracy:" + str(correct / float(total)))

    def predict(self, sent):
        """Predicting sentement """
        text = sent.split(" ")
        sentence = []
        for word in text:
                try:
                    sentence.append(self.word2index[word.replace(' ','')])
                except Exception as e:
                    pass 
        
        input = list(set(sentence))           
        layer_1 = self.sigmoid(np.sum(self.weights_0_1[input],axis=0))
        layer_2 = self.sigmoid(np.dot(layer_1,self.weights_1_2))

        print("{}% Positive".format(layer_2))

    def prepare_input_data(self):
        """Converting text into numbers for training"""
        token = list(map(lambda sentence: set(sentence.split(' ')) ,self.raw_data))
        vocabs = set()

        for sent in token:
            for word in sent:
                if len(word) > 0:
                    vocabs.add(word)
        vocabs = list(vocabs)

        word2index = {}
        for index, word in enumerate(vocabs):
            word2index[word]= index

        input_dataset = list()
        for sent in token:
            raw_sent = list()
            for word in sent:
                try:
                    raw_sent.append(word2index[word])
                except Exception as e:
                    pass
            input_dataset.append(list(set(raw_sent)))

        return (input_dataset,token,word2index, vocabs)    

    def get_labels(self):
        """Converting raw labels into 0 and 1 """
        target_dataset = list()
        for label in self.raw_label:
            if label == 'positive\n':
                target_dataset.append(1)
            else:
                target_dataset.append(0)
        return target_dataset