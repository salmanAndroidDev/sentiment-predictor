from sentiment_network import SentimentNetwork

f = open('reviews.txt')
raw_reviews = f.readlines()
f.close()

f = open('labels.txt')
raw_labels = f.readlines()
f.close()

network = SentimentNetwork(raw_reviews, raw_labels)

network.train()
network.test()
print(".................Enter q to exit...............\n")
while(True):
    text = input('Enter a text: ')

    if text.lower() == 'q': break
    
    network.predict(text)
