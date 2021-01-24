import csv
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from os import getcwd

file_dir = f'{getcwd()}\\bbc-text.csv'
labels = []
articles = []

stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]

with open(file_dir, 'r') as f:
    fulltext = csv.reader(f, delimiter=',',)
    next(fulltext)
    for i in fulltext:
        labels.append(i[0])
        artemp = i[1:]

        # Check if the articles has more than 1 element in the list. 
        # If it has, join them to be one str.
        # But in this case none.
        if len(artemp) != 0:
            artemp = ' '.join(artemp)
        for stop in stopwords:
            stop1 = " "+ stop + " "
            artemp = artemp.replace(stop1,' ')
            artemp = artemp.replace('  ',' ')
        articles.append(artemp)

tokenizer = Tokenizer(oov_token='<oov>')
tokenizer.fit_on_texts(articles)
sequence_articles = tokenizer.texts_to_sequences(articles)
padded_sequence_articles = pad_sequences(sequence_articles, padding='post')
print(padded_sequence_articles[0:2])

# Bulid another tokerizer for labels. 
# Make sure not to use the same tokenizer with articles. 
# Or it would lead to mismatch.
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)
label_seq = label_tokenizer.texts_to_sequences(labels)
label_word_index = label_tokenizer.word_index

print(label_seq)
print(label_word_index)