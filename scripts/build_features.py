# Functions used to build features, for training of CRF model



def word2features(sent, i):
    '''Extract features from word at position i in sentence sent'''

    # Extract features from current word i
    word = sent[i]
    features = {
        'word.lower()': word.lower(),       # lowercase 
        'word.isupper()': word.isupper(),   # are all capital letters?
        'word.istitle()': word.istitle(),   # is title case?
        'word.isdigit()': word.isdigit(),   # is a number?
    }
    if i > 0:
        # Extract features from previous word (i-1)
        prev_word = sent[i-1]
        features.update({
            '-1:word.lower()': prev_word.lower(),
            '-1:word.istitle()': prev_word.istitle(),
            '-1:word.isupper()': prev_word.isupper(),
        })
    else:
        # Beginning of sentence
        features['BOS'] = True

    if i < len(sent) - 1:
        # Extract features from next word (i+1)
        next_word = sent[i+1]
        features.update({
            '+1:word.lower()': next_word.lower(),
            '+1:word.istitle()': next_word.istitle(),
            '+1:word.isupper()': next_word.isupper(),
        })
    else:
        # End of sentence
        features['EOS'] = True

    return features



def sent2features(sent):
    '''Apply word2features to each word in the sentence sent'''
    return [word2features(sent, i) for i in range(len(sent))]