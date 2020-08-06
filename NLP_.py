import sklearn_crfsuite
import nltk
from sklearn_crfsuite.metrics import flat_classification_report

label_to_idx = {'B-location': 0, 'I-location': 1, 'B-title': 2, 'I-title': 3, 'B-company': 4, 'I-company': 5,
                'B-product': 6, 'I-product': 7, 'B-person': 8, 'I-person': 9, 'O': 11, 'B-other': 12, 'I-other': 13,
                'B-group': 14, 'I-group': 15}
inx_label = {0: 'B-location', 1: 'I-location', 2: 'B-title', 3: 'I-title', 4: 'B-company', 5: 'I-company',
                6: 'B-product', 7: 'I-product', 8: 'B-person', 9: 'I-person', 11: 'O', 12: 'B-other', 13: 'I-other',
                14: 'B-group', 15: 'I-group'}

# read the data
train_data = open('train.txt', 'r')
dev_data_ = open('dev.txt', 'r')

sentences = []
y_train = []
s = []
y_ = []
for line in train_data:
    if line.strip():
        line_ = line.split()
        word = line_[0]
        s.append(word)
        y_.append(line_[1])
    else:
        sentences.append(s)
        y_train.append(y_)
        s = []
        y_ = []

sentence_list = []
for i in range(len(sentences)):
    sent = []
    pos = nltk.pos_tag(sentences[i])

    for j in range(len(pos)):
        sent.append((pos[j][0], pos[j][1], inx_label[label_to_idx[y_train[i][j]]]))

    sentence_list.append(sent)


sentences_ = []
Y_train = []
s_ = []
Y_ = []
for line in dev_data_:
    if line.strip():
        line__ = line.split()
        word_ = line__[0]
        s_.append(word_)
        Y_.append(line__[1])
    else:
        sentences_.append(s_)
        Y_train.append(Y_)
        s_ = []
        Y_ = []

dev_list = []
for i in range(len(sentences_)):
    sent_ = []
    pos_ = nltk.pos_tag(sentences_[i])

    for j in range(len(pos_)):
        sent_.append((pos_[j][0], pos_[j][1], inx_label[label_to_idx[Y_train[i][j]]]))

    dev_list.append(sent_)

test_data = open('test_no_tag.txt', 'r', encoding="utf8", )
sentences3 = []
s3 = []
for line in test_data:
    if line.strip():
        line_ = line.split()
        word_3 = line_[0]
        s3.append(word_3)
    else:
        sentences3.append(s3)
        s3 = []

test_list = []
for i in range(len(sentences3)):
    sent_1 = []
    pos = nltk.pos_tag(sentences3[i])
    test_list.append(pos)

# Feature Extraction


def word2features(sent2, k):
    word2 = sent2[k][0]
    postag = sent2[k][1]

    features = {
        'bias': 1.0,
        'word.lower()': word2.lower(),
        'word[-3:]': word2[-3:],
        'word[-2:]': word2[-2:],
        # 'word[:3]': word2[:3],
        # 'word[:2]': word2[:2],
        'word.isupper()': word2.isupper(),
        'word.istitle()': word2.istitle(),
        'word.isdigit()': word2.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if k > 0:
        word1 = sent2[k - 1][0]
        postag1 = sent2[k - 1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if k < len(sent2) - 1:
        word1 = sent2[k + 1][0]
        postag1 = sent2[k + 1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent2):
    return [word2features(sent2, k) for k in range(len(sent2))]


def sent2labels(sent2):
    return [label for token, postag, label in sent2]


def sent2tokens(sent2):
    return [token for token, postag, label in sent2]


X = [sent2features(s) for s in sentence_list]
y = [sent2labels(s) for s in sentence_list]
X_dev = [sent2features(s) for s in dev_list]
y_dev = [sent2labels(s) for s in dev_list]

# train CRF model

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X+X_dev, y+y_dev) # train model on train and dev sets

# Evaluation:

y_pred = crf.predict(X_dev)
report = flat_classification_report(y_pred, y_dev)
print(report)

# test_data prediction
X_test = [sent2features(s) for s in test_list]

y_pred_test = crf.predict(X_test)

# writing the y_pred_test in a text file
words = []
tag = []
for i in range(len(sentences3)):
    for j in range(len(sentences3[i])):
        words.append((sentences3[i])[j])
        tag.append((y_pred_test[i])[j])
word_tag = words + tag
outF = open("test prediction 3.txt", "w", encoding='utf8')

for m in range(len(words)):
    outF.write(word_tag[m])
    outF.write("\t")
    outF.write(word_tag[m+len(tag)])
    outF.write('\n')





