import numpy as np

import csv

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import SGD
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

from keras.layers import Embedding

from keras.optimizers import *
from keras.models import load_model


def encoding(data, total):
    A = np.array([[1.0]])
    C = np.array([[2.0]])
    G = np.array([[3.0]])
    T = np.array([[4.0]])

    # len_kernel = 0
    Train_set = np.zeros((total, 14), dtype=np.float32)

    for i in range(0, total):
        letters = data[i]
        for j in range(0, 14):
            if letters[j] == 'A':
                Train_set[i, j] = A
            if letters[j] == 'C':
                Train_set[i, j] = C
            if letters[j] == 'G':
                Train_set[i, j] = G
            if letters[j] == 'T':
                Train_set[i, j] = T

    return Train_set

# def test_encoding():
#     file_train = open('train2.csv')
#     train_entire = csv.reader(file_train)
# 
#     train_data = []
# 
#     for i in train_entire:
#         train_data.append(i)
# 
#     letters = np.array(train_data)[1:, 1]
#     result_en = encoding(letters, 2)
#     print(result_en)


def main():
    file_train = open('train.csv')
    train_entire = csv.reader(file_train)

    train_data = []

    for i in train_entire:
        train_data.append(i)

    letters = np.array(train_data)[1:, 1]
    label = np.array(train_data)[1:, 2]
    label = np.array(label, dtype='float')
    label = label[:, np.newaxis]
    input = encoding(letters, 2000)


    model = Sequential()
    model.add(Embedding(14, output_dim=28))
    model.add(Conv1D(32, 6, activation='relu', padding="same"))
    model.add(Conv1D(32, 6, activation='relu', padding="same"))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.25))

    model.add(Conv1D(64, 6, activation='relu', padding="same"))
    model.add(Conv1D(64, 6, activation='relu', padding="same"))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.25))

    model.add(Dense(256, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=True)

    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    print(model.summary())

    model.fit(input[:], label[:], epochs = 294, shuffle=True)

    evaluate = model.evaluate(input[:], label[:])

    print("Result - Loss & Accuracy :")
    print(evaluate)

    model.save('cnn_vgg.h5')



def classify():
    model = load_model('cnn_vgg.h5')
    file_test = open('test.csv')
    test_entire = csv.reader(file_test)

    test_data = []

    for i in test_entire:
        test_data.append(i)

    letters = np.array(test_data)[1:, 1]
    test_input = encoding(letters, 400)

    results = model.predict(test_input)
    #print(len(results))
    #print(results)

    file_submission = open('submissioncnnvgg.csv', 'w')
    file_submission.write("id,prediction\n")
    for i in range(len(results)):
        if results[i][0] >= 0.5:
            line = "" + str(i) + "," + "1" + "\n"
        else:
            line = "" + str(i) + "," + "0" + "\n"

        file_submission.write(line)
    file_submission.close()

    print("DONE")

if __name__ == "__main__":
    main()
    #classify()
    #test_encoding()