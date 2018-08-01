from sklearn.neural_network import MLPClassifier
import json
import scikitplot as skplt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def showMatrix(matrix):
    fig, ax = plt.subplots()
    ax.matshow(matrix, cmap=plt.cm.Blues)

    XArr = np.asarray(matrix)
    if XArr.shape[0]<=50 and XArr.shape[1]<=50:
        for i in range(XArr.shape[1]):
            for j in range(XArr.shape[0]):
                c = XArr[j, i]
                ax.text(i, j, str(c), va='center', ha='center')

    fig.show()


if __name__ == "__main__":
    # load text data
    with open('data 2/AT.json') as js:
        assetList = json.load(js)
    with open('data 2/ATkeys.json') as js:
        assetKeys = json.load(js)
    with open('data 2/ATraw.json') as js:
        assets = json.load(js)
    with open('data 2/DT.json') as js:
        docList = json.load(js)
    with open('data 2/DTKeys.json') as js:
        docKeys = json.load(js)
    with open('data 2/DTraw.json') as js:
        documents = json.load(js)

    #showMatrix(docList)

    # add empty termvec for uncertainty class
    assetList.append([0] * len(assetList[0]))
    scaler = StandardScaler()
    scaler.fit(assetList)

    X_train = assetList
    X_train_scaled = scaler.transform(X_train)
    y = [1, 2, 3, 4, 0]
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

    clf.fit(X_train, y)
    MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
                  beta_1=0.9, beta_2=0.999, early_stopping=False,
                  epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='constant',
                  learning_rate_init=0.001, max_iter=200, momentum=0.9,
                  nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
                  solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
                  warm_start=False)

    X_test = docList
    X_test_scaled = scaler.transform(X_test)
    predictions = clf.predict(X_test)
    plt.figure()
    skplt.metrics.plot_confusion_matrix([3, 1, 4, 4, 0, 0, 0, 1, 0, 0], predictions, normalize=True)
    plt.show()
