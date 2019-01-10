import abc


class Strategy(abc.ABCMeta('ABC', (object,), {'__slots__': ()})):
    'Common base class for all strategies'
    __name = None
    __params = None

    def __init__(self, name, params):
        'Constructor'
        self.__name = name
        self.__params = params

    # Getters

    def get_name(self):
        return self.__name

    def get_params(self):
        return self.__params

    # Setters

    # Other methods

    @abc.abstractmethod
    def fit(self, model, data):
        'Trains the model with new data'
        pass

    def predict(self, model, data):
        'Predicts classes for new data'
        predictions = model.predict(data['X'])

    def prepare_data(self, X, is_vgg16):
        'Prepares data for VGG16 and Mobile net'
        import cv2
        import numpy as np
        aux = X
        if len(X[0].shape) < 4:
            MEAN_VALUE = np.array([103.939, 116.779, 123.68])  # BGR
            aux = [cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC) for img in X]
            aux[:] = [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in aux]
            if is_vgg16:
                aux[:] = [img - MEAN_VALUE for img in aux]
        aux = np.array(aux)
        return aux

    def encode_output(self, Y, vocab_size):
        from keras.utils import to_categorical
        import numpy as np
        ylist = list()
        for output in Y:
            encoded = to_categorical(output, num_classes=vocab_size)
            ylist.append(encoded)
        y = np.array(ylist)
        y = y.reshape(Y.shape[0], 1, vocab_size)
        return y

    def classification_metrics(self, Y_predicted, Y_test, labels):
        from sklearn.metrics import confusion_matrix, accuracy_score,log_loss, f1_score, precision_score, recall_score, roc_auc_score, cohen_kappa_score
        import numpy as np

        acc = accuracy_score(Y_test,Y_predicted)
        f1 = f1_score(Y_test,Y_predicted,labels=labels)
        precision = precision_score(Y_test,Y_predicted,labels=labels)
        recall = recall_score(Y_test,Y_predicted,labels=labels)
        tn, fp, fn, tp = confusion_matrix(Y_test,Y_predicted,labels=labels).ravel()
        specificity = float(tn) / float(tn + fp)

        if len(np.unique(Y_predicted)) >= 2:
            roc = roc_auc_score(Y_test,Y_predicted)
        else:
            roc = 0.5

        k = cohen_kappa_score(Y_test,Y_predicted,labels=labels)

        return [acc, tn, fp, fn, tp, precision, recall, specificity, f1, roc, k]

    def regression_metrics(self, Y_predicted, Y_test):
        from sklearn.metrics import r2_score,mean_absolute_error
        import numpy as np

        Y_test = np.argmax(Y_test, axis=1)

        r2 = r2_score(Y_test,Y_predicted)
        mae = mean_absolute_error(Y_test,Y_predicted)

        return [mae,r2]