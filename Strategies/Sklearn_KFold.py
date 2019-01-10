from Strategies.Abstract.Strategy import Strategy


class Sklearn_KFold(Strategy):

    def __init__(self, params):
        'Constructor'
        super(Sklearn_KFold, self).__init__('Sklearn_KFold', params)

    def fit(self, model, data):
        import time, numpy
        from sklearn.model_selection import StratifiedKFold,train_test_split

        architecture = model.get_model()
        sklearn_model = model.get_sklearn_model()

        params = self.get_params()

        data_X = data['X']
        Y = data['Y']

        if model.get_name().lower() == 'vgg16':
            data_X = self.prepare_data(data_X, True)
        elif model.get_name().lower() == 'mobile_net' or model.get_name().lower() == 'resnet' or model.get_name().lower() == 'unet':
            data_X = self.prepare_data(data_X, False)

        X = []
        for element in data_X:
            shape = element.shape
            input = numpy.array(element)
            input = input.reshape(1, shape[0], shape[1], shape[2])
            X.append(architecture.predict(input, batch_size=1)[0])
        X = numpy.array(X)

        title = model.get_name() + '_' + time.strftime("%Y_%m_%d_%H_%M_%S")

        repetitions_scores = []
        for i in range(params['repetitions']):
            test_results = []
            test_orig = []

            if self.get_params()['verbose']:
                print('Repetition: ' + str(i + 1) + '\n')

            import random
            seed = random.randint(1, 1000)

            kfold = StratifiedKFold(n_splits=params['folds'], shuffle=True, random_state=seed)

            if self.get_params()['verbose']:
                print('Repetition: ' + str(i+1) + '\n')

            count = 1

            for train, test in kfold.split(X, Y):

                test_results = []
                test_orig = []

                if self.get_params()['verbose']:
                    print('Fold: ' + str(count))

                X_train, X_test, Y_train, Y_test = X[train], X[test], Y[train], Y[test]

                # Class weight if there are unbalanced classes
                from sklearn.utils import class_weight
                # class_weight = class_weight.compute_class_weight('balanced',numpy.unique(Y), Y)
                sample_weight = class_weight.compute_sample_weight(class_weight='balanced', y=Y_train)

                # Fit the architecture
                sklearn_model.fit(X_train, Y_train, sample_weight=sample_weight)

                # Fit the architecture
                sklearn_model.fit(X_train, Y_train)

                # Evaluate the architecture
                print('Evaluation metrics\n')

                if (self.get_params()['sklearn_model'].lower() != 'linear_regression'):
                    Y_predicted = sklearn_model.predict(X_test)
                    test_results += Y_predicted.tolist()
                    test_orig += Y_test.tolist()
                else:
                    Y_predicted = [0 if x < 0.5 else 1 for x in sklearn_model.predict(X_test).tolist()]
                    test_results += Y_predicted
                    test_orig += [int(numpy.round(x)) for x in Y_test.tolist()]

                count += 1

            if params['problem_type'].lower() == 'classification':
                scores = self.classification_metrics(numpy.array(test_results), test_orig, numpy.unique(Y))
            else:
                scores = self.regression_metrics(numpy.array(test_results), test_orig)

            import csv

            with open(title + '_test_output.csv', 'w+') as file:
                wr = csv.writer(file)
                wr.writerow(test_results)
            file.close()

            repetitions_scores.append(scores)

        global_score_mean = numpy.mean(repetitions_scores, axis=0)
        global_score_std = numpy.std(repetitions_scores, axis=0)

        if params['problem_type'].lower() == 'classification':
            metrics_names = ["sklearn_acc", "tn", "fp", "fn", "tp", "precision", "recall",
                                           "specificity", "f1", "auc_roc", "k"]
        else:
            metrics_names = ["mae", "r2"]


        file = open(title + '_results.txt', 'w')
        for count in range(len(metrics_names)):
            file.write(str(metrics_names[count]) + ": " + str(
                numpy.around(global_score_mean[count], decimals=4)) + chr(177) +str(numpy.around(global_score_std[count],decimals=4))+ '\n')
        file.close()
