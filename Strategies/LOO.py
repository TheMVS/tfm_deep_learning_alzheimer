from Strategies.Abstract.Strategy import Strategy


class LOO(Strategy):

    def __init__(self, params):
        'Constructor'
        super(LOO,self).__init__('LOO', params)

    def fit(self, model, data):
        import time
        import keras
        import numpy
        from keras.callbacks import ModelCheckpoint, EarlyStopping
        from sklearn.model_selection import LeaveOneOut

        architecture = model.get_model()
        params = self.get_params()

        X = data['X']
        Y = data['Y']

        if model.get_name().lower() == 'vgg16':
            X = self.prepare_data(X, True)
        elif model.get_name().lower() == 'mobile_net' or model.get_name().lower() == 'resnet' or model.get_name().lower() == 'unet':
            X = self.prepare_data(X, False)


        title = model.get_name() + '_' + time.strftime("%Y_%m_%d_%H_%M_%S")

        import random
        seed = random.randint(1, 1000)

        loo = LeaveOneOut()
        loo.get_n_splits(X)

        count = 1

        test_results = []
        test_orig = []
        cvscores = []
        for train, test in loo.split(X, Y):

            if self.get_params()['verbose']:
                print('Fold: ' + str(count))

            X_train, X_test = X[train], X[test]

            Y_train, Y_test = Y[train], Y[test]

            validation_percentage = params['validation_split']

            # Class weight if there are unbalanced classes
            from sklearn.utils import class_weight
            # class_weight = class_weight.compute_class_weight('balanced',numpy.unique(Y), Y)
            sample_weight = class_weight.compute_sample_weight(class_weight={1:1,0:0.5}, y=Y_train)

            if (self.get_params()[
                'problem_type'] == 'classification'):  # and not(params['use_distillery'] and model.get_name().lower() != 'distillery_network'):
                Y_train = keras.utils.to_categorical(Y_train, num_classes=len(numpy.unique(Y)))
                Y_test = keras.utils.to_categorical(Y_test, num_classes=len(numpy.unique(Y)))

            callbacks_list = []
            callbacks_list.append(
                ModelCheckpoint(title + "_loo_weights_improvement.hdf5", monitor='val_loss',
                                verbose=params['verbose'], save_best_only=True, mode='min'))
            callbacks_list.append(
                EarlyStopping(monitor='val_loss', min_delta=params['min_delta'], patience=params['patience'],
                              verbose=params['verbose'], mode='min'))

            # Fit the architecture
            architecture.fit(X_train, Y_train, epochs=params['epochs'], batch_size=params['batch'],validation_split=validation_percentage,
                             callbacks=callbacks_list,
                             verbose=params['verbose'],sample_weight=sample_weight)


            # Data Augmentation
            if params['augmentation']:
                from keras.preprocessing.image import ImageDataGenerator

                # Initialize Generator
                datagen = ImageDataGenerator(vertical_flip=True)

                # Fit parameters from data
                datagen.fit(X_train)

                # Fit new data
                architecture.fit_generator(datagen.flow(X_train, Y_train, batch_size=params['batch']))

            model_json = architecture.to_json()
            if count == 1:
                with open(title + '_model.json', "w") as json_file:
                    json_file.write(model_json)
                architecture.save_weights(title + '_weights.h5') # TODO: Check what weights save
                file = open(title + '_seed.txt', 'a+')
                file.write('Repetition: ' + ' , seed: ' + str(seed)+'\n')
                file.close()

            count += 1

            # Evaluate the architecture
            print('Evaluation metrics\n')
            scores = architecture.evaluate(X_test, Y_test, verbose=params['verbose'])
            cvscores.append(scores)
            Y_predicted = numpy.argmax(architecture.predict(X_test, batch_size=params['batch']), axis=1)
            test_results += Y_predicted.tolist()
            test_orig += numpy.argmax(Y_test, axis=1).tolist()

        import csv

        with open(title + '_test_output.csv', 'w+') as file:
            wr = csv.writer(file)
            wr.writerow(test_results)
        file.close()

        if params['problem_type'].lower() == 'classification':
            scores2 = self.classification_metrics(numpy.array(test_results), test_orig, numpy.unique(Y))
        else:
            scores2 = self.regression_metrics(numpy.array(test_results), test_orig)

        scores_mean = numpy.array(numpy.mean(cvscores, axis=0).tolist() + scores2)
        scores_std = numpy.array(numpy.std(cvscores, axis=0).tolist() + scores2)

        if params['problem_type'].lower() == 'classification':
            architecture.metrics_names += ["sklearn_acc", "tn", "fp", "fn", "tp", "precision", "recall",
                                           "specificity", "f1", "auc_roc", "k"]
        else:
            architecture.metrics_names += ["mae", "r2"]

        file = open(title + '_results.txt', 'w')
        for count in range(len(architecture.metrics_names)):
            file.write(str(architecture.metrics_names[count]) + ": " + str(numpy.around(scores_mean[count],decimals=4))+ chr(177) + str(numpy.around(scores_std[count],decimals=4))+ '\n')
        file.close()