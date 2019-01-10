from Problems.Abstract.Problem import Problem


class ADNI(Problem):

    def __init__(self):
        'Constructor'
        super(ADNI, self).__init__('ADNI', '../ADNI_MP-RAGE_REPEAT_data/')

    def read_data(self):
        import cv2
        import numpy
        from numpy.ma import ceil
        from pandas import read_csv

        df = read_csv(self.get_data_path() + 'MP-RAGE_REPEAT_data.csv').fillna(value=0)

        subjects = df.ix[:, 'Subject'].tolist()
        ids = df.ix[:, 'Image Data ID'].tolist()

        X = []

        def postive(x):
            if x == "CN":
                return 0
            else:
                return 1

        Y = df.ix[:, 'Group'].tolist()
        Y = list(map(postive, Y))

        for i in range(len(subjects)):
            # Open image
            img = cv2.imread(self.get_data_path() + subjects[i] +"_"+str(ids[i]) + '.png')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = self.preprocess_image(img)

            X.append(img)

        X = numpy.array(X)
        (images,rows,columns)=X.shape
        X = X.reshape(images,rows,columns,1)

        return {'X': X, 'Y': numpy.array(Y)}
