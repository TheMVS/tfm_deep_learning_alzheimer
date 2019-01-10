from Problems.Abstract.Problem import Problem


class OASIS(Problem):

    def __init__(self):
        'Constructor'
        super(OASIS, self).__init__('OASIS', '../OASIS_data/')

    def read_data(self):
        import cv2
        import numpy
        from numpy.ma import ceil
        from pandas import read_csv

        df = read_csv(self.get_data_path() + 'oasis_cross-sectional.csv').fillna(value=0)

        images = df.ix[:, 'ID'].tolist()

        X = []

        def postive(x):
            if int(ceil(x)) >= 1:
                return 1
            else:
                return 0

        Y = df.ix[:, 'CDR'].tolist()
        Y = list(map(postive, Y))

        for image in images:
            # Open image
            img = cv2.imread(self.get_data_path() + image + '_mpr-1_anon_sag_66.png')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = self.preprocess_image(img)

            X.append(img)

        X = numpy.array(X)
        (images,rows,columns)=X.shape
        X = X.reshape(images,rows,columns,1)

        return {'X': X, 'Y': numpy.array(Y)}
