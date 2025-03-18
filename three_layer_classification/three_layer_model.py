import cupy as cp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


class threeLayerHSIClassificationGPU(BaseEstimator, ClassifierMixin):
    """Fits a logistic regression model on tree embeddings.
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.scaler = MinMaxScaler(feature_range=(0, 0.95), clip=True)

    def fit(self, X, y):
        numOfSamples, numOfFeatures = X.shape
        self.numOfClasses = np.unique(y).shape[0]

        X_scaled = self.scaler.fit_transform(X)
        X_scaled_gpu = cp.asarray(X_scaled)

        # There are 16 contours, each has 100x200 matrix.
        # 100 is the number of bins in the histogram. And 200 is the number of features.
        numOfContourBins = 100
        self.allContours = cp.zeros((self.numOfClasses, numOfContourBins, numOfFeatures))

        # There are 16 references. Each has 200x1 vector.
        self.allReferences = cp.zeros((self.numOfClasses, numOfFeatures))

        for classLabel in range(self.numOfClasses):
            P_c = X_scaled_gpu[(y == classLabel + 1), :]  # current class training samples
            P_r = cp.asarray(X[(y == classLabel + 1), :])  # current class training samples

            # Find the contour
            contour = self.findContour_gpu(P_c, numOfFeatures)
            self.allContours[classLabel] = contour

            # Find the reference
            # Add a if statement so that it doesn't get stuck on Fusion
            if P_r.shape[0] > 0:
                reference = cp.percentile(P_r, 50, axis=0)
                self.allReferences[classLabel] = reference

    # % This function transforms a sample using contours and references
    def transform(self, X, y=None):
        numOfSamples, numOfFeatures = X.shape

        X_transformed = cp.zeros((numOfSamples, 2 * self.numOfClasses))
        X_scaled = self.scaler.transform(X)
        X_scaled_gpu = cp.asarray(X_scaled)

        # This loops transforms every sample. Finds new features from contour and reference learners.
        # Added disable = True to get rid of print statements
        for tsInd in tqdm(range(0, numOfSamples), desc="Transforming Samples", disable=True):
            # Removed tqdm to get rid of printing statements
            # for tsInd in range(0,numOfSamples)
            P_c = X_scaled_gpu[tsInd, :]
            P_r = cp.asarray(X[tsInd, :])

            contour_scores = cp.zeros((self.numOfClasses))
            referrence_scores = cp.zeros((self.numOfClasses))

            for classLabel in range(self.numOfClasses):
                contour = self.allContours[classLabel]
                contour_scores[classLabel] = self.findContourScore_gpu(P_c, contour, numOfFeatures)

                reference = self.allReferences[classLabel]
                referrence_scores[classLabel] = self.findReferenceScore_gpu(P_r, reference, numOfFeatures)

            X_transformed[tsInd, :] = cp.hstack((contour_scores, referrence_scores))

        return cp.asnumpy(X_transformed)  # Convert back to NumPy for compatibility

    # % This function finds the contour of each class
    def findContour_gpu(self, P_c, numOfFeatures):
        contour = cp.zeros(shape=(100, numOfFeatures))
        for j in range(numOfFeatures):
            f_j = P_c[:, j]
            hist, bin_edges = cp.histogram(f_j, bins=cp.arange(0, 1.01, 0.01), density=True)
            contour[:, j] = hist
        return contour

    # % This function calculates contour score of a test sample
    def findContourScore_gpu(self, testSample, contour, numOfFeatures):
        ind = cp.digitize(testSample, cp.arange(0, 1.01, 0.01))
        p = contour[ind, range(numOfFeatures)]

        return cp.sum(p)

    # % This function calculates reference score of a test sample
    def findReferenceScore_gpu(self, testSample, reference, numOfFeatures):
        c = cp.corrcoef(testSample, reference)
        c[cp.isnan(c)] = 0
        return max(0.0, c[0][1])

