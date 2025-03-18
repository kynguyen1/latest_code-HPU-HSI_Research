import cupy as cp


def weightedMedian(data, weights):
    wSum = cp.sum(weights)

    # For the borders - Dr. Kursun
    if cp.isnan(wSum):
        return 0

    # Make sure sum of weights is one (vectorized)
    weights /= wSum

    # Sort elements, weights, and data simultaneously using argsort (vectorized)
    ind = cp.argsort(data)
    weights = weights[ind]
    data = data[ind]

    # Cumulative sum of weights (vectorized)
    weightSum = cp.cumsum(weights)

    # Find the median index (vectorized)
    j = cp.where(weightSum >= 0.5)[0][0]

    return data[j]


def guidedMedianFilter_gpu(inputImg, hsiData):

    outputImg = cp.zeros(cp.shape(inputImg))
    predImage = cp.pad(inputImg, ((1, 1), (1, 1)), 'edge')
    hsiData = cp.pad(hsiData, ((1, 1), (1, 1), (0, 0)), 'edge')

    rows, cols = cp.shape(predImage)[:2]
    bands = hsiData.shape[-1]

    # Vectorized loop using slicing and broadcasting
    for i in range(1, rows - 2):
        for j in range(1, cols - 2):
            predImgWindow = predImage[i - 1: i + 2, j - 1: j + 2]
            hsiDataWindow = hsiData[i - 1: i + 2, j - 1: j + 2, :]

            # Set central pixel of predImgWindow to 0 (vectorized)
            predImgWindow[1, 1] = 0

            # Calculate correlation coefficients using broadcasting (vectorized)
            coeff = cp.corrcoef(hsiDataWindow, rowvar=False)[:, 0]

            # Filter coefficients and pixel values based on non-zero predImgWindow values
            coeff = coeff[predImgWindow.ravel() != 0]
            pImg = predImgWindow.ravel()[predImgWindow.ravel() != 0]

            if cp.sum(predImgWindow.ravel()) == 0:
                outputImg[i - 1, j - 1] = 0
            else:
                outputImg[i - 1, j - 1] = weightedMedian(pImg, coeff)

    return cp.asnumpy(outputImg)  # Convert back to NumPy for compatibility

