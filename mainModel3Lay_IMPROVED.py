import cupy as cp
from three_layer_classification.three_layer_model_gpu import threeLayerHSIClassification_gpu
from three_layer_classification.guidedMedianFilter_gpu import guidedMedianFilter_gpu
from sklearn.ensemble import RandomForestClassifier

# Assuming build_dataset is already optimized (pre-loading data)
X_train, y_train = build_dataset(img, train_gt, ignored_labels=IGNORED_LABELS)
X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
class_weight = "balanced" if CLASS_BALANCING else None

train1 = time.perf_counter()

# Start Training
# Train the first layer of the system on GPU
clf = threeLayerHSIClassification_gpu()
clf.fit(cp.asarray(X_train), cp.asarray(y_train))

# Transform training samples for Random Forest (Second Layer)
X_train_transformed = clf.transform(cp.asarray(X_train))

# Convert back to NumPy for compatibility with Random Forest
X_train_transformed = cp.asnumpy(X_train_transformed)

y_train = np.nan_to_num(y_train)
rf = RandomForestClassifier(n_estimators=500)
rf.fit(X_train_transformed, y_train)

# Transform all data for generating the whole image (GPU)
X_transformed = clf.transform(cp.asarray(img.reshape(-1, N_BANDS)))
X_transformed = cp.asnumpy(X_transformed)

# Stop Training
training_time = time.perf_counter() - train1

# Start Testing
test_time1 = time.perf_counter()

# Prediction using Random Forest
prediction = rf.predict(X_transformed)
prediction = prediction.reshape(img.shape[:2])

# Third layer starts here - GMF on GPU
prediction = guidedMedianFilter_gpu(cp.asarray(prediction), cp.asarray(img))
prediction = cp.asnumpy(prediction)

# ... rest of your code
testing_time = time.perf_counter() - test_time1

if MODEL == "twoLayer":
    prediction = prediction2
    testing_time = testing_time2

