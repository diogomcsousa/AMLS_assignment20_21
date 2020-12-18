from A1.a1_task import GenderDetection
from A2.a2_task import EmotionDetection
from B1.b1_task import FaceShapeDetection
from B2.b2_task import EyeColorDetection

from Utils.data_preprocessing import PreProcessor
from Utils.classifiers import BinaryClassifier, MultiClassClassifier

pre_processor_1 = PreProcessor()
binary = BinaryClassifier()
multiclass = MultiClassClassifier()

params = {'kernel': ['poly'], 'C': [1, 10, 100], 'degree': [2, 3, 4, 5]}

# ======================================================================================================================
# Task A1 - Gender Detection

model_A1 = GenderDetection(pre_processor_1, binary)
X_train, X_test, y_train, y_test = model_A1.feature_extraction('celeba/img')  # Data pre processing
acc_A1_train, clf = model_A1.train(X_train, y_train, params)  # Train model based on the training set
acc_A1_test = model_A1.test(X_test, y_test, clf)  # Test model based on the test set

X_test_extra, y_test_extra = model_A1.feature_extraction('celeba_test/img', extra_test=True)
acc_A1_test_extra = model_A1.test(X_test_extra, y_test_extra, clf)  # Test additional celeba_test with same classifier

# ======================================================================================================================
# Task A2 - Emotion Detection

model_A2 = EmotionDetection(pre_processor_1, binary)
X_train, X_test, y_train, y_test = model_A2.feature_extraction('celeba/img')  # Data pre processing
acc_A2_train, clf = model_A2.train(X_train, y_train, params)  # Train model based on the training set
acc_A2_test = model_A2.test(X_test, y_test, clf)  # Test model based on the test set

X_test_extra, y_test_extra = model_A2.feature_extraction('celeba_test/img', extra_test=True)
acc_A2_test_extra = model_A2.test(X_test_extra, y_test_extra, clf)  # Test additional celeba_test with same classifier

# ======================================================================================================================
# Task B1 - Face Shape Detection
params = {'kernel': ['poly'], 'C': [10, 100, 1000], 'degree': [2, 3, 4, 5, 6]}

model_B1 = FaceShapeDetection(pre_processor_1, multiclass)
X_train, X_test, y_train, y_test = model_B1.feature_extraction('cartoon_set/img')
acc_B1_train, clf = model_B1.train(X_train, y_train, params)
acc_B1_test = model_B1.test(X_test, y_test, clf)

X_test_extra, y_test_extra = model_B1.feature_extraction('cartoon_set_test/img', extra_test=True)
acc_B1_test_extra = model_B1.test(X_test_extra, y_test_extra, clf)  # Test additional celeba_test with same classifier

# ======================================================================================================================
# Task B2 - Eye Color Detection

model_B2 = EyeColorDetection(pre_processor_1, multiclass)
X_train, X_test, y_train, y_test = model_B2.feature_extraction('cartoon_set/img')
acc_B2_train, clf = model_B2.train(X_train, y_train, params)
acc_B2_test = model_B2.test(X_test, y_test, clf)

X_test_extra, y_test_extra = model_B2.feature_extraction('cartoon_set_test/img', extra_test=True)
acc_B2_test_extra = model_B2.test(X_test_extra, y_test_extra, clf)  # Test additional celeba_test with same classifier

# ======================================================================================================================
# Print out your results with following format:
print(
    'TA1:{},{},{};\nTA2:{},{},{};\nTB1:{},{},{};\nTB2:{},{},{};\n'.format(acc_A1_train, acc_A1_test, acc_A1_test_extra,
                                                                          acc_A2_train, acc_A2_test, acc_A2_test_extra,
                                                                          acc_B1_train, acc_B1_test, acc_B1_test_extra,
                                                                          acc_B2_train, acc_B2_test, acc_B2_test_extra))
