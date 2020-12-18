class EyeColorDetection:
    def __init__(self, pre_processor, classifier):
        self.pre_processor = pre_processor
        self.classifier = classifier

    def feature_extraction(self, images_dir, extra_test=False):
        return self.pre_processor.haar_cascade_feature_extraction(label=1, images_dir=images_dir, extra_test=extra_test)

    def train(self, X_train, y_train, params):
        return self.classifier.fit(X_train, y_train, params)

    def test(self, X_test, y_test, clf):
        return self.classifier.predict(X_test, y_test, clf)


