import numpy as np
import time
class KNN:
    def __init__(self, k):
        # KNN state here
        # Feel free to add methods
        self.k = k

    def distance(self, featureA, featureB):
        diffs = (featureA - featureB) ** 2
        return np.sqrt(diffs.sum())

    def train(self, X, y):
        # training logic here
        # input is an array of features and labels
        # implementing train is just memorizing the data
        self.features = X
        self.labels = y

    def predict(self, X):
        # Run model here
        # Return array of predictions where there is one prediction for each set of features
        prediction = np.array([])
        for x in X:
            # Calculate Distance
            distances = np.array([self.distance(x, feature) for feature in self.features])
            # Zip Data
            neighbors = zip(distances, self.labels)
            # K Nearest Neighbors
            kNearestNeighbors = sorted(neighbors, key=lambda x: x[0])[:self.k]
            # Classify by Majority Voting
            classification = self.majorityVoting(kNearestNeighbors)
            # Add Prediction
            prediction = np.append(prediction, [classification])
        return prediction

    def majorityVoting(self, kNearestNeighbors):
        result = {}
        # voting according to label count
        for neighbor in kNearestNeighbors:
            label = neighbor[1]
            if label not in result:
                result[label] = 1
            else:
                result[label] += 1
        # use the max
        return max(result, key=result.get)


class ID3:
    def __init__(self, nbins, data_range):
        # Decision tree state here
        # Feel free to add methods
        self.bin_size = nbins
        self.range = data_range

    def preprocess(self, data):
        # Our dataset only has continuous data
        norm_data = np.clip((data - self.range[0]) / (self.range[1] - self.range[0]), 0, 1)
        categorical_data = np.floor(self.bin_size * norm_data).astype(int)
        return categorical_data

    def train(self, X, y):
        # training logic here
        # input is array of features and labels
        categorical_data = self.preprocess(X)
        features = []
        labels = []
        # Prepare features to create tree
        for i in range(len(categorical_data)):
            feature = []
            for j in categorical_data[i]:
                feature.append(j)
            feature.append(y[i])
            features.append(feature)
        # Prepare labels to create tree
        for i in range(len(features[0])):
            labels.append(i)
        # Create tree
        self.tree = self.createTree(features, labels)

    def createTree(self, data, labels):
        classList = [x[-1] for x in data]
        # Stop split if completely same class
        if classList.count(classList[0]) == len(classList):
            return classList[0]
        # Return labels show the most after go through all features using majority count
        if len(data[0]) == 1:
            count = {}
            # Count how many times each element shows up
            for vote in classList:
                if vote not in count.keys():
                    count[vote] = 1
                count[vote] += 1
            # return element shows the most
            return sorted(count.items(), key=lambda x: x[1], reverse=True)[0][0]
        # Use info gain to choose the best feature to split
        # Count features, best infoGain and its feature
        features = len(data[0]) - 1
        bestInfoGain = float("-inf")
        bestFeature = float("-inf")
        # Calculate expected information entropy for the whole dataset
        entropy = self.entropy(data)
        # Go through all features
        for i in range(features):
            # Get all features for each element
            featList = [feat[i] for feat in data]
            # Use set to eliminate duplicates
            uniqueVals = set(featList)
            entropy_ = 0.0
            # Calculate information gain
            for feat in uniqueVals:
                subData = self.split(data, i, feat)
                probability = float(len(subData) / len(data))
                entropy_ += probability * self.entropy(subData)
            infoGain = entropy - entropy_
            # Choose best info gain and its feature
            if (infoGain > bestInfoGain):
                bestInfoGain = infoGain
                bestFeature = i
        # Acquire labels for best feature and use it to create tree
        bestFeatLabel = labels[bestFeature]
        theTree = {bestFeatLabel: {}}
        del (labels[bestFeature])
        # Get all feature value from dataset
        featValues = [x[bestFeature] for x in data]
        # Use set to eliminate duplicates
        uniqueVals = set(featValues)
        # Traverse features and create decision tree
        for value in uniqueVals:
            subLabels = labels[:]
            theTree[bestFeatLabel][value] = self.createTree(self.split(data, bestFeature, value), subLabels)
        return theTree

    def entropy(self, data):
        # Calculate Shannon Entropy
        entropy = 0.0
        # Transfer label to dictionary
        labels = {}
        # Extract label from data
        for feature in data:
            label = feature[-1]
            # If label not in dictionary, put it in
            if label not in labels.keys():
                labels[label] = 1
        # Calculating shannon entropy
        for label in labels:
            # Probability of choosing each label
            probability = float(labels[label] / len(data))
            # Info(D) function
            entropy -= probability * np.log2(probability)
        return entropy

    def split(self, data, axis, value):
        # Split dataset based on given feature
        result = []
        for feature in data:
            if feature[axis] == value:
                # remove axis part and get expected data
                reducedFeat = feature[:axis]
                reducedFeat.extend(feature[axis + 1:])
                result.append(reducedFeat)
        return result

    def predict(self, X):
        features = self.preprocess(X)
        featureDic = {}
        predictions = []
        for feature in features:
            for x in range(len(feature)):
                # Transform feature to dictionary in order to query from trees
                featureDic[x] = feature[x]
            prediction = self.getLabelFromTree(featureDic, self.tree)
            predictions.append(prediction)
        return np.ravel(predictions)

    def getLabelFromTree(self, feature, tree):
        predictions = []
        for feat in list(feature.keys()):
            if feat in list(tree.keys()):
                # Read label in tree
                try: result = tree[feat][feature[feat]]
                except: result = 1
                # Iterate if it is in dict
                if isinstance(result, dict): return self.getLabelFromTree(feature, result)
                # prepare to return
                else: predictions.append(result)
        return predictions

class Perceptron:
    def __init__(self, w, b, lr):
        # Perceptron state here, input initial weight matrix
        # Feel free to add methods
        self.weight = w
        self.bias = b
        self.learningRate = lr

    def train(self, X, y, steps):
        # training logic here
        # input is array of features and labels
        features = X
        labels = y
        # set step limit & time limit (otherwise it will cost too much time)
        stepCounter = 0
        starttime = time.time()
        # train
        while (stepCounter <= steps and time.time() - starttime <= 0.1):
            # set label index
            labelIdx = 0
            # iterate all features
            for feature in features:
                # activate features
                prediction = self.activate(feature)
                # update weights and bias
                if prediction != labels[labelIdx]:
                    self.weight += self.learningRate * (labels[labelIdx] - prediction) * feature
                    self.bias += self.learningRate * (labels[labelIdx] - prediction)
                labelIdx += 1
            stepCounter += 1

    def predict(self, X):
        # Run model here
        # Return array of predictions where there is one prediction for each set of features
        features = X
        # generate and return predictions, each set of features has one prediction
        predictions = []
        for feature in features:
            predictions.append(self.activate(feature))
        return np.array(predictions)

    def activate(self, feature):
        # activation function
        activation = self.bias + np.dot(self.weight, feature)
        return 1 if activation > 0 else 0


class MLP:
    def __init__(self, w1, b1, w2, b2, lr):
        self.l1 = FCLayer(w1, b1, lr)
        self.a1 = Sigmoid()
        self.l2 = FCLayer(w2, b2, lr)
        self.a2 = Sigmoid()

    def MSE(self, prediction, target):
        return np.square(target - prediction).sum()

    def MSEGrad(self, prediction, target):
        return - 2.0 * (target - prediction)

    def shuffle(self, X, y):
        idxs = np.arange(y.size)
        np.random.shuffle(idxs)
        return X[idxs], y[idxs]

    def train(self, X, y, steps):
        for s in range(steps):
            i = s % y.size
            if (i == 0):
                X, y = self.shuffle(X, y)
            xi = np.expand_dims(X[i], axis=0)
            yi = np.expand_dims(y[i], axis=0)

            pred = self.l1.forward(xi)
            pred = self.a1.forward(pred)
            pred = self.l2.forward(pred)
            pred = self.a2.forward(pred)
            loss = self.MSE(pred, yi)

            grad = self.MSEGrad(pred, yi)
            grad = self.a2.backward(grad)
            grad = self.l2.backward(grad)
            grad = self.a1.backward(grad)
            grad = self.l1.backward(grad)

    def predict(self, X):
        pred = self.l1.forward(X)
        pred = self.a1.forward(pred)
        pred = self.l2.forward(pred)
        pred = self.a2.forward(pred)
        pred = np.round(pred)
        return np.ravel(pred)


class FCLayer:

    def __init__(self, w, b, lr):
        # initiate fully connected layer
        self.learningRate = lr
        self.weight = w  # Each column represents all the weights going into an output node
        self.bias = b

    def forward(self, input):
        # Write forward pass here
        # store x to be used in backward
        self.x = input
        # forward function
        return np.dot(input, self.weight) + self.bias

    def backward(self, gradients):
        # Write backward pass here
        # backward function
        w_ = np.dot(self.x.T, gradients)
        x_ = np.dot(gradients, self.weight.T)
        self.weight -= self.learningRate * w_
        self.bias -= self.learningRate * gradients
        return x_


class Sigmoid:

    def __init__(self):
        None

    def forward(self, input):
        # Write forward pass here
        # to avoid overflow, use -np.abs(input) instead
        self.sig = np.exp(np.fmin(input, 0)) / (1 + np.exp(-np.abs(input)))
        return self.sig

    def backward(self, gradients):
        # Write backward pass here
        # dL (cost function) / dz = (dL / da) * (da / dz)
        # = post-activation gradient * sig'(z)
        # return gradient of the cost with respect to z
        return gradients * (self.sig * (1 - self.sig))

