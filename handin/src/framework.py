import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans
from os import path, mkdir
import pickle

# Mean Squared Error
def MSE(img1, img2):
	err = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
	return err / float(img1.shape[0] * img1.shape[1])

# Kullback Leibler Divergence
def KLD(P, Q):
    return np.sum(P * np.log( np.divide(P, Q) ))

# The function will maintain a queue structure
# param - <cont> is a list we want to maintain
# param - <error> is a new value we want to store in the list
# param - <num> is the desired length of the list
# return - the updated list witht <error> as the last item
def queue_error(cont, error, num):
    length = len(cont)
    if not length == num:
        cont.append(error)
        return cont
    else:
        cont[0] = error
        cont += [cont.pop(0)]
        return cont

# Compute the confusion matrix
# param - <pred> a list with the predictive values
# param - <gt> a list with the ground truth values
# param - <range> - a integer witht the accepted range around the from number
# return - a confusion matrix as [TP, FP, FN, TN]
def confusion_matrix(pred, gt, range):
    TP = 0
    FN = 0

    # create a list of lists, where each list contains
    # the frame number and <range> numbers on each side
    size = range * 2 + 1
    gt_range_list = np.array([np.linspace(frame-range, frame+range, size) for frame in gt]).flatten()
    pred_range_list = np.array([np.linspace(frame-range, frame+range, size) for frame in pred]).flatten()

    for frame in pred:
        if frame in gt_range_list:
            TP += 1

    for frame in gt:
        if frame not in pred_range_list:
            FN += 1

    FP = len(pred) - TP
    TN = len(gt) - FN

    return [TP, FP, FN, TN]

# Display the video stream
def show_video_feed():
    pass

# Display the error as as a plot on screen
def show_error_graph():
    pass

class video_feed:

    def __init__(self, path_to_video, path_to_gt='NO_GT', down_sample=False, mse_range=3):
        self.path = path_to_video
        self.cap = cv2.VideoCapture(self.path)
        self.down_sample = down_sample
        _, frame = self.cap.read()
        if down_sample:
	        self.scale_percent = 50 # percent of original size
	        width = int(frame.shape[1] * self.scale_percent / 100)
	        height = int(frame.shape[0] * self.scale_percent / 100)
	        dim = (width, height)
	        # resize image
	        resized = cv2.resize(frame, dim, interpolation = cv2.INTER_NEAREST)

        self.error = 0
        self.error_list = []
        self.shot_boundary = []
        self.count = 0
        self.temp = 0
        self.range = mse_range

        self.frame_count = 0
        self.frame = resized if down_sample else frame
        self.old_frame = resized if down_sample else frame

        self.accuracy = None
        self.precision = None
        self.recall = None
        self.specificity = None
        self.f1 = None

        self.histogram = []
        self.gradient = []
        self.model = None

        if path_to_gt != 'NO_GT':
	        with open(path_to_gt, 'rb') as file:
	            lines = file.readlines()
	            self.ground_truth = np.loadtxt(lines, dtype='int')

    def compute_error(self):
        self.error = error = MSE(self.old_frame, self.frame)
        self.error_list = queue_error(self.error_list, self.error, self.range)

    def detect_shot_boundary(self, threshold):
        if np.std(self.error_list) > threshold:
			# Check if a cut has just been detected
            if abs(self.count - self.temp) > self.range:
                self.temp = self.count
                self.shot_boundary.append(self.count)

    def next_frame(self):
        _, frame = self.cap.read()
        if self.down_sample and frame is not None:
	        width = int(frame.shape[1] * self.scale_percent / 100)
	        height = int(frame.shape[0] * self.scale_percent / 100)
	        dim = (width, height)
	        # resize image
	        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_NEAREST)

        self.old_frame = self.frame
        self.frame = frame
        self.count += 1

    def show_video_feed(self):
        cv2.imshow('Video Capture: {}'.format(self.path), self.frame)

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def compute_hist(self, type='RGB'):
        if type == 'HSV':
			# Hue range: [0,179], Saturation range: [0,255], Value range: [0,255]
            hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
            hist_hue = cv2.calcHist([hsv], [0], None, [256], [0, 255])
            hist_sat = cv2.calcHist([hsv], [1], None, [256], [0, 255])
            hist_val = cv2.calcHist([hsv], [2], None, [256], [0, 255])
            hist = np.append(np.append(hist_hue, hist_sat), hist_val)
        elif type == 'HSV_RGB':
            hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0],  None, [256], [0, 256])
            hist = np.append(hist, cv2.calcHist([self.frame], [0],  None, [256], [0, 256]))
        elif type == 'both':
            hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0,1],  None, [180,256], [0, 180, 0, 256])
            hist = np.append(hist, cv2.calcHist([self.frame], [0],  None, [256], [0, 256]))
        else:
            hist = cv2.calcHist([self.frame], [0],  None, [256], [0, 256])
        self.histogram.append(hist)

    def get_hist(self):
        return self.histogram

    def compute_gradient(self):
        self.gradient.append(cv2.Laplacian(self.frame, cv2.CV_64F))

    def get_gradient(self):
        return self.gradient

    def k_means(self, data, num):
        k_means = KMeans(n_clusters=num)
        self.model = k_means.fit(data)
        return self.model

    def get_model(self):
        return self.model

    def save_model(self, name, picklejar='picklejar'):
        # Create picklejar if not present
        if not path.exists(f'./{picklejar}'):
            mkdir(f'./{picklejar}')

        model = self.model
        # Write to file
        file_path = f'./{picklejar}/{name}.p'
        file_handle = open(file_path, 'wb')
        pickle.dump(model, file_handle)
        file_handle.close()
        print(f'Model saved to {file_path}')

    def load_model(name, picklejar='picklejar'):
        # Read from file
        file_path = f'{picklejar}/{name}.p'
        # Check if files exists
        if path.isfile(file_path):
            file_handle = open(file_path, 'rb')
            data = pickle.load(file_handle)
            self.load_data = data
            file_handle.close()
        else:
            data = None
        return data

    def compute_performance(self):
        pred = self.shot_boundary
        gt = self.ground_truth
        range = self.range
        TP, FP, FN, TN = confusion_matrix(pred, gt, range)
        if TP or FP or FN or TN:
          self.accuracy = (TP + TN) / (TP + FP + FN + TN)
        else:
          self.accuracy = 0
        if TP or FP:
          self.precision = TP / (TP + FP)
        else:
          self.precision = 0
        if TP or FN:
          self.recall = TP / (TP + FN)
        else:
          self.recall = 0
        if TN or FP:
          self.specificity = TN / (TN + FP)
        else:
          self.specificity = 0
        if self.precision or self.recall:
          self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall)
        else:
          self.f1 = 0

    def show_performance(self, num=4):
        print('')
        print(' -- Performance --')
        print('Accuracy:    {}'.format( round(self.accuracy, num) ))
        print('Precision:   {}'.format( round(self.precision, num) ))
        print('Recall:      {}'.format( round(self.recall, num) ))
        print('Specificity: {}'.format( round(self.specificity, num) ))
        print('F1 score:    {}'.format( round(self.f1, num) ))
        print('')
