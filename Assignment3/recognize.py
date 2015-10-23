import cv2
import numpy as np
from scipy.spatial import distance
import math
from matplotlib import pyplot as plt

class ImageMatcher():
    def __init__(self, query, db):
        self.db = cv2.cvtColor(cv2.imread(db), cv2.COLOR_BGR2GRAY)
        self.query = cv2.cvtColor(cv2.imread(query), cv2.COLOR_BGR2GRAY)

        # init sift
        self.sift = cv2.SIFT()

    def compute_keypoints(self):
        # detect and compute SIFT descriptors
        self.db_kp = self.sift.detectAndCompute(self.db, None)
        self.query_kp = self.sift.detectAndCompute(self.query, None)

    def test_data(self):
        base_keypoints = np.array([
            [1, 3, 5],
            [2, 6, 7],
            [3, 4, 4],
        ])
        query_keypoints = np.array([
            [1, 3, 5],
            [2, 6, 7],
            [3, 4, 4],
        ])

    def get_euclidian(self):
        base_keypoints = self.db_kp[1]
        query_keypoints = self.query_kp[1]
        (n_base, dim_base) = base_keypoints.shape
        (n_query, dim_query) = query_keypoints.shape
        print "Base shape is: ", n_base, "x", dim_base
        print "Query shape is: ", n_query, "x", dim_query

        a2 = np.dot(base_keypoints ** 2, np.ones((dim_query, n_query))) 
        b2 = np.dot(np.ones((n_base, dim_query)), query_keypoints.T ** 2)
        ab = np.dot(base_keypoints, query_keypoints.T)
        
        kp_array = a2 + b2 - 2 * ab
        kp_array = np.sqrt(kp_array)

        (n_val, _) = kp_array.shape
        best_match = np.amin(kp_array, axis=1)
        best_match_idx = np.argmin(kp_array, axis=1)

        print "Best match index of dimension: ", best_match_idx.shape

        # set best matches to inf to get second best matches
        kp_array[range(n_val), best_match_idx] = np.inf
        second_best = np.amin(kp_array, axis=1)

        ratio = best_match / second_best
        best_ratio = np.amin(ratio)
        retained_indicies = np.where(ratio < 0.75)[0]

        retained_tuples = sorted([(x, ratio[x]) for x in retained_indicies], key = lambda x: x[1])

        print "Best ratio was: ", best_ratio
        print "List of best matches: \n", retained_tuples
        #im = cv2.drawKeypoints(self.db[db_idx], good_points)
        #cv2.imwrite('sift_keypoints.jpg',im)


    def match_one_pair(self):
        print "Ecl. distance is ", self.get_euclidian(0)


if __name__ == '__main__':
    im = ImageMatcher("05.jpg", 'toy.jpg')
    im.compute_keypoints()
    im.get_euclidian()
