import cv2
import numpy as np
import math
#from matplotlib import pyplot as plt

class ImageMatcher():
    def __init__(self, query, db):
        print "Loading images"
        self.db = cv2.cvtColor(cv2.imread(db), cv2.COLOR_BGR2GRAY)
        self.query = cv2.cvtColor(cv2.imread(query), cv2.COLOR_BGR2GRAY)

        # init sift
        self.sift = cv2.SIFT()

    def compute_keypoints(self):
        print "Computing keypoints"
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

        # remove this should have no effect
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
        
        # indicies of base keypoints 
        retained_indicies = np.where(ratio < 0.75)[0]

        # indicies of query keypoints 
        matched_indicies = best_match_idx[retained_indicies]

        # extract query and base keypoints from keypoint indicies
        query_kp = map(self.query_kp[0].__getitem__, matched_indicies)
        base_kp = map(self.db_kp[0].__getitem__, retained_indicies)

        # this is the lib call to compute homography
        src = np.float32([ m.pt for m in query_kp]).reshape(-1,1,2)
        dst = np.float32([ m.pt for m in base_kp]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

        print "M is ", M
        self.get_perc_inliners(base_kp, query_kp, M)
        inp = cv2.imread('toy.jpg')
        transformed = cv2.warpPerspective(inp, M, (10000, 10000))
        cv2.imwrite('transformed.jpg', transformed)

        #af = cv2.getPerspectiveTransform(src, dst)
        #print "PERS is ", af
        H = self.compute_homography(base_kp[:4], query_kp[:4])
        
        print "Sample keypoint location is ", query_kp[0].pt
        print "Matched keypoints ", len(query_kp), ' vs ', len (base_kp)
        print "Query keypoints dim is: ", len(query_kp)

        retained_tuples = sorted([(x, ratio[x]) for x in retained_indicies], key = lambda x: x[1])
        print "Best ratio was: ", best_ratio

        im = cv2.drawKeypoints(self.query, query_kp)
        cv2.imwrite('sift_keypoints.jpg',im)

    def RANSAC(self, effort=1):
        # first pick random 4 points

        # then compute the homography

        # check the # of inliners

        # repeat

        # return the best homography
        return 

    def _compute_component(self, db_kp, query_kp):
        """
        Construct H array (2, 9)
        
        [x y 1 0 0 0 -x'x -x'y -x']
        [0 0 0 x y 1 -y'x -y'y -y']
        """
        kp_arr = np.zeros(shape=(2, 9))
        (y, x) = query_kp.pt
        (y_p, x_p) = db_kp.pt

        kp_arr[0][0] = x
        kp_arr[0][1] = y
        kp_arr[0][2] = 1
        kp_arr[0][6] = -1.0 * x * x_p
        kp_arr[0][7] = -1.0 * x_p * y
        kp_arr[0][8] = -1.0 * x_p

        kp_arr[1][3] = x
        kp_arr[1][4] = y
        kp_arr[1][5] = 1
        kp_arr[1][6] = -1.0 * x * y_p
        kp_arr[1][7] = -1.0 * y * y_p
        kp_arr[1][8] = -1.0 * y_p
        return kp_arr

    def compute_homography(self, db_kp, query_kp):
        """
        Compute the linear homogeneous homography
        from query_kp to db_kp

        return value:
            a 3x3 float-valued homographic 
            transformation matrix

            NOTE: this is different from 
            cv2.getPerspectiveTransform()
            where linear non-homogenous homography
            is computed
        """
        if len(db_kp) != 4 or len(query_kp) != 4:
            ValueError("Must provide 4 points of {db, query} to compute homography!")
        H = None
        for i in xrange(len(db_kp)):
            a = self._compute_component(db_kp[i], query_kp[i])
            if H is None:
                H = a
            else:
                H = np.append(H, a, axis=0)

        # assume linear homogenous solution
        H = np.dot(H.T, H)
        (val, vec) = np.linalg.eig(H)
        M = vec[np.argmin(val)].reshape(3,3)
        print "M2 is ", M
        perc = self.get_perc_inliners(db_kp, query_kp, M)
        print "Perc. inliers is ", perc
        return M

    def get_pt(self, kp):
        return [item.pt for item in kp]
    
    def transform(self, src, M):
        """
        Compute the coordinates of transformed src
        under homography M
        """
        (y, x) = src
        base = M[2][0]*x + M[2][1]*y + M[2][2]
        x_p = (M[0][0]*x + M[0][1]*y + M[0][2])/base
        y_p = (M[1][0]*x + M[1][1]*y + M[1][2])/base
        return (y_p, x_p)

    def get_perc_inliners(self, db_kp, query_kp, M, thresh=300):
        """
        Get the percentage of inliers from query_kp to db_kp
        under homographic transformation M. Inliers are points 
        within thresh distance (euclidian)
        """
        # first extract coordinates
        db_coords = self.get_pt(db_kp)
        query_coords = self.get_pt(query_kp)

        # transform query with given M
        transformed_query = [self.transform(coord, M) for coord in query_coords]

        # get % of inliers
        total = len(transformed_query)
        inliner = 0
        
        for i in xrange(len(transformed_query)):
            euclid = (db_coords[i][0] - transformed_query[i][0])**2 + (db_coords[i][1] - transformed_query[i][1])**2 
            euclid = math.sqrt(euclid)
            if euclid <= thresh:
                print "Transformed: ", transformed_query[i], " actual ", db_coords[i], " eucl. dist. ", euclid
                inliner += 1
        return 1.0 * inliner / total

if __name__ == '__main__':
    im = ImageMatcher("09.jpg", 'toy.jpg')
    im.compute_keypoints()
    im.get_euclidian()

