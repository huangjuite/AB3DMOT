# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import numpy as np
import math
from scipy.optimize import linear_sum_assignment
from AB3DMOT_libs.kalman_filter import KalmanPseudo3DTracker


def association_score(det, trk):
    # object location similarity
    s = 1
    k = 1
    ols = math.exp(
        -(math.pow(np.linalg.norm(det[:3]-trk[:3]), 2)) / (2*math.pow(s*k, 2)))

    return ols


def associate_detections_to_trackers(detections, trackers, score_threshold=0.2):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    detections:  N x 6
    trackers:    M x 6

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 6), dtype=int)

    score_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            # det: 6, trk: 6
            score_matrix[d, t] = association_score(det, trk)

    
    # hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-score_matrix)
    matched_indices = np.stack((row_ind, col_ind), axis=1)

    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if (score_matrix[m[0], m[1]] < score_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class AB3DMOT(object):			  # A baseline of 3D multi-object tracking
    # max age will preserve the bbox does not appear no more than n frames, interpolate the detection
    def __init__(self, max_age=2, min_hits=3):
        """
        Sets key parameters for SORT                
        """
        self.state_dim = 6
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0

    def update(self, dets_all):
        """
        Params:
          dets_all: dict
                dets - a numpy array of detections in the format [[x,y,z,stdx,stdy,stdz],...]
                info: a array of other info for each det
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        dets, info = dets_all['dets'], dets_all['info']         # dets: N x 6, float numpy array

        self.frame_count += 1

        # N x 6 , # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), self.state_dim))
        to_del = []
        ret = []

        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict().reshape((-1, 1))
            trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5]]
            if (np.any(np.isnan(pos))):
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks)

        # update matched trackers with assigned detections
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[
                    0], 0]     # a list of index
                trk.update(dets[d, :][0], info[d, :][0])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:        # a scalar of index
            trk = KalmanPseudo3DTracker(dets[i, :], info[i, :])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()      
            if ((trk.time_since_update < self.max_age) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits)):
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate(
                    (d, [trk.id + 1], trk.info)).reshape(1, -1))
            i -= 1

            # remove dead tracklet
            if (trk.time_since_update >= self.max_age):
                self.trackers.pop(i)
        if (len(ret) > 0):
            # x, y, z, stdx, stdy, stdz, ID, other info, confidence
            return np.concatenate(ret)
        return np.empty((0, 14))
