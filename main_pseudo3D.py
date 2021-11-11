# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

from __future__ import print_function
import sys
import time
import numpy as np
from xinshuo_io import load_list_from_folder, fileparts, mkdir_if_missing
from AB3DMOT_libs.model_pseudo3D import AB3DMOT
import os
import matplotlib
matplotlib.use('Agg')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python main.py result_sha(e.g., pointrcnn_Car_test)')
        sys.exit(1)

    result_sha = sys.argv[1]
    save_root = './results'
    # det_id2str = {1:'Pedestrian', 2:'Car', 3:'Cyclist'}
    det_id2str = {0: 'person', 1: 'car',
                  2: 'rider', 3: 'motorcycle', 4: 'bicycle'}

    seq_file_list, num_seq = load_list_from_folder(
        os.path.join('data/filtered_3d', result_sha))
    total_time, total_frames = 0.0, 0
    save_dir = os.path.join(save_root, result_sha)
    mkdir_if_missing(save_dir)
    eval_dir = os.path.join(save_dir, 'data')
    mkdir_if_missing(eval_dir)
    seq_count = 0
    for seq_file in seq_file_list:
        _, seq_name, _ = fileparts(seq_file)
        eval_file = os.path.join(eval_dir, seq_name + '.txt')
        eval_file = open(eval_file, 'w')
        save_trk_dir = os.path.join(save_dir, 'trk_withid', seq_name)
        mkdir_if_missing(save_trk_dir)

        mot_tracker = AB3DMOT()
        # load detections, N x 14
        seq_dets = np.loadtxt(seq_file, delimiter=',')

        # if no detection in a sequence
        if len(seq_dets.shape) == 1:
            seq_dets = np.expand_dims(seq_dets, axis=0)
        if seq_dets.shape[1] == 0:
            eval_file.close()
            continue

        # loop over frame
        min_frame, max_frame = int(
            seq_dets[:, 0].min()), int(seq_dets[:, 0].max())
        for frame in range(min_frame, max_frame + 1):
            # logging
            print_str = 'processing %s: %d/%d, %d/%d   \r' % (
                seq_name, seq_count, num_seq, frame, max_frame)
            sys.stdout.write(print_str)
            sys.stdout.flush()
            save_trk_file = os.path.join(save_trk_dir, '%06d.txt' % frame)
            save_trk_file = open(save_trk_file, 'w')

            # get irrelevant information associated with an object, not used for associationg
            ori_array = seq_dets[seq_dets[:, 0] ==
                                 frame, -1].reshape((-1, 1))		# orientation
            # other information, e.g, 2D box, ...
            other_array = seq_dets[seq_dets[:, 0] == frame, 1:7]
            additional_info = np.concatenate((ori_array, other_array), axis=1)

            # x, y, z, stdx, stdy, stdz in camera coordinate follwing KITTI convention
            dets = seq_dets[seq_dets[:, 0] == frame, 7:13]
            dets_all = {'dets': dets, 'info': additional_info}

            # important
            start_time = time.time()
            trackers = mot_tracker.update(dets_all)
            cycle_time = time.time() - start_time
            total_time += cycle_time

            # saving results, loop over each tracklet
            for d in trackers:
                # x, y, z, stdx, stdy, stdzs in camera coordinate
                pseudo_3d = d[0:6]
                id_tmp = d[6]
                ori_tmp = d[7]
                type_tmp = det_id2str[d[8]]
                bbox2d_tmp_trk = d[9:13]
                conf_tmp = d[13]

                # save in detection format with track ID, can be used for dection evaluation and tracking visualization
                str_to_srite = '%s -1 -1 %f %f %f %f %f %f %f %f %f %f %f %f %d\n' % \
                    (type_tmp, ori_tmp, bbox2d_tmp_trk[0], bbox2d_tmp_trk[1], bbox2d_tmp_trk[2], bbox2d_tmp_trk[3], pseudo_3d[0],
                     pseudo_3d[1], pseudo_3d[2], pseudo_3d[3], pseudo_3d[4], pseudo_3d[5], conf_tmp, id_tmp)
                save_trk_file.write(str_to_srite)

                # save in tracking format, for 3D MOT evaluation
                str_to_srite = '%d %d %s 0 0 %f %f %f %f %f %f %f %f %f %f %f %f\n' % (frame, id_tmp,
                                                                                       type_tmp, ori_tmp, bbox2d_tmp_trk[0], bbox2d_tmp_trk[
                                                                                           1], bbox2d_tmp_trk[2], bbox2d_tmp_trk[3],
                                                                                       pseudo_3d[0], pseudo_3d[1], pseudo_3d[2], pseudo_3d[
                                                                                           3], pseudo_3d[4], pseudo_3d[5],
                                                                                       conf_tmp)
                eval_file.write(str_to_srite)

            total_frames += 1
            save_trk_file.close()
        seq_count += 1
        eval_file.close()
    print('Total Tracking took: %.3f for %d frames or %.1f FPS' %
          (total_time, total_frames, total_frames / total_time))
