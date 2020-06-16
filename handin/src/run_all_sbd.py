import numpy as np
import framework
from sklearn.cluster import KMeans
import pickle
from os import path, mkdir
import cv2
import os

def simple_SBD(video_path, SB_threshold):

    # Make an instance of the video_feed class
    video = framework.video_feed(video_path, down_sample=True)
    count_num_frames = 0

    # loop for the length of the video or user interrupt
    while True:
        count_num_frames += 1
        video.next_frame()
        if (cv2.waitKey(1) & 0xFF == ord('q') or video.frame is None):
            break
        video.compute_error()
        video.detect_shot_boundary(SB_threshold)
        video.compute_hist('both')

        print(f'Frame: {count_num_frames} \r', flush=True, end='')

    video.close()
    sbd, hist = video.shot_boundary, np.array(video.get_hist())
    return sbd, hist, count_num_frames

def k_means_graphics(hist_data):
    model = 'k_means_rgbhsv4c.p'

    file_handler = open(model, "rb")
    k_means_model = pickle.load(file_handler)
    pred = k_means_model.predict(hist_data)
    return pred

def close_gap(labels):
    # The cluster for graphics
    graphics_cluster_num = 3
    # The size of gaps we will accept in the cluster prediction
    tolerance = 10
    count_down = tolerance
    last_known_frame = 0

    for i, label in enumerate(labels):
        # If the curent label index is belonging to the choosen cluster
        if label == graphics_cluster_num:
            # The first time we meet a cluster num
            if last_known_frame:
                frame_range = i-(i-last_known_frame)
                # if a label in the last <tolerance> number of frames is
                # belonging to the choosen cluster, we close the gap
                if graphics_cluster_num in labels[i-tolerance:i]:
                    labels[frame_range:i] = [graphics_cluster_num]*(i-last_known_frame)
                last_known_frame = i
            else:
                last_known_frame = i
    return labels

# SBD_data is an array with index number for each SBD
# clean_labels is an array with a cluster number for each frame
def cleanup(SBD_data, clean_labels, num_frames):
    cluster_num = 3
    last_frame = "Placeholder"
    inside_cluster = False # will be false while we are "in a cluster"
    # Clean the simple SBD data
    clean_SBD = np.zeros(num_frames)
    for shot in SBD_data:
      clean_SBD[shot] = 1

    final_pred = np.zeros(num_frames)

    for index, label in enumerate(clean_labels):
      if label == cluster_num and last_frame != cluster_num:
        # Then we are in the start of a cluster
        final_pred[index] = 2
        inside_cluster = True
      elif label != cluster_num and last_frame == cluster_num:
        # Then we are at the end of a cluster
        final_pred[index] = 2
        inside_cluster = False
      elif not inside_cluster and clean_SBD[index]:
        # If we are not inside a cluster, and there is a SBD
        final_pred[index] = 1

      # Update the last_frame to curent frame label
      last_frame = label

    return final_pred

def main():
    print('\nStart program')
    input_dir  = 'data'
    output_dir = 'SBD_output'

    # Check if k_means model is present
    if not path.exists('pickle_jar/k_means_rgbhsv4c.p'):
        print('k_means model is missing \nEnd')
        return

    # Create a directory for output files
    if not path.exists(output_dir):
        mkdir(output_dir)

    # loop over all files in input_dir
    working_dir = os.listdir(input_dir)
    for video_file in working_dir:

        # Only work on .ts filetype
        if not video_file[-3:] == '.ts':
            print(f'Skipping:   {video_file}')
            continue

        # remve file type, i.e. the last 3 characters '.ts'
        file_name = video_file[:-3]
        output_path = f'{output_dir}/{file_name}.npy'
        input_path  = f'{input_dir}/{video_file}'

        if path.exists(output_path):
            print(f'Skipping:   {file_name}')
            continue
        else:
            print(f'Processing: {file_name}')
            shot_boundary_index, histogram, num_frames = simple_SBD(input_path, 2000)
            label_frames = k_means_graphics(histogram)
            labels = close_gap(label_frames)
            clean_data = cleanup(shot_boundary_index, labels, num_frames)
            np.save(output_path, clean_data)

main()
