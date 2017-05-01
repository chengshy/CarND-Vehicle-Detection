import numpy as np
import cv2
from feature_extraction import *
import pickle
from collections import deque
from scipy.ndimage.measurements import label

class CarDetector:
    
    def __init__(self, svc_model):
        with open(svc_model, 'rb') as model_file:
            model_data = pickle.load(model_file)
            self.colorspace = model_data["colorspace"]
            self.orient = model_data["orient"]
            self.pix_per_cell = model_data["pix_per_cell"]
            self.cell_per_block = model_data["cell_per_block"]
            self.hog_channel = model_data["hog_channel"]
            self.spatial = model_data["spatial"]
            self.histbin = model_data["histbin"]
            self.svc = model_data["svc"]
            self.X_scaler = model_data["X_scaler"]

        self.frame_count = 0
        self.full_search_interval = 7
            
        self.x_start_stop = [0, 1280]
        self.y_start_stop_size_list = [[380, 500, 96], [400, 600, 128], [500, 660, 160]]

        self.x_start_stop_scaled_list = []
        self.y_start_stop_scaled_list = []

        self.heatmap_queue = deque(maxlen = 7)
        self.heat_threshold = 7.5
        self.heatmap_thresholded = None

        self.window_scales = []
        self.window_scaled_croped_lists = []

        self.cell_overlap = 2 # 2 cells hog

        for (y_start, y_stop, window_size) in self.y_start_stop_size_list:
            scale = window_size / 64.
            self.window_scales.append(scale)

            y_start_scaled = int(y_start / scale)
            y_stop_scaled = int(y_stop / scale)
            x_start_scaled = int(self.x_start_stop[0]/ scale)
            x_stop_scaled = int(self.x_start_stop[1]/ scale)

            x_start_stop_scaled = [x_start_scaled, x_stop_scaled]
            y_start_stop_scaled = [y_start_scaled, y_stop_scaled]

            self.x_start_stop_scaled_list.append(x_start_stop_scaled)
            self.y_start_stop_scaled_list.append(y_start_stop_scaled)

    def find_cars(self, image):
        ''' The image input is RGB'''
        output_image = np.copy(image)

        image = image.astype(np.float32) / 255
        if self.frame_count % self.full_search_interval == 0:
            mask = np.ones_like(image[:,:,0])
        else:
            mask = np.sum(self.heatmap_queue, axis = 0)
            mask[mask > 0] = 1
            mask = cv2.dilate(mask, np.ones((75,75)), iterations = 1)

        self.frame_count += 1

        box_list = []

        for i in range(len(self.y_start_stop_size_list)):
            scale = self.window_scales[i]
            y_start = self.y_start_stop_size_list[i][0]
            y_stop = self.y_start_stop_size_list[i][1]

            y_start_scaled = self.y_start_stop_scaled_list[i][0]
            y_stop_scaled = self.y_start_stop_scaled_list[i][1]

            x_start_scaled = self.x_start_stop_scaled_list[i][0]
            x_stop_scaled = self.x_start_stop_scaled_list[i][1]

            # Crop and resize the image
            image_crop = image[y_start:y_stop, self.x_start_stop[0]:self.x_start_stop[1]]
            image_crop_resize = cv2.resize(image_crop, (x_stop_scaled - x_start_scaled, y_stop_scaled - y_start_scaled))
            image_crop_resize = cv2.cvtColor(image_crop_resize, cv2.COLOR_RGB2YCrCb)

            mask_crop = mask[y_start:y_stop, self.x_start_stop[0]:self.x_start_stop[1]]
            mask_crop_resize = cv2.resize(mask_crop, (x_stop_scaled - x_start_scaled, y_stop_scaled - y_start_scaled))

            # Get hog feature
            ch1 = image_crop_resize[:, :, 0]
            ch2 = image_crop_resize[:, :, 1]
            ch3 = image_crop_resize[:, :, 2]

            # Compute individual channel HOG features for the entire image
            hog1 = get_hog_features(ch1, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
            hog2 = get_hog_features(ch2, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
            hog3 = get_hog_features(ch3, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)

            # Define blocks and steps as above
            nxblocks = (ch1.shape[1] // self.pix_per_cell) - self.cell_per_block + 1
            nyblocks = (ch1.shape[0] // self.pix_per_cell) - self.cell_per_block + 1 
            nfeat_per_block = self.orient * self.cell_per_block**2
    
            # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
            window = 64
            nblocks_per_window = (window // self.pix_per_cell) - self.cell_per_block + 1
            cells_per_step = self.cell_overlap  # Instead of overlap, define how many cells to step
            nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
            nysteps = (nyblocks - nblocks_per_window) // cells_per_step

            for xb in range(nxsteps + 1):
                for yb in range(nysteps + 1):
                    ypos = yb * cells_per_step
                    xpos = xb * cells_per_step

                    xleft = xpos * self.pix_per_cell
                    ytop = ypos * self.pix_per_cell

                    submask = mask_crop_resize[ytop:ytop + window, xleft:xleft + window]

                    if(np.sum(submask) == 0):
                        continue

                    # Extract HOG for this patch
                    hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                    hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                    hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                    hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                    # Extract the image patch
                    subimg = image_crop_resize[ytop:ytop + window, xleft:xleft + window]

                    # Get color features
                    spatial_features = bin_spatial(subimg, size=(self.spatial, self.spatial))
                    hist_features = color_hist(subimg, nbins=self.histbin)

                    # Scale features and make a prediction
                    test_features = self.X_scaler.transform(
                        np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
                    test_prediction = self.svc.predict(test_features)

                    if test_prediction == 1:
                        xbox_left = np.int(xleft*scale) + self.x_start_stop[0]
                        ytop_draw = np.int(ytop*scale) + y_start
                        win_draw = np.int(window*scale)
                        box_list.append(((xbox_left, ytop_draw), (xbox_left + win_draw, ytop_draw + win_draw)))

        # Genrate heatmap
        heatmap = np.zeros_like(image[:,:,0], dtype = np.float64)

        for box in box_list:
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        self.heatmap_queue.append(heatmap)
        heatmap = np.sum(self.heatmap_queue, axis = 0)
        heatmap[heatmap < self.heat_threshold] = 0
        self.heatmap_thresholded = heatmap

        # Label and draw
        labels = label(heatmap)

        for label_id in range(1, labels[1] + 1):
            label_position = (labels[0] == label_id).nonzero()
            nonzerox = label_position[1]
            nonzeroy = label_position[0]
            x_min = np.min(nonzerox)
            y_min = np.min(nonzeroy)
            x_max = np.max(nonzerox)
            y_max = np.max(nonzeroy)
            cv2.rectangle(output_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 6)

        return output_image
