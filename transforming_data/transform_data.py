import numpy as np
import scipy.io as sio
import os



DIR = "fruitflydata/Courtship/"

raw_tracking_data = []
raw_label_data = []

SEQUENCE_LENGTH = 100
action = "copulation"


for x in os.walk(DIR):
	folder_name = x[0]
	movie_name = folder_name.replace(DIR, "")
	tracking_file = folder_name + "/" + movie_name + "_track.mat"
	interval_file = folder_name + "/" + movie_name + "_actions.mat"
	if os.path.exists(tracking_file):
		temp = sio.loadmat(tracking_file)
		interval_file_data = sio.loadmat(interval_file)
		behs = interval_file_data["behs"][0]
		bouts = interval_file_data["bouts"]
		for index in range(len(behs)):
			if behs[index][0] == action:
				index_of_action = index

		fly_1_bouts = bouts[0][index_of_action]
		fly_2_bouts = bouts[1][index_of_action]

		fly_1_bouts = np.sort(fly_1_bouts, axis=0)
		fly_2_bouts = np.sort(fly_2_bouts, axis=0)
		fly_1_bouts_index = 0
		fly_2_bouts_index = 0

		index_of_action = None




		data = temp["trk"][0]

		flag_frames = data[0][0]
		names = data[0][1]
		tracking_data = data[0][2]

		positions_only = tracking_data[:, :, :2]

		final = []
		final_action_labels = []

		flies_switched = flag_frames[0]
		switched = False

		for j in range(0, positions_only.shape[1], SEQUENCE_LENGTH): 
			# assume flags_switched[j] == 1 means that from this frame on (inclusive) flies have switched
			label_fly_1 = False
			label_fly_2 = False
			if fly_1_bouts_index < len(fly_1_bouts) and j >= fly_1_bouts[fly_1_bouts_index][0]:
				label_fly_1 = True
				fly_1_bouts_index += 1

			if fly_2_bouts_index < len(fly_2_bouts) and j >= fly_2_bouts[fly_2_bouts_index][0]:
				label_fly_2 = True
				fly_2_bouts_index += 1

			# if (flies_switched[j] == 1):
			# 	switched = not switched

			# if (switched):
			x_1 = positions_only[0, j: j + SEQUENCE_LENGTH, 0] 
			y_1 = positions_only[0, j: j + SEQUENCE_LENGTH, 1] 
			x_2 = positions_only[1, j: j + SEQUENCE_LENGTH, 0] 
			y_2 = positions_only[1, j: j + SEQUENCE_LENGTH, 1] 
			# else:
				# x_2 = positions_only[0][j: j + SEQUENCE_LENGTH][0] 
				# y_2 = positions_only[0][j: j + SEQUENCE_LENGTH][1] 
				# x_1 = positions_only[1][j: j + SEQUENCE_LENGTH][0] 
				# y_1 = positions_only[1][j: j + SEQUENCE_LENGTH][1] 	

			add_arr = np.array([x_1, y_1, x_2, y_2]) 
			add_arr = np.transpose(add_arr)
			add_arr_length = add_arr.shape[0]
			if (add_arr_length < SEQUENCE_LENGTH):
				last_row = add_arr[add_arr_length - 1]
				add_arr = np.vstack((add_arr, last_row))

			print(add_arr.shape)
			final.append(add_arr) 
			final_action_labels.append([label_fly_1, label_fly_2])



		final = np.array(final) 
		final_actions_labels = np.array(final_action_labels)

		raw_tracking_data.append(final)
		raw_label_data.append(final_action_labels)

raw_tracking_data = np.array(raw_tracking_data)
raw_label_data = np.array(raw_label_data)
final_data = np.array([raw_tracking_data, raw_label_data])

np.savez("fruitflydata/compressed_final_data/courtship",data=final_data)
