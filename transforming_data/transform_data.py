import numpy as np
import scipy.io as sio
import os



DIR = "fruitflydata/Courtship/"

raw_tracking_data = []
raw_label_data = []

# Specify sequence length here
SEQUENCE_LENGTH = 100

# Specify sub action (from bouts array) within action here
action = "copulation"


def overlap_interval(a1, a2, b1, b2):
	return (b2 >= a1 and b2 <= a2) or (b1 >= a1 and b1 <= a2) or (b1 >= a1 and b2 <= a2) or (b1 <= a1 and b2 >= a2)

for x in os.walk(DIR):
	folder_name = x[0]
	movie_name = folder_name.replace(DIR, "")


	if (len(movie_name.split("/")) > 1):
		movie_name = movie_name.split("/")[1]
		tracking_file = folder_name + "/" + movie_name + "_track.mat"
		interval_file = folder_name + "/" + movie_name + "_actions.mat"
		print("TRACKING FILE " + tracking_file)

		# Now, that we've found a file i.e a movie to focus on
		if os.path.exists(tracking_file):
			temp = sio.loadmat(tracking_file)
			interval_file_data = sio.loadmat(interval_file)
			behs = interval_file_data["behs"][0]
			bouts = interval_file_data["bouts"]

			# Find the index of the specified sub action within behs array
			for index in range(len(behs)):
				if behs[index][0] == action:
					index_of_action = index

			# Find the arrays of intervals where the flies have committed said action
			fly_1_bouts = bouts[0][index_of_action]
			fly_2_bouts = bouts[1][index_of_action]

			fly_1_bouts_index = 0
			fly_2_bouts_index = 0

			index_of_action = None

			number_of_true_actions_fly_1 = 0
			number_of_true_actions_fly_2 = 0


			data = temp["trk"][0]

			flag_frames = data[0][0]
			names = data[0][1]
			tracking_data = data[0][2]

			positions_only = tracking_data[:, :, :2]

			final = []
			final_action_labels = []

			flies_switched = flag_frames[0]
			switched = False

			# Loop through sequences at SEQUENCE_LENGTH step
			for j in range(0, positions_only.shape[1], SEQUENCE_LENGTH): 
				label_fly_1 = False
				label_fly_2 = False

				# check if fly_1 should be labeled as copulation
				for fly_1_t in range(len(fly_1_bouts)):
					if overlap_interval(j, j + SEQUENCE_LENGTH, fly_1_bouts[fly_1_t][0], fly_1_bouts[fly_1_t][1]):
						label_fly_1 = True
						break
	
				# check if fly_2 should be labeled as copulation
				for fly_2_t in range(len(fly_2_bouts)):
					if overlap_interval(j, j + SEQUENCE_LENGTH, fly_2_bouts[fly_2_t][0], fly_2_bouts[fly_2_t][1]):
						label_fly_2 = True
						break

				# save the corresponding x,y positions
				x_1 = positions_only[0, j: j + SEQUENCE_LENGTH, 0] 
				y_1 = positions_only[0, j: j + SEQUENCE_LENGTH, 1] 
				x_2 = positions_only[1, j: j + SEQUENCE_LENGTH, 0] 
				y_2 = positions_only[1, j: j + SEQUENCE_LENGTH, 1] 


				# add the data to our running matrices 
				add_arr = np.array([x_1, y_1, x_2, y_2]) 
				add_arr = np.transpose(add_arr)
				add_arr_length = add_arr.shape[0]
				if (add_arr_length < SEQUENCE_LENGTH):
					last_row = add_arr[add_arr_length - 1]
					add_arr = np.vstack((add_arr, last_row))

				final.append(add_arr) 
				final_action_labels.append([label_fly_1, label_fly_2])


			if (len(raw_tracking_data) == 0):
				raw_tracking_data = np.array(final)
				raw_label_data = np.array(final_action_labels)
			else:
				raw_tracking_data = np.vstack((raw_tracking_data, final))
				raw_label_data = np.vstack((raw_label_data, final_action_labels))



np.savez("compressed_final_data/copulation_raw_tracking_data",data=raw_tracking_data)
np.savez("compressed_final_data/copulation_raw_label_data",data=raw_label_data)
