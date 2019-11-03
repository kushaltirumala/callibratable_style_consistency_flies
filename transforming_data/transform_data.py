import numpy as np
import scipy.io as sio
import os



DIR = "fruitflydata/Aggression/"

raw_tracking_data = []

for x in os.walk(DIR):
	folder_name = x[0]
	movie_name = folder_name.replace(DIR, "")
	tracking_file = folder_name + "/" + movie_name + "_track.mat"
	if os.path.exists(tracking_file):
		temp = sio.loadmat(tracking_file)
		data = temp["trk"][0]

		flag_frames = data[0][0]
		names = data[0][1]
		tracking_data = data[0][2]

		positions_only = tracking_data[:, :, :2]

		final = []

		flies_switched = flag_frames[0]
		switched = False

		for j in range(positions_only.shape[1]): 
			# assume flags_switched[j] == 1 means that from this frame on (inclusive) flies have switched
			if (flies_switched[j] == 1):
				switched = not switched

			if (switched):
				x_1 = positions_only[0][j][0] 
				y_1 = positions_only[0][j][1] 
				x_2 = positions_only[1][j][0] 
				y_2 = positions_only[1][j][1] 
			else:
				x_2 = positions_only[0][j][0] 
				y_2 = positions_only[0][j][1] 
				x_1 = positions_only[1][j][0] 
				y_1 = positions_only[1][j][1] 	

			add_arr = np.array([x_1, y_1, x_2, y_2]) 
			final.append(add_arr) 

		final = np.array(final) 

		raw_tracking_data.append(final)

raw_tracking_data = np.array(raw_tracking_data)
print(raw_tracking_data)
print(raw_tracking_data.shape)

np.savez("fruitflydata/compressed_final_data/aggresion",data=raw_tracking_data)
