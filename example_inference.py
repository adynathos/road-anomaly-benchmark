
import numpy as np
from tqdm import tqdm
import cv2 as cv

def method_dummy(image, **_):
	
	image_hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV_FULL)

	anomaly_p = image_hsv[:, :, 1].astype(np.float) * (1./255.)

	return anomaly_p

def main():
	from road_anomaly_benchmark.evaluation import Evaluation

	ev = Evaluation(method_name = 'Dummy', dataset_name = 'RoadObstacleTrack-test')

	for frame in tqdm(ev.get_dataset()):
		result = method_dummy(frame.image)
		ev.save_output(frame, result)

	ev.wait_to_finish_saving()

if __name__ == '__main__':
	main()
