import importlib
from tqdm import tqdm
from road_anomaly_benchmark.evaluation import Evaluation

methods = {
    "Mahalanobis": "O5AMylqoSaq5Wjc",
    "Max_softmax": "WVFTc4ka37xASZV",
    "ODIN": "WVFTc4ka37xASZV",
    "Entropy_max": "kCgnr0LQuTbrArA",
}


def main():
    for methodname, modelid in methods.items():
        print("\nmethod_name:", methodname)
        method_object = getattr(importlib.import_module("methods.baselines"), methodname)
        method = method_object(modelid)

        ev = Evaluation(
            method_name=methodname,
            # dataset_name = 'ObstacleTrack-all',
            # dataset_name='AnomalyTrack-test',
            dataset_name='ObstacleTrack-validation',
        )

        for frame in tqdm(ev.get_frames()):
            result = method.anomaly_score(frame.image)
            ev.save_output(frame, result)
        ev.wait_to_finish_saving()


if __name__ == '__main__':
    main()
