import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset
from image_recog import ClipEnsembleDetector, detect_group

YOLO_WEIGHTS = os.path.join("runs", "captcha_cls", "weights", "best.pt")


class HybridSolver:
    def __init__(self, clip=None):
        self.clip = clip or ClipEnsembleDetector()
        self.yolo = None
        self.yolo_loaded = False
        self._lock = threading.Lock()

    def _get_yolo(self):
        with self._lock:
            if not self.yolo_loaded:
                self.yolo_loaded = True
                try:
                    if os.path.exists(YOLO_WEIGHTS):
                        from train import CaptchaYolo

                        self.yolo = CaptchaYolo()
                        print("hybrid: yolo model loaded")
                except Exception as e:
                    print(f"hybrid: yolo load failed ({e})")
                    self.yolo = None
        return self.yolo

    def solve(self, image_paths, target):
        hashes = dataset.store_images(image_paths)
        verdicts = dataset.lookup_grid(hashes, target)
        unknown = [i for i, v in enumerate(verdicts) if v is None]

        used = "db"
        if unknown:
            yolo = self._get_yolo()
            if yolo and target in dataset.TARGETS:
                preds = yolo.predict_batch([image_paths[i] for i in unknown])
                for i, (cls, conf) in zip(unknown, preds):
                    verdicts[i] = cls == target and conf >= 0.5
                used = f"db+yolo ({len(unknown)} unknown)"
            else:
                results = detect_group(image_paths, target, self.clip)
                for i in unknown:
                    verdicts[i] = results[i]["has_target"]
                used = f"db+clip ({len(unknown)} unknown)"

        solution = [i for i, v in enumerate(verdicts) if v]
        if not solution and unknown:
            results = detect_group(image_paths, target, self.clip)
            solution = [max(range(len(results)), key=lambda i: results[i]["score"])]
            used += " +fallback"
        return solution, hashes, used
