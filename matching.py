"""SIFT and mutual BF matching, adapted from https://github.com/zaarAli/i-sfm.

The upstream MIT copyright and license are preserved in LICENSE.
"""
from dataclasses import dataclass
from typing import List, Tuple, Dict
import cv2
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

@dataclass
class MatchConfig:
    ratio_thresh: float = 0.8
    nfeatures: int = 0
    num_threads: int = 4
    mutual_check: bool = False


class FeatureMatcher:
    def __init__(self, n_imgs: int, cfg: MatchConfig):
        self.n_imgs = n_imgs
        self.cfg = cfg
        self.images: List[np.ndarray] = []
        self.kps: List[List[cv2.KeyPoint]] = []
        self.des: List[np.ndarray] = []
        self.matches: Dict[Tuple[int, int], List[cv2.DMatch]] = {}
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        self.sift = cv2.SIFT_create(nfeatures=self.cfg.nfeatures)

    def extract_features(self) -> None:
        logger.info("Extracting features...")
        for idx, img in enumerate(self.images):
            kp, des = self.sift.detectAndCompute(img, None)
            self.kps.append(kp)
            self.des.append(des)
            logger.debug(f"Image {idx}: {len(kp)} keypoints detected")
        logger.info(f"Extracted features from {len(self.images)} images.")

    def _match_pair(self, i: int, j: int) -> Tuple[Tuple[int, int], List[cv2.DMatch]]:
        if self.des[i] is None or self.des[j] is None or len(self.des[i]) == 0 or len(self.des[j]) < 2:
            return (i, j), []
        raw = self.matcher.knnMatch(self.des[i], self.des[j], k=2)
        good = [pair[0] for pair in raw if len(pair) == 2
                and pair[0].distance < self.cfg.ratio_thresh * pair[1].distance]
        if self.cfg.mutual_check:
            backward = self.matcher.knnMatch(self.des[j], self.des[i], k=2)
            reverse = {pair[0].queryIdx: pair[0].trainIdx for pair in backward
                       if len(pair) == 2 and pair[0].distance < self.cfg.ratio_thresh * pair[1].distance}
            good = [m for m in good if reverse.get(m.trainIdx) == m.queryIdx]
        return (i, j), good

    def match_pairs(self) -> None:
        logger.info("Matching feature pairs...")
        futures = []
        with ThreadPoolExecutor(max_workers=self.cfg.num_threads) as executor:
            for i in range(self.n_imgs):
                for j in range(i + 1, self.n_imgs):
                    futures.append(executor.submit(self._match_pair, i, j))

            for future in futures:
                key, matches = future.result()
                self.matches[key] = matches
        logger.info(f"Matched {len(self.matches)} image pairs.")
