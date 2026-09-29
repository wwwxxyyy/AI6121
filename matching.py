from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import cv2
import numpy as np
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from matplotlib.cm import get_cmap

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MatchConfig:
    dataset_path: str
    img_pattern: str            # e.g. '{idx:02d}.JPG'
    ratio_thresh: float = 0.8
    ransac_thresh: float = 4.0
    min_inliers: int = 15
    use_flann: bool = True      # Use FLANN for faster matching
    nfeatures: int = 0          # Add this line
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
        self.adjacency: np.ndarray = np.zeros((n_imgs, n_imgs), dtype=np.uint8)

        # Choose matcher
        if cfg.use_flann:
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=75)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        # Feature detector
        self.sift = cv2.SIFT_create(nfeatures=self.cfg.nfeatures)  # No restriction on features

    def load_images(self) -> None:
        logger.info("Loading images...")
        for i in range(self.n_imgs):
            fname = f"{self.cfg.dataset_path}/{self.cfg.img_pattern.format(idx=i)}"
            img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Cannot load image {fname}")
            self.images.append(img)
        logger.info(f"Loaded {len(self.images)} images.")

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

    def filter_geometric(self) -> None:
        logger.info("Filtering matches with geometric constraints...")
        filtered = {}
        for (i, j), mlist in self.matches.items():
            if len(mlist) < self.cfg.min_inliers:
                continue

            pts_i = np.float32([self.kps[i][m.queryIdx].pt for m in mlist])
            pts_j = np.float32([self.kps[j][m.trainIdx].pt for m in mlist])

            F, mask = cv2.findFundamentalMat(pts_i, pts_j, cv2.FM_RANSAC, self.cfg.ransac_thresh)
            if mask is None or mask.sum() < self.cfg.min_inliers:
                continue

            inliers = [m for m, ok in zip(mlist, mask.ravel()) if ok]
            if len(inliers) >= self.cfg.min_inliers:
                filtered[(i, j)] = inliers

        self.matches = filtered
        logger.info(f"{len(self.matches)} pairs left after geometric verification.")

    def build_adjacency(self) -> List[Tuple[int, int]]:
        logger.info("Building adjacency graph...")
        self.adjacency.fill(0)
        pairs = []
        for (i, j), mlist in self.matches.items():
            if len(mlist) >= self.cfg.min_inliers:
                self.adjacency[i, j] = self.adjacency[j, i] = 1
                pairs.append((i, j))
        logger.info(f"Adjacency: {len(pairs)} connected pairs.")
        return pairs

    def run(self) -> np.ndarray:
        self.load_images()
        self.extract_features()
        self.match_pairs()
        self.filter_geometric()
        return self.build_adjacency()

    def plot_keypoints(self, idx: int = 0) -> None:
        """Plot the detected keypoints on an image."""
        img = self.images[idx]
        kp = self.kps[idx]
        img_kp = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.figure(figsize=(10, 6))
        plt.imshow(img_kp, cmap='gray')
        plt.title(f"Detected Keypoints - Image {idx}")
        plt.axis('off')
        plt.show()

    def plot_best_match(self) -> None:
        """Plot the best matched image pair (with most matches)."""
        if not self.matches:
            logger.warning("No matches available to plot.")
            return

        # Find pair with maximum number of matches
        best_pair = max(self.matches.items(), key=lambda x: len(x[1]))[0]
        i, j = best_pair
        matches = self.matches[(i, j)]

        logger.info(f"Best pair: ({i}, {j}) with {len(matches)} matches")

        # Draw matches
        img1 = self.images[i]
        img2 = self.images[j]
        kp1 = self.kps[i]
        kp2 = self.kps[j]

        match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        plt.figure(figsize=(16, 8))
        plt.imshow(match_img)
        plt.title(f"Matches between Image {i} and Image {j}")
        plt.axis('off')
        plt.show()

    def plot_feature_histogram(self) -> None:
        """Plot histogram of number of features per image."""
        counts = [len(kp) for kp in self.kps]
        plt.figure(figsize=(10, 6))
        cmap = get_cmap('tab10')
        bars = plt.bar(range(len(counts)), counts, color=cmap(0), alpha=0.7)
        plt.xlabel("Image Index")
        plt.ylabel("Number of Features")
        plt.title("Feature Count per Image")
        plt.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        plt.show()
