import numpy as np
import math


class TemporalChunkCrop(object):

    def __init__(self, size: int = 4):
        self.S = size

    def __call__(self, frame_indices, tracklet_index):
        sample_clip = []
        F = len(frame_indices)
        if F < self.S:
            strip = list(range(0, F)) + [F-1] * (self.S - F)
            for s in range(self.S):
                pool = strip[s * 1:(s + 1) * 1]
                sample_clip.append(list(pool))
        else:
            interval = math.ceil(F / self.S)
            strip = list(range(0, F)) + [F-1] * (interval * self.S - F)
            for s in range(self.S):
                pool = strip[s * interval:(s + 1) * interval]
                sample_clip.append(list(pool))
        return [ frame_indices[idx] for idx
                 in np.array(sample_clip)[:, 0].tolist() ]


class RandomTemporalChunkCrop(object):

    def __init__(self, size: int = 4):
        self.S = size

    def __call__(self, frame_indices, tracklet_index):
        sample_clip = []
        F = len(frame_indices)
        if F < self.S:
            strip = list(range(0, F)) + [F-1] * (self.S - F)
            for s in range(self.S):
                pool = strip[s * 1:(s + 1) * 1]
                sample_clip.append(list(pool))
        else:
            interval = math.ceil(F / self.S)
            strip = list(range(0, F)) + [F-1] * (interval * self.S - F)
            for s in range(self.S):
                pool = strip[s * interval:(s + 1) * interval]
                sample_clip.append(list(pool))

        sample_clip = np.array(sample_clip)
        sample_clip = sample_clip[np.arange(self.S),
                        np.random.randint(0, sample_clip.shape[1], self.S)]
        return [ frame_indices[idx] for idx in sample_clip ]


class MultiViewTemporalTransform(object):

    def __init__(self, size: int = 4):
        self.size = size

    def __call__(self, candidate, tracklet_index):
        img_paths = []
        candidate_perm = np.random.permutation(len(candidate))
        for idx in range(self.size):
            cur_tracklet = candidate_perm[idx % len(candidate_perm)]
            cur_frame = np.random.randint(0, len(candidate[cur_tracklet][0]))
            cur_img_path = candidate[cur_tracklet][0][cur_frame]
            img_paths.append(cur_img_path)
        return img_paths


class TemporalRandomFrames(object):
    """
    Get size random frames (without replacement if possible) from a video
    """

    def __init__(self, num_images=4):
        self.num_images = num_images

    def __call__(self, frame_indices, tracklet_index):
        frame_indices = list(frame_indices)
        if len(frame_indices) < self.num_images:
            return list(np.random.choice(frame_indices, size=self.num_images, replace=True))

        return list(np.random.choice(frame_indices, size=self.num_images, replace=False))
