import numpy as np

def map_cloud_to_grid(voxel_size, points, values):
    voxel = np.zeros((40, 40, 40), dtype=np.float32)
    indices = (points // voxel_size).astype(int)
    voxel[tuple(indices.T)] = values.squeeze()
    return voxel


def grid_to_map_cloud(voxel_size, voxel, threshold=1e-2):
    points = np.argwhere(voxel > threshold) * voxel_size
    values = np.expand_dims(voxel[voxel > threshold], 1)
    return points, values