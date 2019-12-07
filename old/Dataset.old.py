import numpy as np
import torch
#from matplotlib import pyplot as plt
from skimage.segmentation import slic
from scipy.spatial.distance import cdist
import scipy.ndimage

knn_graph = 28  # maximum number of neighbors for each node

def sparsify_graph(A, knn_graph):
    if knn_graph is not None and knn_graph < A.shape[0]:
        idx = np.argsort(A, axis=0)[:-knn_graph, :]
        np.put_along_axis(A, idx, 0, axis=0)
        idx = np.argsort(A, axis=1)[:, :-knn_graph]
        np.put_along_axis(A, idx, 0, axis=1)
    return A


def spatial_graph(coord, img_size, knn_graph=32):
    coord = coord / np.array(img_size, np.float)
    dist = cdist(coord, coord)
    sigma = 0.1 * np.pi
    A = np.exp(- dist / sigma ** 2)
    A[np.diag_indices_from(A)] = 0  # remove self-loops
    sparsify_graph(A, knn_graph)
    return A  # adjacency matrix (edges)


def visualize_superpixels(avg_values, superpixels):
    n_ch = avg_values.shape[1]
    img_sp = np.zeros((*superpixels.shape, n_ch))
    for sp in np.unique(superpixels):
        mask = superpixels == sp
        for c in range(n_ch):
            img_sp[:, :, c][mask] = avg_values[sp, c]
    return img_sp


def superpixel_features(img, superpixels):
    n_sp = len(np.unique(superpixels))
    n_ch = img.shape[2]
    avg_values = np.zeros((n_sp, n_ch))
    coord = np.zeros((n_sp, 2))
    masks = []
    for sp in np.unique(superpixels):
        mask = superpixels == sp
        for c in range(n_ch):
            avg_values[sp, c] = np.mean(img[:, :, c][mask])
        coord[sp] = np.array(scipy.ndimage.measurements.center_of_mass(mask))  # row, col
        masks.append(mask)
    return avg_values, coord, masks


def prepareGraph(imgBatch):
    listOfAvg = []
    listOfCoord = []
    listofA_spatial = []
    for img in imgBatch:
        #print('img size', img.shape)
        img = img.permute(1,2,0).double().numpy()
        superpixels = slic(img)
        avg_values, coord, masks = superpixel_features(img, superpixels)
        A_spatial = spatial_graph(coord, img.shape[:2], knn_graph=knn_graph)  # keep only 16 neighbors for each node
        listOfAvg.append(avg_values)
        listOfCoord.append(coord)
        listofA_spatial.append(A_spatial)
    return np.asarray(listOfCoord),np.asarray(listOfAvg),np.asarray(listofA_spatial)



# img_sp = visualize_superpixels(avg_values, superpixels)
# plt.figure(figsize=(15, 5))
# plt.subplot(121)
# plt.imshow(img_sp)
# plt.title('$N=${} superpixels, mean color {} and coord {} features'.format(len(np.unique(superpixels)),
#                                                                            avg_values.shape,
#                                                                            coord.shape), fontsize=10)
# plt.subplot(122)
# plt.imshow(A_spatial ** 0.2)
# plt.colorbar()
# plt.title('Adjacency matrix of spatial edges')
# plt.show()