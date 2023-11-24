import os
import errno
from os import path
from glob import glob

import cv2
import numpy as np
from scipy.sparse import linalg as linalg
from scipy.sparse import lil_matrix as lil_matrix

# Helper enum
OMEGA = 0
DEL_OMEGA = 1
OUTSIDE = 2


# Determine if a given index is inside omega, on the boundary (del omega),
# or outside the omega region
def point_location(index, mask):
    if in_omega(index, mask) == False:
        return OUTSIDE
    if edge(index, mask) == True:
        return DEL_OMEGA
    return OMEGA


# Determine if a given index is either outside or inside omega
def in_omega(index, mask):
    return mask[index] == 1


# Deterimine if a given index is on del omega (boundary)
def edge(index, mask):
    if in_omega(index, mask) == False:
        return False
    for pt in get_surrounding(index):
        # If the point is inside omega, and a surrounding point is not,
        # then we must be on an edge
        if in_omega(pt, mask) == False:
            return True
    return False


# Apply the Laplacian operator at a given index
def lapl_at_index(source, index):
    i, j = index
    val = (
        (4 * source[i, j])
        - (1 * source[i + 1, j])
        - (1 * source[i - 1, j])
        - (1 * source[i, j + 1])
        - (1 * source[i, j - 1])
    )
    return val


# Find the indicies of omega, or where the mask is 1
def mask_indices(mask):
    nonzero = np.nonzero(mask)
    return list(zip(nonzero[0], nonzero[1]))


# Get indicies above, below, to the left and right
def get_surrounding(index):
    i, j = index
    return [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]


# Create the A sparse matrix
def poisson_sparse_matrix(points):
    # N = number of points in mask
    N = len(points)
    A = lil_matrix((N, N))
    # Set up row for each point in mask
    for i, index in enumerate(points):
        # Should have 4's diagonal
        A[i, i] = 4
        # Get all surrounding points
        for x in get_surrounding(index):
            # If a surrounding point is in the mask, add -1 to index's
            # row at correct position
            if x not in points:
                continue
            j = points.index(x)
            A[i, j] = -1
    return A


# Main method
# Does Poisson image editing on one channel given a source, target, and mask
def process(source, target, mask):
    indicies = mask_indices(mask)
    N = len(indicies)
    # Create poisson A matrix. Contains mostly 0's, some 4's and -1's
    A = poisson_sparse_matrix(indicies)
    # Create B matrix
    b = np.zeros(N)
    for i, index in enumerate(indicies):
        # Start with left hand side of discrete equation
        b[i] = lapl_at_index(source, index)
        # If on boundry, add in target intensity
        # Creates constraint lapl source = target at boundary
        if point_location(index, mask) == DEL_OMEGA:
            for pt in get_surrounding(index):
                if in_omega(pt, mask) == False:
                    b[i] += target[pt]

    # Solve for x, unknown intensities
    x = linalg.cg(A, b)
    # Copy target photo, make sure as int
    composite = np.copy(target).astype(int)
    # Place new intensity on target at given index
    for i, index in enumerate(indicies):
        composite[index] = x[0][i]
    return composite


# Naive blend, puts the source region directly on the target.
# Useful for testing
def preview(source, target, mask):
    return (target * (1.0 - mask)) + (source * (mask))


IMG_EXTENSIONS = ["png", "jpeg", "jpg", "JPG"]
SRC_FOLDER = "input"
OUT_FOLDER = "output"


def collect_files(prefix, extension_list=IMG_EXTENSIONS):
    filenames = sum(map(glob, [prefix + ext for ext in extension_list]), [])
    return filenames


for dirpath, dirnames, fnames in os.walk(SRC_FOLDER):
    image_dir = os.path.split(dirpath)[-1]
    output_dir = os.path.join(OUT_FOLDER, image_dir)
    print(f"Processing input {image_dir}...")

    # Search for images to process
    source_names = collect_files(os.path.join(dirpath, "*source."))
    target_names = collect_files(os.path.join(dirpath, "*target."))
    mask_names = collect_files(os.path.join(dirpath, "*mask."))

    if not len(source_names) == len(target_names) == len(mask_names) == 1:
        print("There must be one source, one target, and one mask per input.")
        continue

    # Read images
    source_img = cv2.imread(source_names[0], cv2.IMREAD_COLOR)
    target_img = cv2.imread(target_names[0], cv2.IMREAD_COLOR)
    mask_img = cv2.imread(mask_names[0], cv2.IMREAD_GRAYSCALE)

    # Normalize mask to range [0,1]
    mask = np.atleast_3d(mask_img).astype(float) / 255.0
    # Make mask binary
    mask[mask != 1] = 0
    # Trim to one channel
    mask = mask[:, :, 0]
    channels = source_img.shape[-1]
    # Call the poisson method on each individual channel
    result_stack = [
        process(source_img[:, :, i], target_img[:, :, i], mask)
        for i in range(channels)
    ]
    # Merge the channels back into one image
    result = cv2.merge(result_stack)
    # Make result directory if needed
    try:
        os.makedirs(output_dir)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    # Write result
    cv2.imwrite(path.join(output_dir, "result.png"), result)
    print("Finished processing input {image_dir}.")
