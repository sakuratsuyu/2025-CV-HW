import cv2
import numpy as np

def multiband_blend_opencv(img1, img2, mask1, mask2, num_bands=5):

    mask1 = mask1.astype(np.uint8) * 255
    mask2 = mask2.astype(np.uint8) * 255

    h, w = img1.shape[:2]

    # create MultiBandBlender
    blender = cv2.detail_MultiBandBlender(0, 0)
    blender.prepare((0, 0, w, h))

    blender.feed(img1, mask1, (0, 0))
    blender.feed(img2, mask2, (0, 0))

    # blending
    result, result_mask = blender.blend(None, None)

    return result.astype(np.uint8)


def build_gaussian_pyramid(img, num_levels):
    G = [img.astype(np.float32)]
    for i in range(num_levels):
        img = cv2.pyrDown(img, borderType=cv2.BORDER_REPLICATE)
        G.append(img.astype(np.float32))
    return G

def build_laplacian_pyramid(G):
    L = []
    for i in range(len(G) - 1):
        size = (G[i].shape[1], G[i].shape[0])
        GE = cv2.pyrUp(G[i + 1], dstsize=size)
        L.append(G[i] - GE)
    L.append(G[-1])
    return L

def reconstruct_from_laplacian(L):
    img = L[-1]
    for i in range(len(L) - 2, -1, -1):
        size = (L[i].shape[1], L[i].shape[0])
        img = cv2.pyrUp(img, dstsize=size)
        img = img + L[i]
    return img

def multiband_blend_manual(img1, img2, mask, num_levels=8):
    # Convert mask to float32 and 3 channels
    mask = mask.astype(np.float32)
    if len(mask.shape) == 2:
        mask = np.repeat(mask[:, :, None], 3, axis=2)

    # Gaussian pyramids for images and mask
    G1 = build_gaussian_pyramid(img1, num_levels)
    G2 = build_gaussian_pyramid(img2, num_levels)
    GM = build_gaussian_pyramid(mask, num_levels)

    # Laplacian pyramids for images
    L1 = build_laplacian_pyramid(G1)
    L2 = build_laplacian_pyramid(G2)

    # Blend each Laplacian layer
    LS = []
    for l1, l2, gm in zip(L1, L2, GM):
        LS.append(gm * l1 + (1 - gm) * l2)

    # Reconstruct blended image
    blended = reconstruct_from_laplacian(LS)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return blended


# ---------- Gradient Mask ----------
def create_feather_mask(img1_mask, img2_mask):

    overlap = (img1_mask > 0) & (img2_mask > 0)
    ys, xs = np.where(overlap)

    x_min, x_max = xs.min(), xs.max()

    if np.where(img1_mask > 0)[1].min() < np.where(img2_mask > 0)[1].min():
        gradient = np.linspace(1, 0, x_max - x_min + 1)
    else:
        gradient = np.linspace(0, 1, x_max - x_min + 1)

    mask = np.zeros_like(img1_mask, dtype=np.float32)
    mask[img1_mask > 0] = 1.0

    for y, x in zip(ys, xs):
        mask[y, x] = gradient[x - x_min]

    mask = cv2.GaussianBlur(mask, (0, 0), 30)

    return mask


# ---------- Edge Mask ----------
# def create_feather_mask(img1_mask, img2_mask):

#     both_mask = img1_mask | img2_mask
#     no_mask = ~both_mask

#     def border(mask):
#         mask = mask.astype(np.uint8)

#         gx = cv2.Sobel(mask, cv2.CV_32F, 1, 0, ksize=3)
#         gy = cv2.Sobel(mask, cv2.CV_32F, 0, 1, ksize=3)
#         border_mag = cv2.magnitude(gx, gy)

#         border_bin = (border_mag > 50 / 255).astype(np.uint8)
#         return border_bin

#     border1 = 1 - border(img1_mask)
#     border2 = 1 - border(img2_mask)

#     # distance transform
#     d1 = cv2.distanceTransform(border1, cv2.DIST_L2, 5)
#     d2 = cv2.distanceTransform(border2, cv2.DIST_L2, 5)
#     d1 = d1 / d1.max()
#     d2 = d2 / d2.max()

#     dsum = d1 + d2
#     dsum[dsum == 0] = 1e-8
#     mask_mb = d2 / dsum

#     mask_mb[no_mask] = 1.0
#     mask_mb[img1_mask & (~img2_mask)] = 1.0
#     mask_mb[img2_mask & (~img1_mask)] = 0.0

#     # smooth slightly for stability
#     # mask_mb = cv2.GaussianBlur(mask_mb, (0,0), 5)

#     return mask_mb.astype(np.float32)


# ---------- Distance Mask ----------
# def create_feather_mask(img1_mask, img2_mask):

#     # distance transform
#     d1 = cv2.distanceTransform(img1_mask.astype(np.uint8), cv2.DIST_L2, 5)
#     d2 = cv2.distanceTransform(img2_mask.astype(np.uint8), cv2.DIST_L2, 5)
#     d1 = d1 / d1.max()
#     d2 = d2 / d2.max()

#     dsum = d1 + d2
#     dsum[dsum == 0] = 1e-8
#     mask_mb = d2 / dsum

#     # smooth slightly for stability
#     mask_mb = cv2.GaussianBlur(mask_mb, (0, 0), 30)

#     return mask_mb.astype(np.float32)