import cv2
import numpy as np
from tqdm import tqdm, trange
import time
from multiband_blending import create_feather_mask, multiband_blend_manual, multiband_blend_opencv

# ---------- Load images ----------

def load_imgs_tree(tree):

    if isinstance(tree, str):
        img = cv2.imread(tree)
        if img is None:
            raise FileNotFoundError(f"Image file not found: {tree}")
        return img

    assert isinstance(tree, list) and len(tree) == 2, "tree must be a list of length 2"

    return [load_imgs_tree(tree[0]), load_imgs_tree(tree[1])]


# ---------- Feature Detection and Description ----------

def detect_and_describe_SIFT(img):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return keypoints, descriptors

def describe_concatenation(keypoints, img, patch_size=11):
    half = patch_size // 2
    kps = []
    descs = []

    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        if x - half < 0 or x + half >= img.shape[1] or \
           y - half < 0 or y + half >= img.shape[0]:
            continue

        patch = img[y - half : y + half + 1, x - half : x + half + 1]

        kps.append(kp)
        descs.append(patch.flatten())

    descs = np.array(descs, dtype=np.float32)
    return kps, descs

def match_features(desc1, desc2, ratio=0.7):
    matches = []

    a2 = np.sum(desc1 ** 2, axis=1)[:, None]      # (M, 1)
    b2 = np.sum(desc2 ** 2, axis=1)[None, :]      # (1, N)
    distances = a2 + b2 - 2 * desc1 @ desc2.T     # (M, N)

    # sort distances for each descriptor in image1
    idx_sorted = np.argsort(distances, axis=1)
    d1_idx = idx_sorted[:, 0]
    d2_idx = idx_sorted[:, 1]

    # get distances
    d1 = distances[np.arange(distances.shape[0]), d1_idx]
    d2 = distances[np.arange(distances.shape[0]), d2_idx]

    # ratio test
    mask = (d1 / (d2 + 1e-10)) < ratio
    matches = [(i, d1_idx[i]) for i in range(len(mask)) if mask[i]]

    return matches

def get_matched_points(kp1, kp2, matches):
    pts1 = np.float32([kp1[i].pt for i, _ in matches])
    pts2 = np.float32([kp2[j].pt for _, j in matches])
    return pts1, pts2


# ---------- Homography ----------

def normalize_points(pts, width, height):
    x = pts[:, 0]
    y = pts[:, 1]
    x_norm = 2.0 * (x / width) - 1.0
    y_norm = 2.0 * (y / height) - 1.0
    return np.stack([x_norm, y_norm], axis=1)

def denormalize_homography(H, width1, height1, width2, height2):
    T1 = np.array([
        [2.0 / width1, 0, -1.0],
        [0, 2.0 / height1, -1.0],
        [0, 0, 1],
    ])

    T2_inv = np.array([
        [width2 / 2.0, 0, width2 / 2.0],
        [0, height2 / 2.0, height2 / 2.0],
        [0, 0, 1]
    ])

    H_norm = T2_inv @ H @ T1
    return H_norm

def project_points(points, H):
    pts_h = np.hstack([points, np.ones((points.shape[0], 1))])
    proj = (H @ pts_h.T).T
    proj = proj[:, :2] / proj[:, 2:]
    return proj

def ransac_homography_opencv(pts1, pts2):
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC)
    return H

def compute_homography_DLT(pts1, pts2):
    A = []
    for (x, y), (u, v) in zip(pts1, pts2):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    A = np.array(A)

    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1, :].reshape(3, 3)
    H = H / H[-1, -1]
    return H

def ransac_homography(pts1, pts2, max_iter=100, threshold=0.01):

    n_pts = pts1.shape[0]
    best_inliers = []
    best_H = None

    SUCCESS_INLIER = 20
    MEDIAN_ERROR_THRESHOLD = 3.0
    cnt = 0

    for _ in trange(max_iter):
        # randomly select 4 points and compute homography
        idx = np.random.choice(n_pts, 4, replace=False)
        H = compute_homography_DLT(pts1[idx], pts2[idx])

        # compute reprojection error
        pts1_proj = project_points(pts1, H)
        errors = np.linalg.norm(pts1_proj - pts2, axis=1)
        inliers = np.where(errors < threshold)[0]

        num_inliers = inliers.shape[0]
        median_error = np.median(errors[inliers])
        # print("num inliers: ", num_inliers)
        # print("median error: ", median_error)
        success = (num_inliers >= SUCCESS_INLIER) and (median_error < MEDIAN_ERROR_THRESHOLD)
        if success:
            cnt += 1

        # update best model
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_H = H

    print(f"ransac success: {cnt} / {max_iter}")
    if cnt < max_iter / 2:
        print(f"RANSAC failed, {cnt} / {max_iter} iterations")

    return best_H, best_inliers


# ---------- M-estimator ----------

def tukey_weights(errors, c=4.685):
    weights = np.zeros_like(errors)
    mask = errors < c
    r = errors[mask] / c
    weights[mask] = (1 - r**2)**2
    return weights

def refine_with_m_estimator(pts1, pts2, H_init):
    H = H_init.copy()

    for _ in range(10):
        proj = project_points(pts1, H)
        errors = np.linalg.norm(proj - pts2, axis=1)

        w = tukey_weights(errors)
        W = np.diag(w)

        # Solve weighted DLT
        A = []
        for i, ((x, y), (u, v)) in enumerate(zip(pts1, pts2)):
            wi = W[i, i]
            A.append(wi * np.array([-x, -y, -1, 0, 0, 0, u*x, u*y, u]))
            A.append(wi * np.array([0, 0, 0, -x, -y, -1, v*x, v*y, v]))

        A = np.array(A)
        U, S, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)
        H /= H[2, 2]

    return H


# ---------- Stitching ----------

def draw_matches(img1, kp1, img2, kp2, matches):
    match_cv2 = []
    for (i, j) in matches:
        match_cv2.append(
            cv2.DMatch(_queryIdx=i, _trainIdx=j, _imgIdx=0, _distance=0)
        )
    return cv2.drawMatches(img1, kp1, img2, kp2, match_cv2, None)

def stitch_two_images(img1, img2, H):
    # Warp img1 into img2's coordinate space
    h2, w2 = img2.shape[:2]

    # Compute the corners of img1 after warping
    h1, w1 = img1.shape[:2]
    corners_img1 = np.float32([
        [0, 0],
        [w1, 0],
        [w1, h1],
        [0, h1]
    ]).reshape(-1, 1, 2)

    corners_img1_warped = cv2.perspectiveTransform(corners_img1, H)

    # Get corners of image2
    corners_img2 = np.float32([
        [0, 0],
        [w2, 0],
        [w2, h2],
        [0, h2]
    ]).reshape(-1, 1, 2)

    # Combine all corners to compute bounding box
    all_corners = np.concatenate((corners_img1_warped, corners_img2), axis=0)

    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
 
    # Translation to avoid negative coordinates
    translation = [-xmin, -ymin]

    # Output size
    width = xmax - xmin
    height = ymax - ymin

    # Warp image1
    H_translate = np.array([
        [1, 0, translation[0]],
        [0, 1, translation[1]],
        [0, 0, 1]
    ], dtype=np.float32)

    warped_img1 = cv2.warpPerspective(img1, H_translate @ H, (width, height), flags=cv2.INTER_NEAREST)


    canvas_img2 = np.zeros((height, width, 3), dtype=np.uint8)
    canvas_img2[translation[1]: translation[1] + h2,
                translation[0]: translation[0] + w2] = img2
    
    img1_mask = (cv2.cvtColor(warped_img1, cv2.COLOR_BGR2GRAY) > 0)
    img2_mask = (cv2.cvtColor(canvas_img2, cv2.COLOR_BGR2GRAY) > 0)
    
    both_mask = np.logical_or(img1_mask, img2_mask)
    overlap_mask = np.logical_and(img1_mask, img2_mask)
    img1_only_mask = np.logical_and(img1_mask, ~img2_mask)
    img2_only_mask = np.logical_and(~img1_mask, img2_mask)

    canvas_linear = np.zeros((height, width, 3), dtype=np.uint8)
    canvas_linear[overlap_mask, :] = warped_img1[overlap_mask, :] * 0.5 + canvas_img2[overlap_mask, :] * (1 - 0.5)
    canvas_linear[img1_only_mask, :] = warped_img1[img1_only_mask]
    canvas_linear[img2_only_mask, :] = canvas_img2[img2_only_mask]

    feather_mask = create_feather_mask(img1_mask, img2_mask)
    canvas_gradient = warped_img1 * feather_mask[:, :, None] + canvas_img2 * (1 - feather_mask[:, :, None])

    feather_mask = create_feather_mask(img1_mask, img2_mask)
    warped_img1_filled = cv2.inpaint(warped_img1, ((1 - img1_mask) * 255).astype(np.uint8), 3, cv2.INPAINT_NS)
    canvas_img2_filled = cv2.inpaint(canvas_img2, ((1 - img2_mask) * 255).astype(np.uint8), 3, cv2.INPAINT_NS)
    canvas_band = multiband_blend_manual(warped_img1_filled, canvas_img2_filled, feather_mask)
    canvas_band[~both_mask] = 0.0
    canvas_band[img1_only_mask, :] = warped_img1[img1_only_mask]
    canvas_band[img2_only_mask, :] = canvas_img2[img2_only_mask]

    return canvas_band


# ---------- Main ----------

def run_stitch(img1, img2):
    start = time.time()
    print(f"Processing image")

    # SIFT detect + describe
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kp1, desc1 = detect_and_describe_SIFT(img1_gray)
    kp2, desc2 = detect_and_describe_SIFT(img2_gray)

    # Concatenation descriptors
    # kp1, desc1 = describe_concatenation(kp1, img1, patch_size=11)
    # kp2, desc2 = describe_concatenation(kp2, img2, patch_size=11)

    # manual feature matching (ratio test)
    matches = match_features(desc1, desc2, ratio=0.7)
    pts1, pts2 = get_matched_points(kp1, kp2, matches)

    # estimate homography
    pts1 = normalize_points(pts1, img1.shape[1], img1.shape[0])
    pts2 = normalize_points(pts2, img2.shape[1], img2.shape[0])
    H, inliers = ransac_homography(pts1, pts2)
    # H = refine_with_m_estimator(pts1[inliers], pts2[inliers], H)
    H = denormalize_homography(H, img1.shape[1], img1.shape[0], img2.shape[1], img2.shape[0])

    # H = ransac_homography_opencv(pts1, pts2)

    # stitch two images
    merged = stitch_two_images(img1, img2, H)

    # visualize matches
    # matched_vis = draw_matches(img1, kp1, img2, kp2, matches[:100])
    # cv2.imwrite("matches.jpg", matched_vis)

    # stitch images
    # panorama = stitch_images(img1, img2, H)
    # cv2.imwrite("panorama.jpg", panorama)

    # warp image1 to image2
    # h, w, _ = img2.shape
    # warped = cv2.warpPerspective(img1, H, (w, h))
    # cv2.imwrite("warped_1_to_2.jpg", warped)

    # # warp image2 to image1
    # h, w, _ = img1.shape
    # warped = cv2.warpPerspective(img2, np.linalg.inv(H), (w, h))
    # cv2.imwrite("warped_2_to_1.jpg", warped)

    end = time.time()
    print(f"Elapsed time: {end - start:.6f} s")

    return merged

def stitch_tree(tree):
    
    if not isinstance(tree, list):
        return tree
    
    assert len(tree) == 2

    left = stitch_tree(tree[0])
    right = stitch_tree(tree[1])

    merged = run_stitch(left, right)
    return merged


if __name__ == "__main__":

    np.random.seed(42)

    # names = [
    #     [
    #         [
    #             "../data/data1/112_1300.JPG",
    #             "../data/data1/113_1301.JPG",
    #         ],
    #         [
    #             "../data/data1/112_1298.JPG",
    #             "../data/data1/112_1299.JPG",
    #         ],
    #     ],
    #     [
    #         "../data/data1/113_1302.JPG",
    #         "../data/data1/113_1303.JPG",
    #     ]
    # ]
    
    names = [
        [
            "../data/data2/IMG_0491.JPG",
            "../data/data2/IMG_0490.JPG",
        ],
        [
            "../data/data2/IMG_0488.JPG",
            "../data/data2/IMG_0489.JPG",
        ],
    ]

    # names = [
    #     "../data/data3/IMG_0677.JPG",
    #     [
    #         "../data/data3/IMG_0675.JPG",
    #         "../data/data3/IMG_0676.JPG",
    #     ],
    # ]

    # names = [
    #     [
    #         "../data/data4/IMG_7358.JPG",
    #         "../data/data4/IMG_7357.JPG",
    #     ],
    #     [
    #         "../data/data4/IMG_7355.JPG",
    #         "../data/data4/IMG_7356.JPG",
    #     ],
    # ]

    imgs = load_imgs_tree(names)
    result = stitch_tree(imgs)

    cv2.imwrite("canvas.jpg", result)