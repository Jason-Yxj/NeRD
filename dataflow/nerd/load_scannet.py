import os
from subprocess import check_output

import imageio
import numpy as np
import PIL.Image

import utils.exif_helper as ehelp
from utils.exposure_helper import calculate_ev100_from_metadata
import tensorflow as tf

# Slightly modified version of LLFF data loading code
# see https://github.com/Fyusion/LLFF for original


def ensureNoChannel(x: np.ndarray) -> np.ndarray:
    ret = x
    if len(x.shape) > 2:
        ret = ret[:, :, 0]

    return ret


def handle_exif(basedir):
    files = [
        os.path.join(basedir, "exported/color", f)
        for f in sorted(os.listdir(os.path.join(basedir, "exported/color")))
        if f.endswith("JPG") or f.endswith("jpg")
    ]

    if len(files) == 0:
        raise Exception(
            "Only jpegs can be used to gain EXIF data needed to handle tonemapping"
        )

    ret = []
    for f in files:
        try:
            img = PIL.Image.open(f)
            exif_data = ehelp.formatted_exif_data(img)

            iso = int(exif_data["ISOSpeedRatings"])
            aperture = float(exif_data["FNumber"])

            exposureTime = ehelp.getExposureTime(exif_data)

            ev100 = calculate_ev100_from_metadata(aperture, exposureTime, iso)
        except:
            ev100 = 8

        ret.append(ev100)

    if len(files) != len(ret):
        raise Exception("Not all images are valid!")

    return np.stack(ret, 0).astype(np.float32)


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3, :3].T, (pts - c2w[:3, 3])[..., np.newaxis])[..., 0]
    return tt


def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.0])
    hwf = c2w[:, 4:5]

    for theta in np.linspace(0.0, 2.0 * np.pi * rots, N + 1)[:-1]:
        c = np.dot(
            c2w[:3, :4],
            np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0])
            * rads,
        )
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses


def recenter_poses(poses):

    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.0], [1, 4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses


#####################


def spherify_poses(poses, bds):
    def p34_to_44(p):
        return np.concatenate(
            [p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1
        )

    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(
            -np.linalg.inv((np.transpose(A_i, [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0)
        )
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)

    center = pt_mindist
    up = (poses[:, :3, 3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([0.1, 0.2, 0.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))

    sc = 1.0 / rad
    poses_reset[:, :3, 3] *= sc
    bds *= sc
    rad *= sc

    centroid = np.mean(poses_reset[:, :3, 3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad ** 2 - zh ** 2)
    new_poses = []

    for th in np.linspace(0.0, 2.0 * np.pi, 120):

        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0, 0, -1.0])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)

    new_poses = np.concatenate(
        [new_poses, np.broadcast_to(poses[0, :3, -1:], new_poses[:, :3, -1:].shape)], -1
    )
    poses_reset = np.concatenate(
        [
            poses_reset[:, :3, :4],
            np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape),
        ],
        -1,
    )

    return poses_reset, new_poses, bds


def load_scannet_data(
    basedir, recenter=True, spherify=False, path_zflat=False,
):
    poses = []
    pose_dir = os.path.join(basedir, "exported/pose")
    pose_files = os.listdir(pose_dir)
    pose_files.sort(key=lambda x: int(x[:-4]))
    pose_paths = [os.path.join(pose_dir, f) for f in pose_files]
    for pose_path in pose_paths:
        w2c = np.loadtxt(pose_path)
        c2w = np.linalg.inv(w2c)
        poses.append(c2w[:3, :])
    poses = np.array(poses).astype(np.float32)

    imgs = []
    img_dir = os.path.join(basedir, "exported/color")
    img_files = os.listdir(img_dir)
    img_files.sort(key=lambda x: int(x[:-4]))
    img_paths = [os.path.join(img_dir, f) for f in img_files]
    for img_path in img_paths:
        img = imageio.imread(img_path)
        img = img[..., :3]/255.0 
        imgs.append(img)
    images = np.stack(imgs, -1)

    intrinsic_path = os.path.join(basedir, "exported/intrinsic/intrinsic_color.txt")
    K = np.loadtxt(intrinsic_path)
    f = K[0, 0]
    n, h, w = images.shape[:3]
    hwf = [[[h], [w], [f]]]
    hwfs = np.tile(hwf, (n, 1, 1)) # nx3x1
    poses = np.concatenate((poses, hwfs), axis=-1)

    masks = np.ones((n, h, w))
    bd = [[2, 6]]
    bds = np.tile(bd, (n, 1))

    ev100s = handle_exif(basedir)

    if recenter:
        poses = recenter_poses(poses)

    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)

    else:

        c2w = poses_avg(poses)
        print("recentered", c2w.shape)
        print(c2w[:3, :4])

        # Get spiral
        # Get average pose
        up = normalize(poses[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = bds.min() * 0.9, bds.max() * 5.0
        dt = 0.75
        mean_dz = 1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))
        focal = mean_dz

        # Get radii for spiral path
        zdelta = close_depth * 0.2
        tt = poses[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        N_views = 120
        N_rots = 2
        if path_zflat:
            #             zloc = np.percentile(tt, 10, 0)[2]
            zloc = -close_depth * 0.1
            c2w_path[:3, 3] = c2w_path[:3, 3] + zloc * c2w_path[:3, 2]
            rads[2] = 0.0
            N_rots = 1
            N_views /= 2

        # Generate poses for spiral path
        render_poses = render_path_spiral(
            c2w_path, up, rads, focal, zdelta, zrate=0.5, rots=N_rots, N=N_views
        )

    render_poses = np.array(render_poses).astype(np.float32)

    c2w = poses_avg(poses)
    print("Data:")
    print(poses.shape, images.shape, masks.shape, bds.shape)

    dists = np.sum(np.square(c2w[:3, 3] - poses[:, :3, 3]), -1)
    i_test = np.argmin(dists)
    print("HOLDOUT view is", i_test)

    images = images.astype(np.float32)
    masks = masks.astype(np.float32)
    poses = poses.astype(np.float32)

    H = 480
    W = 640
    imgs = tf.image.resize(imgs, [H, W], method="area").numpy()
    masks = tf.image.resize(masks, [H, W], method="area").numpy()   
    fx = K[0, 0] * (630 / w)
    fy = K[1, 1] * (480 / h)


    return images, masks, ev100s, poses, bds, render_poses, [H, W, fx, fy], i_test
