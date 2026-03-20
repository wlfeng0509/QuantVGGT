import argparse
import os
import sys
import glob
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# CUDA backend config (match demo settings)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Ensure project root is in sys.path for absolute imports like `vggt.*`
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pycolmap
import trimesh
import time

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.utils.eval_utils import (
    load_images_rgb,
    get_vgg_input_imgs,
    compute_original_coords,
)


def _build_pycolmap_intri(fidx, intrinsics, camera_type, extra_params=None):
    """
    Build pycolmap camera params from intrinsics for different camera models.
    """
    if camera_type == "PINHOLE":
        pycolmap_intri = np.array(
            [
                intrinsics[fidx][0, 0],
                intrinsics[fidx][1, 1],
                intrinsics[fidx][0, 2],
                intrinsics[fidx][1, 2],
            ]
        )
    elif camera_type == "SIMPLE_PINHOLE":
        focal = (intrinsics[fidx][0, 0] + intrinsics[fidx][1, 1]) / 2
        pycolmap_intri = np.array(
            [focal, intrinsics[fidx][0, 2], intrinsics[fidx][1, 2]]
        )
    else:
        raise ValueError(f"Camera type {camera_type} is not supported yet")
    return pycolmap_intri


def batch_np_matrix_to_pycolmap_wo_track(
    points3d,
    points_xyf,
    points_rgb,
    extrinsics,
    intrinsics,
    image_size,
    shared_camera=False,
    camera_type="SIMPLE_PINHOLE",
):
    """
    Convert batched numpy arrays to a pycolmap.Reconstruction without building tracks.
    Only used to export an initialized reconstruction for visualization or as init.
    """
    N = len(extrinsics)
    P = len(points3d)

    reconstruction = pycolmap.Reconstruction()

    for vidx in range(P):
        reconstruction.add_point3D(points3d[vidx], pycolmap.Track(), points_rgb[vidx])

    camera = None
    for fidx in range(N):
        if camera is None or (not shared_camera):
            pycolmap_intri = _build_pycolmap_intri(fidx, intrinsics, camera_type)
            camera = pycolmap.Camera(
                model=camera_type,
                width=image_size[0],
                height=image_size[1],
                params=pycolmap_intri,
                camera_id=fidx + 1,
            )
            reconstruction.add_camera(camera)

        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(extrinsics[fidx][:3, :3]), extrinsics[fidx][:3, 3]
        )
        image = pycolmap.Image(
            id=fidx + 1,
            name=f"image_{fidx + 1}",
            camera_id=camera.camera_id,
            cam_from_world=cam_from_world,
        )

        points2D_list = []
        point2D_idx = 0
        points_belong_to_fidx = points_xyf[:, 2].astype(np.int32) == fidx
        points_belong_to_fidx = np.nonzero(points_belong_to_fidx)[0]

        for point3D_batch_idx in points_belong_to_fidx:
            point3D_id = point3D_batch_idx + 1
            point2D_xyf = points_xyf[point3D_batch_idx]
            point2D_xy = point2D_xyf[:2]
            points2D_list.append(pycolmap.Point2D(point2D_xy, point3D_id))

            track = reconstruction.points3D[point3D_id].track
            track.add_element(fidx + 1, point2D_idx)
            point2D_idx += 1

        try:
            image.points2D = pycolmap.ListPoint2D(points2D_list)
            image.registered = True
        except Exception:
            print(f"frame {fidx + 1} does not have any points")
            image.registered = False

        reconstruction.add_image(image)

    return reconstruction


def rename_colmap_recons_and_rescale_camera(
    reconstruction,
    image_paths,
    original_coords,
    img_size,
    shift_point2d_to_original_res=False,
    shared_camera=False,
):
    rescale_camera = True

    for pyimageid in reconstruction.images:
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]

        if rescale_camera:
            pred_params = copy.deepcopy(pycamera.params)

            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_image_size) / img_size
            pred_params = pred_params * resize_ratio
            real_pp = real_image_size / 2
            pred_params[-2:] = real_pp

            pycamera.params = pred_params
            pycamera.width = real_image_size[0]
            pycamera.height = real_image_size[1]

        if shift_point2d_to_original_res:
            top_left = original_coords[pyimageid - 1, :2]
            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * resize_ratio

        if shared_camera:
            rescale_camera = False

    return reconstruction


def run_vggt(model, vgg_input, dtype, image_paths=None):
    """
    Run VGGT to predict extrinsics, intrinsics, depth map and depth confidence.
    images: tensor [N, 3, H, W] in [0,1]
    """
    assert len(vgg_input.shape) == 4 and vgg_input.shape[1] == 3

    depth_conf_thresh = 3.0

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=dtype):
            vgg_input_cuda = vgg_input.cuda().to(torch.bfloat16)

            predictions = model(vgg_input_cuda, image_paths=image_paths)

    torch.cuda.synchronize()
    end = time.time()
    inference_time_ms = (end - start) * 1000.0

    print(
        f"VGGT inference time: {inference_time_ms:.1f} ms for {vgg_input.shape[0]} images"
    )
    # Measure max GPU VRAM usage
    if torch.cuda.is_available():
        max_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"Max GPU VRAM used: {max_mem_mb:.2f} MB")

    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], (vgg_input.shape[2], vgg_input.shape[3])
    )

    # conversion stuff
    depth_tensor = predictions["depth"]
    depth_np = depth_tensor[0].detach().float().cpu().numpy()
    depth_conf = predictions["depth_conf"]
    depth_conf_np = depth_conf[0].detach().float().cpu().numpy()
    depth_mask = depth_conf_np >= depth_conf_thresh
    depth_filtered = depth_tensor[0].detach().float().cpu().numpy()
    depth_filtered[~depth_mask] = np.nan
    depth_np = depth_filtered

    extrinsic_np = extrinsic[0].detach().float().cpu().numpy()
    intrinsic_np = intrinsic[0].detach().float().cpu().numpy()

    return extrinsic_np, intrinsic_np, depth_np, depth_conf_np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export COLMAP reconstruction from images using VGGT (images-only)"
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Dataset root containing images/ directory",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default="./colmap_export_custom",
        help="Output directory (will create sparse/ with COLMAP files)",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/home/sy/code/FastVGGT/ckpt/model_tracker_fixed_e20.pt",
        help="Model checkpoint file path",
    )
    parser.add_argument("--merging", type=int, default=0, help="Merging parameter")
    parser.add_argument(
        "--depth_conf_thresh",
        type=float,
        default=3.0,
        help="Depth confidence threshold to filter low-confidence depth",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=100000,
        help="Max number of 3D points to keep when exporting",
    )
    parser.add_argument(
        "--save_images",
        action="store_true",
        help="Save the output images",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Device and dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (
        torch.bfloat16
        if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8)
        else torch.float16
    )
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    images_dir = args.data_path / "images"
    if not images_dir.exists():
        print(f"❌ images directory not found: {images_dir}")
        return

    image_path_list = sorted(glob.glob(os.path.join(str(images_dir), "*")))
    if len(image_path_list) == 0:
        print(f"❌ No images found in {images_dir}")
        return
    base_image_path_list = [os.path.basename(p) for p in image_path_list]
    print(f"🖼️  Found {len(image_path_list)} images")

    # Load model
    print(f"🔄 Loading model: {args.ckpt_path}")
    model = VGGT(merging=args.merging, vis_attn_map=False)
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model = model.cuda().eval()
    model = model.to(torch.bfloat16)
    print("✅ Model loaded")

    # Load and preprocess images
    original_coords = compute_original_coords(
        image_path_list,
    ).to(device)
    # Load images
    print(f"🔄 Loading images...")
    images = load_images_rgb(image_path_list)

    if not images or len(images) < 3:
        print(f"❌ Error: Not enough valid images (need at least 3)")
        return
    print(f"✅ Loaded {len(images)} images")
    images_array = np.stack(images)
    vgg_input, patch_width, patch_height = get_vgg_input_imgs(images_array)
    print(f"📐 Image patch dimensions: {patch_width}x{patch_height}")

    # Update attention layer patch dimensions in the model
    model.update_patch_dimensions(patch_width, patch_height)

    extrinsic, intrinsic, depth_map, depth_conf = run_vggt(
        model, vgg_input, dtype, base_image_path_list
    )

    # Back-project depth to 3D (camera/world coords as defined by util func)
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

    # Colors (resize to match depth/points map grid from vgg_input shape)
    _, _, grid_h, grid_w = vgg_input.shape
    points_rgb = F.interpolate(
        vgg_input,
        size=(grid_h, grid_w),
        mode="bilinear",
        align_corners=False,
    )
    points_rgb = (points_rgb.detach().cpu().numpy() * 255).astype(np.uint8)
    points_rgb = points_rgb.transpose(0, 2, 3, 1)  # [N,H,W,3]

    # Pixel grid with frame index
    num_frames, height, width, _ = points_3d.shape
    points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

    # Confidence filtering and random downsample
    conf_mask = depth_conf >= args.depth_conf_thresh
    conf_mask = randomly_limit_trues(conf_mask, args.max_points)

    points_3d = points_3d[conf_mask]
    points_xyf = points_xyf[conf_mask]
    points_rgb = points_rgb[conf_mask]

    # Build pycolmap reconstruction
    print("🧩 Converting to COLMAP format...")
    image_size = np.array([grid_w, grid_h])
    camera_type = "PINHOLE"  # feedforward mode supports PINHOLE here
    reconstruction = batch_np_matrix_to_pycolmap_wo_track(
        points_3d,
        points_xyf,
        points_rgb,
        extrinsic,
        intrinsic,
        image_size,
        shared_camera=False,
        camera_type=camera_type,
    )

    reconstruction = rename_colmap_recons_and_rescale_camera(
        reconstruction,
        base_image_path_list,
        original_coords.detach().cpu().numpy(),
        img_size=grid_w,
        shift_point2d_to_original_res=True,
        shared_camera=False,
    )

    # Save
    sparse_dir = args.output_path / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    print(f"💾 Saving reconstruction to {sparse_dir}")
    reconstruction.write(str(sparse_dir))

    # Also prepare images directory next to sparse for direct COLMAP import (copy only)
    if args.save_images:
        try:
            images_out_dir = args.output_path / "images"
            images_out_dir.mkdir(parents=True, exist_ok=True)
            import shutil

            num_copied = 0
            for src_path in image_path_list:
                dst_path = images_out_dir / os.path.basename(src_path)
                if dst_path.exists():
                    continue
                try:
                    shutil.copy2(src_path, dst_path)
                    num_copied += 1
                except Exception as e:
                    print(f"⚠️  Failed to copy image {src_path}: {e}")
            print(f"💾 Copied {num_copied} images to {images_out_dir}")
        except Exception as e:
            print(f"⚠️  Failed to prepare images directory: {e}")

    # Quick point cloud PLY for visualization
    try:
        trimesh.PointCloud(points_3d, colors=points_rgb).export(
            str(sparse_dir / "points.ply")
        )
        print(f"💾 Saved point cloud: {sparse_dir / 'points.ply'}")
    except Exception as e:
        print(f"⚠️  Failed to save PLY: {e}")

    print("🎉 Done.")


if __name__ == "__main__":
    with torch.no_grad():
        main()
