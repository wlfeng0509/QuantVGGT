import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 

import time
import torch
import argparse
import numpy as np
import open3d as o3d
import os.path as osp
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm
from collections import defaultdict
import torchvision.transforms as transforms


import torch
import numpy as np
import gzip
import json
import random
from vggt.models.vggt import VGGT
from vggt.utils.rotation import mat_to_quat
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import closed_form_inverse_se3

from quarot.utils import quarot_smooth_quant_model, set_ignore_quantize,load_qs_parameters
from quarot.args_utils import get_config

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from eval.criterion import Regr3D_t_ScaleShiftInv, L21

def get_args_parser():
    parser = argparse.ArgumentParser("3D Reconstruction evaluation", add_help=False)

    parser.add_argument("--model_name", type=str, default="VGGT")
    parser.add_argument(
        "--conf_thresh", type=float, default=0.0, help="confidence threshold"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/wmq/QuantVGGT/eval_results",
        help="value for outdir",
    )
    parser.add_argument("--size", type=int, default=518)
    parser.add_argument("--revisit", type=int, default=1, help="revisit times")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--use_proj", action="store_true")
    parser.add_argument("--kf", type=int, default=2, help="key frame")
    parser.add_argument('--dataset', type=str, default='7s', help='Dataset type: 7s or nr')
    parser.add_argument('--dataset_path', type=str, default="Null", help='Dataset path for 7s or nr')

    parser.add_argument('--class_mode', type=str, default='all', help='Only for the categories when selecting CO3D as the calibration set')
    parser.add_argument('--use_ba', action='store_true', default=False, help='Enable bundle adjustment')
    parser.add_argument('--min_num_images', type=int, default=50, help='Minimum number of images for a sequence')
    parser.add_argument('--num_frames', type=int, default=10, help='Number of frames to use')
    parser.add_argument('--co3d_dir', type=str, required=True, help='Path to CO3D dataset as the calibration set')
    parser.add_argument('--co3d_anno_dir', type=str, required=True, help='Path to CO3D annotations as the calibration set')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the VGGT model checkpoint')

    parser.add_argument('--dtype', type=str, default='quarot_w6a6', help='Data type for model inference')
    parser.add_argument('--each_nsamples', type=int, default=2, help='Number of samples to select per classb when calibration')
    parser.add_argument('--not_smooth', action='store_true', help='Disable smooth (enabled by default)')
    parser.add_argument('--not_rot', action='store_true', help='Disable rot (enabled by default)')
    parser.add_argument('--lwc', action='store_true', help='Use lwc')
    parser.add_argument('--lac', action='store_true', help='Use lac')
    parser.add_argument('--rv', action='store_true', help='Use rv')
    parser.add_argument('--exp_name', type=str, default="TEST", help='Experiment name')
    parser.add_argument('--cache_path', type=str, default=None, help='Load prepared calibration set; will be created a simple one automatically if not provided')

    parser.add_argument('--resume_qs', action='store_true', default=False, help='Load quantized model corresponding to exp_name')


    return parser

def convert_pt3d_RT_to_opencv(Rot, Trans):
    """
    Convert Point3D extrinsic matrices to OpenCV convention.

    Args:
        Rot: 3D rotation matrix in Point3D format
        Trans: 3D translation vector in Point3D format

    Returns:
        extri_opencv: 3x4 extrinsic matrix in OpenCV format
    """
    rot_pt3d = np.array(Rot)
    trans_pt3d = np.array(Trans)

    trans_pt3d[:2] *= -1
    rot_pt3d[:, :2] *= -1
    rot_pt3d = rot_pt3d.transpose(1, 0)
    extri_opencv = np.hstack((rot_pt3d, trans_pt3d[:, None]))
    return extri_opencv


def build_pair_index(N, B=1):
    """
    Build indices for all possible pairs of frames.

    Args:
        N: Number of frames
        B: Batch size

    Returns:
        i1, i2: Indices for all possible pairs
    """
    i1_, i2_ = torch.combinations(torch.arange(N), 2, with_replacement=False).unbind(-1)
    i1, i2 = [(i[None] + torch.arange(B)[:, None] * N).reshape(-1) for i in [i1_, i2_]]
    return i1, i2


def rotation_angle(rot_gt, rot_pred, batch_size=None, eps=1e-15):
    """
    Calculate rotation angle error between ground truth and predicted rotations.

    Args:
        rot_gt: Ground truth rotation matrices
        rot_pred: Predicted rotation matrices
        batch_size: Batch size for reshaping the result
        eps: Small value to avoid numerical issues

    Returns:
        Rotation angle error in degrees
    """
    q_pred = mat_to_quat(rot_pred)
    q_gt = mat_to_quat(rot_gt)

    loss_q = (1 - (q_pred * q_gt).sum(dim=1) ** 2).clamp(min=eps)
    err_q = torch.arccos(1 - 2 * loss_q)

    rel_rangle_deg = err_q * 180 / np.pi

    if batch_size is not None:
        rel_rangle_deg = rel_rangle_deg.reshape(batch_size, -1)

    return rel_rangle_deg


def translation_angle(tvec_gt, tvec_pred, batch_size=None, ambiguity=True):
    """
    Calculate translation angle error between ground truth and predicted translations.

    Args:
        tvec_gt: Ground truth translation vectors
        tvec_pred: Predicted translation vectors
        batch_size: Batch size for reshaping the result
        ambiguity: Whether to handle direction ambiguity

    Returns:
        Translation angle error in degrees
    """
    rel_tangle_deg = compare_translation_by_angle(tvec_gt, tvec_pred)
    rel_tangle_deg = rel_tangle_deg * 180.0 / np.pi

    if ambiguity:
        rel_tangle_deg = torch.min(rel_tangle_deg, (180 - rel_tangle_deg).abs())

    if batch_size is not None:
        rel_tangle_deg = rel_tangle_deg.reshape(batch_size, -1)

    return rel_tangle_deg


def compare_translation_by_angle(t_gt, t, eps=1e-15, default_err=1e6):
    """
    Normalize the translation vectors and compute the angle between them.

    Args:
        t_gt: Ground truth translation vectors
        t: Predicted translation vectors
        eps: Small value to avoid division by zero
        default_err: Default error value for invalid cases

    Returns:
        Angular error between translation vectors in radians
    """
    t_norm = torch.norm(t, dim=1, keepdim=True)
    t = t / (t_norm + eps)

    t_gt_norm = torch.norm(t_gt, dim=1, keepdim=True)
    t_gt = t_gt / (t_gt_norm + eps)

    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))

    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = default_err
    return err_t


def calculate_auc_np(r_error, t_error, max_threshold=30):
    """
    Calculate the Area Under the Curve (AUC) for the given error arrays using NumPy.

    Args:
        r_error: numpy array representing R error values (Degree)
        t_error: numpy array representing T error values (Degree)
        max_threshold: Maximum threshold value for binning the histogram

    Returns:
        AUC value and the normalized histogram
    """
    error_matrix = np.concatenate((r_error[:, None], t_error[:, None]), axis=1)
    max_errors = np.max(error_matrix, axis=1)
    bins = np.arange(max_threshold + 1)
    histogram, _ = np.histogram(max_errors, bins=bins)
    num_pairs = float(len(max_errors))
    normalized_histogram = histogram.astype(float) / num_pairs
    return np.mean(np.cumsum(normalized_histogram)), normalized_histogram


def se3_to_relative_pose_error(pred_se3, gt_se3, num_frames):
    """
    Compute rotation and translation errors between predicted and ground truth poses.

    Args:
        pred_se3: Predicted SE(3) transformations
        gt_se3: Ground truth SE(3) transformations
        num_frames: Number of frames

    Returns:
        Rotation and translation angle errors in degrees
    """
    pair_idx_i1, pair_idx_i2 = build_pair_index(num_frames)

    # Compute relative camera poses between pairs
    # We use closed_form_inverse to avoid potential numerical loss by torch.inverse()
    relative_pose_gt = closed_form_inverse_se3(gt_se3[pair_idx_i1]).bmm(
        gt_se3[pair_idx_i2]
    )
    relative_pose_pred = closed_form_inverse_se3(pred_se3[pair_idx_i1]).bmm(
        pred_se3[pair_idx_i2]
    )

    # Compute the difference in rotation and translation
    rel_rangle_deg = rotation_angle(
        relative_pose_gt[:, :3, :3], relative_pose_pred[:, :3, :3]
    )
    rel_tangle_deg = translation_angle(
        relative_pose_gt[:, :3, 3], relative_pose_pred[:, :3, 3]
    )

    return rel_rangle_deg, rel_tangle_deg


def align_to_first_camera(camera_poses):
    """
    Align all camera poses to the first camera's coordinate frame.

    Args:
        camera_poses: Tensor of shape (N, 4, 4) containing camera poses as SE3 transformations

    Returns:
        Tensor of shape (N, 4, 4) containing aligned camera poses
    """
    first_cam_extrinsic_inv = closed_form_inverse_se3(camera_poses[0][None])
    aligned_poses = torch.matmul(camera_poses, first_cam_extrinsic_inv)
    return aligned_poses


def get_simple_calibration_data( device, min_num_images,num_frames,
    co3d_dir, co3d_anno_dir, SEEN_CATEGORIES, each_nsamples=5, cache_path=None):
    total_num = 0
    calib_data = []
    for category in SEEN_CATEGORIES:
        print(f"Loading calibration annotation for {category} test set")
        annotation_file = os.path.join(co3d_anno_dir, f"{category}_test.jgz")

        print(f"annotation_file: {annotation_file}")

        try:
            with gzip.open(annotation_file, "r") as fin:
                annotation = json.loads(fin.read())
        except FileNotFoundError:
            print(f"Annotation file not found for {category}, skipping")
            continue

        seq_names = sorted(list(annotation.keys()))
        seq_names = random.sample(seq_names, min(each_nsamples, len(seq_names)))  # Random sample of sequences
        total_num += min(each_nsamples, len(seq_names))
        print(f"Processing Sequences: {seq_names}")

        for seq_name in seq_names:
            seq_data = annotation[seq_name]
            print("-" * 50)
            print(f"Processing {seq_name} for {category} test set")

            if len(seq_data) < min_num_images:  # Ensure sufficient data
                continue

            metadata = []
            for data in seq_data:
      
                if data["T"][0] + data["T"][1] + data["T"][2] > 1e5:
                    continue
                extri_opencv = convert_pt3d_RT_to_opencv(data["R"], data["T"])
                metadata.append({
                    "filepath": data["filepath"],
                    "extri": extri_opencv,
                })

            ids = np.random.choice(len(metadata), num_frames, replace=False) 
            image_names = [os.path.join(co3d_dir, metadata[i]["filepath"]) for i in ids]
            images = load_and_preprocess_images(image_names).to(device)

            input_dict = {}
            input_dict["images"] = images
            input_dict["category"] = category
            input_dict["seq_name"] = seq_name
            calib_data.append(input_dict)
            print("len(input_dict[images])",len(input_dict["images"]))
 
    print("cache_path",{cache_path},"total_num",total_num)
    if cache_path:
        print(f"Saving calibration data to: {cache_path}")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(calib_data, cache_path)

    return calib_data,len(calib_data)


def print_model_modules(model):
    for name, module in model.named_modules():
        print(f"module: {name}, type: {type(module)}")

def get_module_info(module, prefix=''):
    info = {
        'name': prefix,
        'type': str(type(module)),
        'param_count': sum(p.numel() for p in module.parameters() if p.requires_grad),
        'children': []
    }
    
    for name, child in module.named_children():
        child_prefix = f"{prefix}.{name}" if prefix else name
        child_info = get_module_info(child, child_prefix)
        info['children'].append(child_info)
    
    return info

def save_model_structure_to_json(model, output_path):
    model_info = get_module_info(model)
    with open(output_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    print(f"save_model_structure_to_json {output_path}")
    exit()


def set_random_seeds(seed):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def process_sequence(model, seq_name, seq_data, category, co3d_dir, min_num_images, num_frames, use_ba, device, dtype):
    """
    Process a single sequence and compute pose errors.

    Args:
        model: VGGT model
        seq_name: Sequence name
        seq_data: Sequence data
        category: Category name
        co3d_dir: CO3D dataset directory
        min_num_images: Minimum number of images required
        num_frames: Number of frames to sample
        use_ba: Whether to use bundle adjustment
        device: Device to run on
        dtype: Data type for model inference

    Returns:
        rError: Rotation errors
        tError: Translation errors
    """
    if len(seq_data) < min_num_images:
        return None, None

    metadata = []
    for data in seq_data:
        if data["T"][0] + data["T"][1] + data["T"][2] > 1e5:
            return None, None
        extri_opencv = convert_pt3d_RT_to_opencv(data["R"], data["T"])
        metadata.append({
            "filepath": data["filepath"],
            "extri": extri_opencv,
        })

    ids = np.random.choice(len(metadata), num_frames, replace=False)
    print("Image ids", ids)

    image_names = [os.path.join(co3d_dir, metadata[i]["filepath"]) for i in ids]
    gt_extri = [np.array(metadata[i]["extri"]) for i in ids]
    gt_extri = np.stack(gt_extri, axis=0)
    images = load_and_preprocess_images(image_names).to(device)

    if use_ba:
        try:
            pred_extrinsic = run_vggt_with_ba(model, images, image_names=image_names, dtype=dtype)
        except Exception as e:
            print(f"BA failed with error: {e}. Falling back to standard VGGT inference.")
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype):
                   
                    predictions = model(images)
                     
            with torch.cuda.amp.autocast(dtype=torch.float64):
                extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:]) 
                pred_extrinsic = extrinsic[0]
    else:
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)
        with torch.cuda.amp.autocast(dtype=torch.float64):
            extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
            pred_extrinsic = extrinsic[0]

    with torch.cuda.amp.autocast(dtype=torch.float64):
        gt_extrinsic = torch.from_numpy(gt_extri).to(device)
        add_row = torch.tensor([0, 0, 0, 1], device=device).expand(pred_extrinsic.size(0), 1, 4)

        pred_se3 = torch.cat((pred_extrinsic, add_row), dim=1)
        gt_se3 = torch.cat((gt_extrinsic, add_row), dim=1)

        # Set the coordinate of the first camera as the coordinate of the world
        # NOTE: DO NOT REMOVE THIS UNLESS YOU KNOW WHAT YOU ARE DOING
        pred_se3 = align_to_first_camera(pred_se3)
        gt_se3 = align_to_first_camera(gt_se3)

        rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(pred_se3, gt_se3, num_frames)

        Racc_5 = (rel_rangle_deg < 5).float().mean().item()
        Tacc_5 = (rel_tangle_deg < 5).float().mean().item()

        print(f"{category} sequence {seq_name} R_ACC@5: {Racc_5:.4f}")
        print(f"{category} sequence {seq_name} T_ACC@5: {Tacc_5:.4f}")

        return rel_rangle_deg.cpu().numpy(), rel_tangle_deg.cpu().numpy()
    


def load_model(device, each_nsamples = 0,min_num_images=0, num_frames=0,category=None,co3d_anno_dir=None,co3d_dir=None,
               model_path=None, dtype=None, compile=False, fuse_qkv=False, resume_qs=False, use_gptq=False, resume_gptq=False,debug_mode='all',
               not_smooth=False, not_rot=False, lwc=True,lac=True,rv=True,exp_name=None,cache_path = None):
 
    print("Initializing and loading VGGT model...")
    model = VGGT()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model = model.to(device)


    calib_data = None
    if dtype in['raw']:
        if os.path.exists(cache_path):
            print(f"✅ Calib_data is exist: {cache_path}")
            calib_data = torch.load(cache_path)

        elif not os.path.exists(cache_path):
            print(f"⚠️ {cache_path} not exist! Random Select Calibation Data")
            calib_data, calib_data_num = get_simple_calibration_data(device, min_num_images,num_frames,
                                                co3d_dir,co3d_anno_dir, category, each_nsamples, cache_path=cache_path)

        return model,calib_data


    if dtype in ["quarot_w4a4","quarot_w6a6","quarot_w8a8"]:
        import re
        wbit = re.search(r'w(\d+)', dtype)
        abit = re.search(r'a(\d+)', dtype)

        if wbit:
            wbit = int(wbit.group(1)) 
        if abit:
            abit = int(abit.group(1)) 


        config = get_config() 
        config.update_from_args(wbit=wbit, abit=abit, not_smooth=not_smooth, not_rot=not_rot, lwc=lwc, lac=lac, rv=rv, model_id=model_path, exp_name=exp_name)
        
        print(f"W-Bit:{wbit},\tA-Bit:{abit}")
        print(f"Not_Rot:{config.not_rot},\tNot_Smooth:{config.not_smooth},\tLWC:{config.lwc},\tLAC:{config.lac},\tRV:{config.rv}")
       
        if cache_path is not None:
            cache_path = cache_path
        else:
            cache_path = f"{config.cache_dir}/{model_path.replace('/', '_')}_{debug_mode}_{each_nsamples}_calib_data.pt"
        print("exp_dir:",config.exp_dir)
        print("cache_dir:",config.cache_dir)
        print("cache_path:",cache_path)

        if os.path.exists(cache_path) and not resume_qs:
            print(f"Calib data is exist: {cache_path}")
            calib_data = torch.load(cache_path)
            config.update_nsamples(len(calib_data))
        
        elif not os.path.exists(cache_path) and not resume_qs :
            print(f"⚠️ {cache_path} not exist! Random Select Calibation Data")
            calib_data, calib_data_num = get_simple_calibration_data(device, min_num_images,num_frames,
                                                 co3d_dir,co3d_anno_dir, category, each_nsamples, cache_path=cache_path)
         
            config.update_nsamples(calib_data_num)
            print(f"⚠️ Calibation nums = {len(calib_data)} ")
        else:
            calib_data = None

        quarot_smooth_quant_model(config,model,calib_data, wbit=wbit, abit=abit,
                            resume_qs=resume_qs, use_gptq=use_gptq, resume_gptq=resume_gptq,
                            model_id=model_path, exp_name=exp_name)
        
        model.to(device)

    if compile:
        model.to(memory_format=torch.channels_last)
        model = torch.compile(model, mode="max-autotune", fullgraph=True)
    
    return model,calib_data



def main(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16


    from eval.data import SevenScenes, NRGBD
    from eval.utils import accuracy, completion
    if args.size == 512:
        resolution = (512, 384)
    elif args.size == 224:
        resolution = 224
    elif args.size == 518:
        resolution = (518, 392)
    else:
        raise NotImplementedError
    if args.dataset == "7s":
        datasets_all = {
            "7scenes": SevenScenes(
                split="test",
                ROOT = args.dataset_path,
                resolution=resolution,
                num_seq=1,
                full_video=True,
                kf_every=args.kf,
            ), 
        }
    elif args.dataset == "nr":
        datasets_all = {
            "NRGBD": NRGBD(
            split="test",
            ROOT=args.dataset_path,
            resolution=resolution,
            num_seq=1,
            full_video=True,
            kf_every=args.kf,
        ),
        }

    SEEN_CATEGORIES = ["apple"]
    if args.class_mode == "apple":
        SEEN_CATEGORIES = ["apple"]
    elif args.class_mode == "five":
        SEEN_CATEGORIES = ["apple","bicycle", "bottle", "bowl","handbag"]
    elif args.class_mode == "more":
        SEEN_CATEGORIES = ["apple","bicycle", "bottle", "bowl","handbag","carrot","cellphone", "motorcycle","umbrella","toaster"]
    elif args.class_mode == "all":
        SEEN_CATEGORIES = [
        "apple", "backpack", "banana", "baseballbat", "baseballglove",
        "bench", "bicycle", "bottle", "bowl", "broccoli",
        "cake", "car", "carrot", "cellphone", "chair",
        "cup", "donut", "hairdryer", "handbag", "hydrant",
        "keyboard", "laptop", "microwave", "motorcycle", "mouse",
        "orange", "parkingmeter", "pizza", "plant", "stopsign",
        "teddybear", "toaster", "toilet", "toybus", "toyplane",
        "toytrain", "toytruck", "tv", "umbrella", "vase", "wineglass",
    ]


    model , _ = load_model(device,args.each_nsamples,
                        min_num_images =args.min_num_images, num_frames =  args.num_frames,category =SEEN_CATEGORIES,co3d_anno_dir = args.co3d_anno_dir,co3d_dir = args.co3d_dir, 
                            model_path=args.model_path, dtype=args.dtype,
                            compile= False ,fuse_qkv= False,
                            resume_qs=args.resume_qs, use_gptq= False,
                            resume_gptq= False,debug_mode=args.class_mode,
                            not_smooth=args.not_smooth, not_rot=args.not_rot,lac=args.lac,lwc=args.lwc,rv=args.rv,exp_name=args.exp_name,cache_path=args.cache_path)
  
    
    model = VGGT()
    ckpt = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(
        ckpt, strict=False
    )  
    model.eval()
    model = model.to(device)
    del ckpt

    # evaluation
    output_path = osp.join(args.output_dir, f"{args.kf}")
    os.makedirs(output_path, exist_ok=True)
    print("-----------------------")
    print("✅ Evaluation Start! ")
    print("✅ Result path:",output_path)
    criterion = Regr3D_t_ScaleShiftInv(L21, norm_mode=False, gt_scale=True)
    with torch.no_grad():
        for name_data, dataset in datasets_all.items():
            save_path = osp.join(osp.join(args.output_dir, f"{args.kf}"), name_data ,args.exp_name)
            os.makedirs(save_path, exist_ok=True)
            log_file = osp.join(save_path, "logs.txt")

            acc_all = 0
            acc_all_med = 0
            comp_all = 0
            comp_all_med = 0
            nc1_all = 0
            nc1_all_med = 0
            nc2_all = 0
            nc2_all_med = 0
            scene_infer_times = defaultdict(list)

            for data_idx in tqdm(range(len(dataset))):
                batch = default_collate([dataset[data_idx]])
                ignore_keys = set(
                    [
                        "depthmap", 
                        "dataset",
                        "label",
                        "instance",
                        "idx",
                        "true_shape",
                        "rng",
                    ]
                )
                for view in batch:
                    for name in view.keys():  # pseudo_focal
                        if name in ignore_keys:
                            continue
                        if isinstance(view[name], tuple) or isinstance(
                            view[name], list
                        ):
                            view[name] = [
                                x.to(device, non_blocking=True) for x in view[name]
                            ]
                        else:
                            view[name] = view[name].to(device, non_blocking=True)

                pts_all = []
                pts_gt_all = []
                images_all = []
                masks_all = []
                conf_all = []
                in_camera1 = None


                with torch.cuda.amp.autocast(dtype=dtype):
                    if isinstance(batch, dict) and "img" in batch:
                        batch["img"] = (batch["img"] + 1.0) / 2.0
                    elif isinstance(batch, list) and all(
                        isinstance(v, dict) and "img" in v for v in batch
                    ):
                        for view in batch:
                            view["img"] = (view["img"] + 1.0) / 2.0
                        # Gather all `img` tensors into a single tensor of shape [N, C, H, W]
                        imgs_tensor = torch.cat([v["img"] for v in batch], dim=0)


    
                with torch.cuda.amp.autocast(dtype=dtype):
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(dtype=dtype):
                            preds = model(imgs_tensor) 
                
    
                    predictions = preds
                    views = batch

                    if "pose_enc" in predictions:
                        B, S = predictions["pose_enc"].shape[:2]
                    elif "world_points" in predictions:
                        B, S = predictions["world_points"].shape[:2]
                    else:
                        raise KeyError(
                            "predictions is missing a key to infer sequence length"
                        )

                    ress = []
                    for s in range(S):
                        res = {
                            "pts3d_in_other_view": predictions["world_points"][:, s],
                            "conf": predictions["world_points_conf"][:, s],
                            "depth": predictions["depth"][:, s],
                            "depth_conf": predictions["depth_conf"][:, s],
                            "camera_pose": predictions["pose_enc"][:, s, :],
                        }
                       

                        if (
                            isinstance(views, list)
                            and s < len(views)
                            and "valid_mask" in views[s]
                        ):
                            res["valid_mask"] = views[s]["valid_mask"]
                        if "track" in predictions:
                            res.update(
                                {
                                    "track": predictions["track"][:, s],
                                    "vis": (
                                        predictions.get("vis", None)[:, s]
                                        if "vis" in predictions
                                        else None
                                    ),
                                    "track_conf": (
                                        predictions.get("conf", None)[:, s]
                                        if "conf" in predictions
                                        else None
                                    ),
                                }
                            )
                        
                   
                        ress.append(res)
                    preds = ress   
                    valid_length = len(preds) // args.revisit
                    if args.revisit > 1:
                        preds = preds[-valid_length:]
                        batch = batch[-valid_length:]
                    print(f"Evaluation for {name_data} {data_idx+1}/{len(dataset)}")
                    gt_pts, pred_pts, gt_factor, pr_factor, masks, monitoring = (
                        criterion.get_all_pts3d_t(batch, preds)
                    )
                
                    in_camera1 = None
                    pts_all = []
                    pts_gt_all = [] 
                    images_all = []
                    masks_all = [] 
                    conf_all = [] 

                    for j, view in enumerate(batch):
                        if in_camera1 is None:
                            in_camera1 = view["camera_pose"][0].cpu()

                        image = view["img"].permute(0, 2, 3, 1).cpu().numpy()[0]
                        mask = view["valid_mask"].cpu().numpy()[0]

                        pts = pred_pts[j].cpu().numpy()[0]
                        conf = preds[j]["conf"].cpu().data.numpy()[0]

                        pts_gt = gt_pts[j].detach().cpu().numpy()[0]

                        H, W = image.shape[:2]
                        cx = W // 2
                        cy = H // 2
                        l, t = cx - 112, cy - 112
                        r, b = cx + 112, cy + 112
                        image = image[t:b, l:r]
                        mask = mask[t:b, l:r]
                        pts = pts[t:b, l:r]
                        pts_gt = pts_gt[t:b, l:r]

                        images_all.append(image[None, ...])
                        pts_all.append(pts[None, ...])
                        pts_gt_all.append(pts_gt[None, ...])
                        masks_all.append(mask[None, ...])
                        conf_all.append(conf[None, ...])

                images_all = np.concatenate(images_all, axis=0)
                pts_all = np.concatenate(pts_all, axis=0)
              
                pts_gt_all = np.concatenate(pts_gt_all, axis=0)
                masks_all = np.concatenate(masks_all, axis=0)
                scene_id = view["label"][0].rsplit("/", 1)[0]

                save_params = {}

                save_params["images_all"] = images_all
                save_params["pts_all"] = pts_all
                save_params["pts_gt_all"] = pts_gt_all
                save_params["masks_all"] = masks_all

                pts_all_masked = pts_all[masks_all > 0]
                pts_gt_all_masked = pts_gt_all[masks_all > 0]
                images_all_masked = images_all[masks_all > 0]

                mask = np.isfinite(pts_all_masked)
                pts_all_masked = pts_all_masked[mask]

                mask_gt = np.isfinite(pts_gt_all_masked)
                pts_gt_all_masked = pts_gt_all_masked[mask_gt]
                images_all_masked = images_all_masked[mask]

                # Reshape to point cloud (N, 3) before sampling
                pts_all_masked = pts_all_masked.reshape(-1, 3)
                pts_gt_all_masked = pts_gt_all_masked.reshape(-1, 3)
                images_all_masked = images_all_masked.reshape(-1, 3)

                # If number of points exceeds threshold, sample by points
                if pts_all_masked.shape[0] > 999999:
                    sample_indices = np.random.choice(
                        pts_all_masked.shape[0], 999999, replace=False
                    )
                    pts_all_masked = pts_all_masked[sample_indices]
                    images_all_masked = images_all_masked[sample_indices]

                # Apply the same sampling to GT point cloud
                if pts_gt_all_masked.shape[0] > 999999:
                    sample_indices_gt = np.random.choice(
                        pts_gt_all_masked.shape[0], 999999, replace=False
                    )
                    pts_gt_all_masked = pts_gt_all_masked[sample_indices_gt]

                if args.use_proj:

                    def umeyama_alignment(
                        src: np.ndarray, dst: np.ndarray, with_scale: bool = True
                    ):
                        assert src.shape == dst.shape
                        N, dim = src.shape

                        mu_src = src.mean(axis=0)
                        mu_dst = dst.mean(axis=0)
                        src_c = src - mu_src
                        dst_c = dst - mu_dst

                        Sigma = dst_c.T @ src_c / N  # (3,3)

                        U, D, Vt = np.linalg.svd(Sigma)

                        S = np.eye(dim)
                        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
                            S[-1, -1] = -1

                        R = U @ S @ Vt

                        if with_scale:
                            var_src = (src_c**2).sum() / N
                            s = (D * S.diagonal()).sum() / var_src
                        else:
                            s = 1.0

                        t = mu_dst - s * R @ mu_src

                        return s, R, t

                    pts_all_masked = pts_all_masked.reshape(-1, 3)
                    pts_gt_all_masked = pts_gt_all_masked.reshape(-1, 3)
                    s, R, t = umeyama_alignment(
                        pts_all_masked, pts_gt_all_masked, with_scale=True
                    )
                    pts_all_aligned = (s * (R @ pts_all_masked.T)).T + t  # (N,3)
                    pts_all_masked = pts_all_aligned

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts_all_masked)
                pcd.colors = o3d.utility.Vector3dVector(images_all_masked)

                pcd_gt = o3d.geometry.PointCloud()
                pcd_gt.points = o3d.utility.Vector3dVector(pts_gt_all_masked)
                pcd_gt.colors = o3d.utility.Vector3dVector(images_all_masked)

                trans_init = np.eye(4)
                threshold = 0.1
                reg_p2p = o3d.pipelines.registration.registration_icp(
                    pcd,
                    pcd_gt,
                    threshold,
                    trans_init,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                )

                transformation = reg_p2p.transformation

                pcd = pcd.transform(transformation)
                pcd.estimate_normals()
                pcd_gt.estimate_normals()

                gt_normal = np.asarray(pcd_gt.normals)
                pred_normal = np.asarray(pcd.normals)

                acc, acc_med, nc1, nc1_med = accuracy(
                    pcd_gt.points, pcd.points, gt_normal, pred_normal
                )

                print(acc, acc_med, nc1, nc1_med) 
                comp, comp_med, nc2, nc2_med = completion(
                    pcd_gt.points, pcd.points, gt_normal, pred_normal
                )
                print(
                    f"Idx: {scene_id}, Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}"
                )
                print(
                    f"Idx: {scene_id}, Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}",
                    file=open(log_file, "a"),
                )

                acc_all += acc
                comp_all += comp
                nc1_all += nc1
                nc2_all += nc2

                acc_all_med += acc_med
                comp_all_med += comp_med
                nc1_all_med += nc1_med
                nc2_all_med += nc2_med

                # release cuda memory
                torch.cuda.empty_cache()

            # Get depth from pcd and run TSDFusion
            to_write = ""
            # Read the log file
            if os.path.exists(osp.join(save_path, "logs.txt")):
                with open(osp.join(save_path, "logs.txt"), "r") as f_sub:
                    to_write += f_sub.read()

            with open(osp.join(save_path, f"logs_all.txt"), "w") as f:
                log_data = to_write
                metrics = defaultdict(list)
                for line in log_data.strip().split("\n"):
                    match = regex.match(line)
                    if match:
                        data = match.groupdict()
                        # Exclude 'scene_id' from metrics as it's an identifier
                        for key, value in data.items():
                            if key != "scene_id":
                                metrics[key].append(float(value))
                        metrics["nc"].append(
                            (float(data["nc1"]) + float(data["nc2"])) / 2
                        )
                        metrics["nc_med"].append(
                            (float(data["nc1_med"]) + float(data["nc2_med"])) / 2
                        )
                mean_metrics = {
                    metric: sum(values) / len(values)
                    for metric, values in metrics.items()
                }

                c_name = "mean"
                print_str = f"{c_name.ljust(20)}: "
                for m_name in mean_metrics:
                    print_num = np.mean(mean_metrics[m_name])
                    print_str = print_str + f"{m_name}: {print_num:.3f} | "
                print_str = print_str + "\n"
                # Summarize per-scene average inference time
                time_lines = []
                for sid, times in scene_infer_times.items():
                    if len(times) > 0:
                        time_lines.append(
                            f"Idx: {sid}, Time_avg_ms: {np.mean(times):.2f}"
                        )
                time_block = "\n".join(time_lines) + (
                    "\n" if len(time_lines) > 0 else ""
                )

                f.write(to_write + time_block + print_str)
            print(f"✅ Evaluation End ,save in {log_file}")
            print("-----------------------------------")

from collections import defaultdict
import re

pattern = r"""
    Idx:\s*(?P<scene_id>[^,]+),\s*
    Acc:\s*(?P<acc>[^,]+),\s*
    Comp:\s*(?P<comp>[^,]+),\s*
    NC1:\s*(?P<nc1>[^,]+),\s*
    NC2:\s*(?P<nc2>[^,]+)\s*-\s*
    Acc_med:\s*(?P<acc_med>[^,]+),\s*
    Compc_med:\s*(?P<comp_med>[^,]+),\s*
    NC1c_med:\s*(?P<nc1_med>[^,]+),\s*
    NC2c_med:\s*(?P<nc2_med>[^,]+)
"""

regex = re.compile(pattern, re.VERBOSE)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
