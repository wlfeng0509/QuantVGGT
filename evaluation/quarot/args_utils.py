import os
import argparse
from datetime import datetime
import logging
from termcolor import colored
import pprint


class SmoothRotQuantConfig:
    def __init__(self):
        # General Arguments
        self.seed = 0                  # Random seed for HuggingFace and PyTorch
        self.hf_token = None          # HuggingFace token for model access

        # Activation Quantization Arguments
        self.a_bits = 16              # Number of bits for inputs of the linear layers
        self.a_groupsize = -1         # Groupsize for activation quantization
        self.a_asym = False         # Use asymmetric activation quantization

        # Weight Quantization Arguments
        self.w_bits = 16              # Number of bits for weights of the linear layers
        self.w_groupsize = -1         # Groupsize for weight quantization
        self.w_asym = False           # Use asymmetric weight quantization✅
        self.gptq = False             # Use GPTQ for weight quantization
        self.gptq_mse = False         # Use MSE search for optimal clipping threshold
        self.percdamp = 0.01          # Percent of average Hessian diagonal for dampening
        self.act_order = False        # Use act-order in GPTQ

        # FlatQuant calibration Arguments
        self.epochs = 15             # Number of training epochs

        # 控制类型
        self.not_smooth = False            # 是否启用smooth对角矩阵
        self.not_rot = False
        self.lwc = False             # Use learnable weight clipping
        self.lac = False             # Use learnable activation clipping
        self.rv = False  

        # debug
        self.nsamples = 10           # Number of calibration data samples,
        self.cali_bsz = 1            # Batch size for FlatQuant
        self.qs_lr = 1e-3           # Learning rate for learnable transformation
        self.cali_trans = False       # ⭐Enable calibration of transformations
        self.add_diag = True         # Add per-channel scaling
        self.resume = False           # Resume from previous checkpoint
        self.save_matrix = False      # Save matrix-style parameters
        self.reload_matrix = False    # Reload matrices for evaluation
        self.matrix_path = None       # Path to pre-trained matrix parameters
        self.diag_init = "sq_style"   # Way to initialize per-channel scaling
        self.diag_alpha = 0.5         # Hyperparameter for SmoothQuant initialization
        self.warmup = False           # Warm up learning rate during training
        self.deactive_amp = False     # Disable AMP training
        self.direct_inv = False       # Use PyTorch inverse method
        self.separate_vtrans = False  # Disable vtrans transformation integration
 

        # Experiments Arguments
        self.output_dir = "./outputs" # Output directory path ⭐
        # self.output_dir = "./outputs" # Output directory path ⭐
        self.exp_name = "exp_5b_test"         # Experiment name
        self.cache_dir = None
    
    def update_nsamples(self,calib_data_num):
        self.nsamples = calib_data_num

    def update_from_args(self, wbit, abit, model_id, not_smooth, not_rot, lwc , lac,rv ,exp_name):
        self.w_bits = wbit
        self.a_bits = abit
        self.not_smooth = not_smooth
        self.not_rot = not_rot
        self.lwc = lwc
        self.lac = lac
        self.rv = rv
  
        if self.a_groupsize > -1:
            raise NotImplementedError
            
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.quantize = (self.w_bits < 16) or (self.a_bits < 16)
        
        model_name = model_id.split("/")[-1].lower()
        sym_flag = "asym" if self.w_asym else "sym"
        self.exp_dir = os.path.join(self.output_dir,  
                                   f"w{self.w_bits}a{self.a_bits}", f"{exp_name}_{model_name}_{sym_flag}")
        self.cache_dir = os.path.join(self.output_dir,  
                                   f"w{self.w_bits}a{self.a_bits}")
        
        os.makedirs(self.exp_dir, exist_ok=True)

def get_config():
    return SmoothRotQuantConfig()


def parser_gen():
    parser = argparse.ArgumentParser()

    # General Arguments
    parser.add_argument('--seed', type=int, default=0, help='Random seed for HuggingFace and PyTorch.')
    parser.add_argument('--hf_token', type=str, default=None, help='HuggingFace token for model access.')

    # Activation Quantization Arguments
    parser.add_argument('--a_bits', type=int, default=16,
                        help='''Number of bits for inputs of the linear layers.
                                This applies to all linear layers in the model, including down-projection and out-projection.''')
    parser.add_argument('--a_groupsize', type=int, default=-1, 
                        help='Groupsize for activation quantization. Note that this should be the same as w_groupsize.')
    parser.add_argument('--a_asym', action="store_true", default=False,
                        help='Use asymmetric activation quantization.')

    # Weight Quantization Arguments
    parser.add_argument('--w_bits', type=int, default=16, 
                        help='Number of bits for weights of the linear layers.')
    parser.add_argument('--w_groupsize', type=int, default=-1, 
                        help='Groupsize for weight quantization. Note that this should be the same as a_groupsize.')
    parser.add_argument('--w_asym', action="store_true", default=False,
                        help='Use asymmetric weight quantization.')
    parser.add_argument('--gptq', action="store_true", default=False,
                        help='Quantize the weights using GPTQ. If w_bits < 16 and this flag is not set, use RtN.')
    parser.add_argument('--gptq_mse', action="store_true", default=False,
                        help='''Use MSE search to find the optimal clipping threshold for weight quantization. 
                                NOTE: Do not activate while using LWC.''')
    parser.add_argument('--percdamp', type=float, default=.01,
                        help='Percent of the average Hessian diagonal to use for dampening.')
    parser.add_argument('--act_order', action="store_true", default=False,
                        help='Use act-order in GPTQ.')

    # FlatQuant calibration Arguments
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs.')
    parser.add_argument('--nsamples', type=int, default=128,
                        help='Number of calibration data samples for FlatQuant and GPTQ.')
    parser.add_argument('--cali_bsz', type=int, default=4,
                        help='Batch size for FlatQuant. Default is 4.')
    parser.add_argument("--flat_qs", type=float, default=1e-5, 
                        help='Learning rate for learnable transformation.')
    parser.add_argument("--cali_trans", default=False, action="store_true", 
                        help="Enable calibration of transformations.")
    parser.add_argument("--add_diag", default=False, action="store_true", 
                        help="Add per-channel scaling.")
    parser.add_argument("--lwc", default=False, action="store_true", 
                        help="Use learnable weight clipping.")
    parser.add_argument("--lac", default=False, action="store_true", 
                        help="Use learnable activation clipping.")
    parser.add_argument('--resume', action="store_true", default=False, 
                        help='Resume from a previous checkpoint for evaluation.')
    parser.add_argument('--save_matrix', action="store_true", default=False, 
                        help='Save the matrix-style parameters of FlatQuant.')
    parser.add_argument('--reload_matrix', action="store_true", default=False, 
                        help='Reload matrices and the inverse matrices for evaluation.')
    parser.add_argument('--matrix_path', type=str, default=None,
                        help='Path to the pre-trained matrix-style parameters of FlatQuant.')
    parser.add_argument("--diag_init", type=str, default="sq_style", choices=["sq_style", "one_style"], 
                        help='The way to initialize per-channel scaling. Default is SmoothQuant style.')
    parser.add_argument("--diag_alpha", type=float, default=0.3, 
                        help='Hyperparameter for the SmoothQuant style initialization of per-channel scaling.')
    parser.add_argument("--warmup", default=False, action="store_true", help="Warm up the learning rate during training.")
    parser.add_argument("--deactive_amp", default=False, action="store_true", help="Disable AMP training.")
    parser.add_argument("--direct_inv", default=False, action="store_true", 
                        help="Use the inverse method in PyTorch to directly get the inverse matrix rather than SVD.")
    parser.add_argument("--separate_vtrans", default=False, action="store_true", 
                        help="Disable the integration of the vtrans transformation.")
    
    # Experiments Arguments
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory path.")
    parser.add_argument("--exp_name", type=str, default="exp", help="Experiment name.")

    args = parser.parse_args()
    if args.a_groupsize > -1:
        raise NotImplementedError
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args.quantize = (args.w_bits < 16) or (args.a_bits < 16)
    # cache path
    args.cache_dir = os.path.join(args.output_dir, ".cache")
    os.makedirs(args.cache_dir, exist_ok=True)
    # output path
    args.model_name = args.model.split("/")[-1]
    args.exp_dir = os.path.join(args.output_dir, args.model_name, f"w{args.w_bits}a{args.a_bits}", args.exp_name)
    os.makedirs(args.exp_dir, exist_ok=True)
    
    logger = create_logger(args.exp_dir)
    logger.info('Arguments: ')
    logger.info(pprint.pformat(vars(args)))
    logger.info('--' * 30)
    return args, logger


def create_logger(exp_dir, dist_rank=0, name=''):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # create file handlers
    log_file = os.path.join(exp_dir, f'log_rank{dist_rank}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger