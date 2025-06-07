import argparse
# Training settings
parser = argparse.ArgumentParser(description="Super-Resolution")
parser.add_argument("--upscale_factor", default=8, type=int, help="super resolution upscale factor")
parser.add_argument('--seed', type=int, default=1,  help='random seed (default: 1)')
parser.add_argument("--batchSize", type=int, default=16, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=60, help="maximum number of epochs to train")
parser.add_argument("--show", action="store_true", help="show Tensorboard")
parser.add_argument('--method', type=str, default='ours')
parser.add_argument('--pretrained_model_path', type=str, default=None)
parser.add_argument("--lr", type=int, default=1.0e-4, help="lerning rate")
parser.add_argument("--cuda", help="Use cuda",default=True)
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--threads", type=int, default=1, help="number of threads for dataloader to usea")
parser.add_argument("--resume", default="", type=str, help="")
parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("----datasetName",default="Chikusei",type=str,help="data name")
# Network settings

parser.add_argument('--n_feats', type=int, default=80, help='number of feature maps')
parser.add_argument('--n_colors', type=int, default=128, help='number of band')
parser.add_argument('--res_scale', type=int, default=1, help='number of band')

parser.add_argument("--use_share", type=bool, default=True, help="f_share, default set to 1")
parser.add_argument("--end_epoch", type=int, default=60, help="number of epochs")
parser.add_argument('--alpha', type=float, default=0.1, help='hybrid loss coefficient')

# Test image
parser.add_argument('--model_name', default='./checkpoint/Chikusei_model_8_epoch_60.pth', type=str, help='super resolution model name ')
opt = parser.parse_args()

# Chikusei