from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
import os
import natsort
from argparse import ArgumentParser
import shutil
# lightning deepspeed has saved a directory instead of a file

##### Please modify this part #####

parser = ArgumentParser()
parser.add_argument('--ckpt', type=str, default='ckpt/exaone_bi_8.8b_lr1e-5_bs32_chatgpt/')

args = parser.parse_args()

lst = os.listdir(args.ckpt)

lst=natsort.natsorted(lst)
for i,l in enumerate(lst):
    file = args.ckpt+l
    out = args.ckpt+'step%d.ckpt'%(i)
    # print()
    try:
        convert_zero_checkpoint_to_fp32_state_dict(file, out)
        print("%s->%s"%(file,out))
        shutil.rmtree(file)
        print("%s is deleted"%file)
    except:
        print("can't convert %s"%file)