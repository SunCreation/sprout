import argparse as ap
import os
from time import sleep

parser = ap.ArgumentParser()
# parser.add_argument('-c','--config',default='config.yaml',type=str,help='config yaml file')
parser.add_argument('-s','--submit_dir',default='pre',type=str,help='filepath to submit')
parser.add_argument('-p','--post_dir',default='post',type=str,help='filepath to store')

args = parser.parse_args()

submit_path = args.submit_dir

post_path = args.post_dir
filelist = os.listdir(submit_path)

while filelist:
    filename = filelist.pop(0)
    print('  try:  ',os.path.join(submit_path,filename),post_path)
    os.system(f'python3 submit/submit.py -s {os.path.join(submit_path,filename)} -p {post_path}')
    sleep(3605)
    addlist = list(set(os.listdir(submit_path)) - set(filelist))
    filelist.extend(addlist)
