import os
from time import sleep
import argparse as ap

parser = ap.ArgumentParser()
parser.add_argument('-w','--wait_time',default=600,type=int,help='waiting time')
args = parser.parse_args()

time = args.wait_time
print(f"I'll sleep {time//60} minute")
sleep(time)

os.system('python3 main_xla.py')
