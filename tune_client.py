from tuning_utils import tuning_main, run_seperate_script, gen_hyperopt_space,\
    gen_optunity_space
import sys
import os
import pickle
import argparse
import json
from multiprocessing.connection import Client
import pprint
from datetime import timedelta, datetime
import logging

start_time = datetime.now().strftime("%Y%m%d%H%M%S")
log_file = 'hparams-{}.log'.format(start_time)
logging.basicConfig(filename=log_file,
    format='[%(levelname)s] %(message)s', level=logging.INFO)
# also logggin to console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
logging.getLogger('').addHandler(console)

logging.info("log store to: %s", log_file)

pp = pprint.PrettyPrinter(indent=2)
INT_HYPER = []
domain_socket = "/tmp/tuneconn"
conn_authkey = b'physionet'

def tune_start():
    args = arg_config()
    space_file = args.space
    with open(space_file, "r") as f:
        logging.info("space config: %s", space_file)
        arg_space = json.load(f)
    for i in arg_space:
        if i["type"] == "int":
            INT_HYPER.append(i["name"])
    if args.tool == "optunity":
        tune_optunity(arg_space, args.num)
    elif args.tool == "hyperopt":
        tune_hyperopt(arg_space, args.num)
    else:
        logging.info("the tuning tool you specified is not supported currently.")
    
    with Client(domain_socket, authkey=conn_authkey) as conn:
        conn.send("kill connection!")

def tune_optunity(arg_space, num):
    space = gen_optunity_space(arg_space)
    tuning_main("optunity", obf, space, num)

def tune_hyperopt(arg_space, num):
    space = gen_hyperopt_space(arg_space)
    tuning_main("hyperopt", obf, space, num)

def obf(**kwargs):
    """run `train.py` and pass necessary flags into it"""
    hparams = {}
    for k, v in iter(kwargs.items()):
        if k in INT_HYPER:
            val = round(v)
        else:
            val = v
        hparams[k] = val
    with Client(domain_socket, authkey=conn_authkey) as conn:
        logging.info("current hparams:\n%s", pp.pformat(hparams))
        conn.send(hparams)
        while True:
            res_sig = conn.recv()
            if res_sig == "res":
                res = conn.recv()
                logging.info("result: %s", res)
                conn.recv() # info
                logging.info(conn.recv())
                return res
            

def arg_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", default="optunity", help="specify the tuning tool \
    (optunity/hyperopt), default is optunity")
    parser.add_argument("--num", default=20, help="the number of tuning iteration", type=int)
    parser.add_argument("--space", default="space.json", help="the space config file", type=str)
    args = parser.parse_args()
    return args

if __name__=="__main__":
    tune_start()
