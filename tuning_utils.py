import pickle
import csv
import time
import optunity
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from subprocess import run,CalledProcessError
from datetime import datetime, timedelta
import logging


def tuning_main(tuning_tool, obf, space, evals_num, solver_name="particle swarm", 
    optunity_constraints=None):
    """
    The main function to start tuning process

    :param tuning_tool: string, choose the tuning tool(optunity/hyperopt)
    :param obf: the objective function user want to minimize
    :param space: tuning tool will search on this space
    :param evals_num: int, tuning tool evaluate number
    :param solver_name: string, the solver user want optunity to use
    :param optunity_constraints: list. if there is a space constraint when doing 
        search with optunity, user can pass the contraint(s) in this parameter
    """
    start_time = datetime.now()
    result_filename = "{}_result-{}.csv".format(tuning_tool, start_time.strftime("%Y%m%d%H%M%S"))
    if tuning_tool == "optunity":
        logging.info("start using optunity...")
        _optunity_exec(obf, evals_num, space, solver_name, optunity_constraints, result_filename)
    elif tuning_tool == "hyperopt":
        logging.info("start using hyperopt...")
        _hyperopt_exec(obf, evals_num, space, result_filename)
    else:
        logging.info("the tuning tool you specify is not supported")
    end_time = datetime.now()
    exec_time = end_time - start_time
    logging.info("result store to: %s", result_filename)
    logging.info("execution time: %s", str(exec_time))

def _optunity_exec(obf, evals_num, space, solver_name, 
        optunity_constraints, result_filename):
    if optunity_constraints is None:
        constraint_list = []
    else:
        constraint_list = optunity_constraints
    obf_con = optunity.wrap_constraints(obf,default = 1000000,custom=constraint_list)
    optimal_pars, info, _ = \
        optunity.minimize(obf_con, num_evals=evals_num, **space, solver_name=solver_name)

    df = optunity.call_log2dataframe(info.call_log)
    #df = df.sort_values('value', ascending=False)
    logging.info("\n%s", df)
    df.to_csv(result_filename)

def _hyperopt_store_result(trials, space, result_filename):
    """output trials to csv and pickle"""
    with open("hyperopt_result.pickle", "wb") as f:
        pickle.dump(trials.trials, f)
    space_keys = tuple(str(i) for i in iter(space.keys()))
    headers = ("tid", "loss") + space_keys
    data = []
    for i_t in trials:
        if i_t['result']['loss'] < 500:
            t_row = (i_t['tid'], i_t['result']['loss']) + \
                    tuple(i_t['misc']['vals'][h_param][0] for h_param in space_keys)
            data.append(t_row)
    with open(result_filename, "w") as f:
        f_csv=csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(data)

def _ob_f4hyperopt_wrapper(func):
    """
    the wrapper to make objective function return the object hyperopt need
    """
    def wrapper(params):
        return {'loss': func(**params), 'status': STATUS_OK}
    return wrapper

def _hyperopt_exec(obf, evals_num,space, result_filename):
    # create trial instance, which can store info during search
    trials = Trials()
    ob_f4hyperopt = _ob_f4hyperopt_wrapper(obf)
    # run fmin to find the optimal hyperparameter
    best = fmin(ob_f4hyperopt, space, algo=tpe.suggest, max_evals=evals_num, trials=trials)
    logging.info("best: ")
    logging.info(best)

    _hyperopt_store_result(trials, space, result_filename)

def gen_optunity_space(space):
    """generate optunity space"""
    space4optunity = {}
    for i in space:
        space4optunity[i["name"]] = [i["lower"], i["upper"]]
    return space4optunity

def gen_hyperopt_space(space):
    """generate optunity space"""
    space4hyperopt = {}
    for i in space:
        if i["type"] == "int":
            space4hyperopt[i["name"]] = hp.quniform(i["name"], i["lower"], i["upper"], 1)
        else:
            space4hyperopt[i["name"]] = hp.uniform(i["name"], i["lower"], i["upper"])
    return space4hyperopt

def run_seperate_script(cmd_string,max_run_num=10,sleep_sec=10):
    """
    The utility function to run seperate script using run in subprocess module

    :param cmd_string: the cmd user want to run in another process
    :param max_run_num: the maxmum try again number when the cmd is failed,
        default: 10 times
    :param sleep_sec: sleep second when the last run, default: 10 seconds
    :return: the status of the running command. True means success, otherwise fail.
    """
    run_num = 0
    while True:
        try:
            run(cmd_string, check=True)
            return True
        except CalledProcessError as e:
            cmd_name = " ".join(cmd_string[:2])
            logging.info("Error happened when running {cmd_name}:".format(cmd_name=cmd_name))
            logging.info(str(e.output))
            if run_num < max_run_num:
                time.sleep(sleep_sec)
                run_num+=1
                logging.info("Try again, #{n}!".format(n=run_num))
                continue
            else:
                return False
        break
