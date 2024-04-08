import glob
import os
import sys
import time
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import LinearLocator
from matplotlib import cm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from smt.surrogate_models import KPLSK,KRG
import smt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF as RBFsk
from sklearn.cluster import KMeans
from scipy.stats import pearsonr
from pyoptsparse import IPOPT,SLSQP, Optimization,OPT
import warnings
from scipy.optimize import minimize
from pulp import LpProblem, LpVariable, LpMinimize, LpStatus, value
from deap import base, creator, tools, algorithms
import random


import numpy as np
import pandas as pd 
import multiprocessing as mp 
import pickle as pkl
import matplotlib.pyplot as plt

import openmdao.api as om
from weis.aeroelasticse import FileTools

def load_OMsql(log):
    print('loading {}'.format(log))
    cr = om.CaseReader(log)
    rec_data = {}
    driver_cases = cr.list_cases('driver')
    cases = cr.get_cases('driver')
    for case in cases:
        for key in case.outputs.keys():
            if key not in rec_data:
                rec_data[key] = []
            rec_data[key].append(case[key])
        
    return rec_data


if __name__ == '__main__':
    
    # Multiprocssing?
    post_multi = False

    # sql outfile directory
    run_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))# os.path.dirname(os.path.realpath(__file__))   
    #output_dir = os.path.join(run_dir, "outputs/15_RAFT_optimization_freq_step1_Cntr_Clmns_and_Outr_Clmns_Reduced_included") #Design of Experiment
    output_dir = os.path.join(run_dir, "outputs/15_RAFT_Control_params") # 15_RAFT_optimization_freq_step1_clmnSpacing_sideColmnsdiam_lowPntoon_diam") #opt coByla
    doe_logs = glob.glob(os.path.join(output_dir,'log_opt.sql*'))
    if len(doe_logs) < 1:
        raise FileExistsError('No output logs to post process!')
        
    # Remove the 'meta' log
    for idx, log in enumerate(doe_logs):
        if 'meta' in log:
            doe_logs.pop(idx)

    # run multiprocessing
    if post_multi:
        cores = mp.cpu_count()
        pool = mp.Pool(min(len(doe_logs), cores))

        # load sql file
        outdata = pool.map(load_OMsql, doe_logs)
        pool.close()
        pool.join()
    # no multiprocessing
    else:
        outdata = [load_OMsql(log) for log in doe_logs]

    collected_data = {}
    for data in outdata:
        for key in data.keys():
            if key not in collected_data.keys():
                collected_data[key] = []

            for key_idx, _ in enumerate(data[key]):
                if key=='costs.turbine_number':
                    data[key][key_idx]=int(data[key][key_idx])
                if isinstance(data[key][key_idx], int):
                    collected_data[key].append(np.array(data[key][key_idx]))
                elif len(data[key][key_idx]) == 1:
                    try:
                        collected_data[key].append(np.array(data[key][key_idx][0]))
                    except:
                        collected_data[key].append(np.array(data[key][key_idx]))
                else:
                    collected_data[key].append(np.array(data[key][key_idx]))

    df = pd.DataFrame.from_dict(collected_data)

    #-------------------------------------Design Variables------------------------------------
    Kp_float=np.squeeze(outdata[0]['tune_rosco_ivc.Kp_float'])
    omega_pc=np.squeeze(outdata[0]['tune_rosco_ivc.omega_pc'])
    omega_vs=np.squeeze(outdata[0]['tune_rosco_ivc.omega_vs'])
    ps_percent=np.squeeze(outdata[0]['tune_rosco_ivc.ps_percent'])
    ptfm_freq=np.squeeze(outdata[0]['tune_rosco_ivc.ptfm_freq'])
    zeta_pc=np.squeeze(outdata[0]['tune_rosco_ivc.zeta_pc'])
    zeta_vs=np.squeeze(outdata[0]['tune_rosco_ivc.zeta_vs'])

    #--------------------------------------------------------------------------------------

    #-------------------------------------Constraints-------------------------------------
    Max_PtfmPitch_vec=np.squeeze(outdata[0]['raft.Max_PtfmPitch'])
    max_nac_accel_vec=np.squeeze(outdata[0]['raft.max_nac_accel'])
    surge_period_vec=np.squeeze(outdata[0]['raft.surge_period'])
    sway_period_vec=np.squeeze(outdata[0]['raft.sway_period'])
    heave_period_vec=np.squeeze(outdata[0]['raft.heave_period'])
    roll_period_vec=np.squeeze(outdata[0]['raft.roll_period'])
    pitch_period_vec=np.squeeze(outdata[0]['raft.pitch_period'])
    yaw_period_vec=np.squeeze(outdata[0]['raft.yaw_period'])
    tower_structural_frequencies_vec=np.squeeze(outdata[0]['towerse.tower.structural_frequencies'])
    tower_fore_aft_freqs_vec=np.squeeze(outdata[0]['towerse.tower.fore_aft_freqs'])
    tower_side_side_freqs_vec=np.squeeze(outdata[0]['towerse.tower.side_side_freqs'])
    Std_PtfmPitch_vec=np.squeeze(outdata[0]['raft.Std_PtfmPitch'])
    raft_freq_Hz=np.array([1.0/surge_period_vec, 1.0/sway_period_vec, 1.0/heave_period_vec, 1.0/roll_period_vec, 1.0/pitch_period_vec, 1.0/yaw_period_vec]).T
    #Max_PtfmPitch_vc[Max_PtfmPitch_vc > 10] = 10
    #Std_PtfmPitch_vec[Std_PtfmPitch_vec > 2] = 2.0
    #height_constraint_vec=np.squeeze(outdata[0]['towerse.height_constraint'])
    #tower_slope_vec=np.squeeze(outdata[0]['towerse.slope'])
    #nacelle_acceleration_vec=np.squeeze(outdata[0]['raft.response_nacelle acceleration'])
    #--------------------------------------------------------------------------------------

    #-------------------------------------Objectives------------------------------------
    floatingse_platform_mass_vec=np.squeeze(outdata[0]['floatingse.platform_mass'])
    AEP_vec=np.squeeze(outdata[0]['rotorse.rp.AEP'])
    LCOE_vec=np.squeeze(outdata[0]['financese.lcoe'])

    #--------------------------------------------------------------------------------------

    df = pd.DataFrame()
    df['Kp_float'] = Kp_float
    df['omega_pc'] = omega_pc
    df['omega_vs'] = omega_vs
    df['ps_percent'] = ps_percent
    df['ptfm_freq'] = ptfm_freq
    df['zeta_pc'] = zeta_pc
    df['zeta_vs'] = zeta_vs

    df['m_p'] = floatingse_platform_mass_vec
    df['tp_max'] = Max_PtfmPitch_vec
    df['max_nac_acc'] = max_nac_accel_vec
    df[['twr_f1','twr_f2', 'twr_f3','twr_f4','twr_f5','twr_f6']] = tower_structural_frequencies_vec
    df[['rft_f1','rft_f2', 'rft_f3','rft_f4','rft_f5','rft_f6']] = raft_freq_Hz
    df['AEP'] = AEP_vec
    df['LCOE'] = LCOE_vec

    #df['tp_max'] = df['tp_max'].fillna(12).clip(upper=12)
    df = df[df['tp_max'] <= 12.0]
    k=1
