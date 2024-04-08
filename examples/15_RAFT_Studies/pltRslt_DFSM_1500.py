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

def custom_crossover(ind1, ind2, variable_sets):
    """Perform a crossover and then correct any values to ensure they match the discrete sets."""
    # Perform a one-point crossover
    size = min(len(ind1), len(ind2))
    cxpoint = random.randint(1, size - 1)
    ind1[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:], ind1[cxpoint:]
    
    # Function to snap values to the nearest valid set value
    def snap_to_set(value, set_values):
        # Find the nearest value in set_values to value
        idx = np.abs(set_values - value).argmin()
        return set_values[idx]
    
    # Correct any values in ind1 and ind2 to ensure they adhere to the sets
    for i in range(size):
        ind1[i] = snap_to_set(ind1[i], variable_sets[i])
        ind2[i] = snap_to_set(ind2[i], variable_sets[i])

    return ind1, ind2

def plot_Mesh(Z_val, x_mesh, y_mesh, folder_path, filename='m_p_predict.jpg',
                         x_label='$D_\mathrm{Main- Clmn}\,\, \mathrm{[m]}$', 
                         y_label='$D_\mathrm{Side- Clmn}\,\, \mathrm{[m]}$', 
                         title='$\mathrm{platform\,mass} \,\, [\mathrm{Kilo \,\,Tons}]$',
                         highlight_x_value=10, highlight_y_value=12.5):

    # Normalize the data
    norm = Normalize(vmin=Z_val.min(), vmax=Z_val.max())
    rgba_colors = ScalarMappable(norm=norm, cmap='Reds').to_rgba(Z_val, alpha=1)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(rgba_colors, aspect='auto', origin='lower')

    # Set x and y ticks
    x_indices = np.arange(Z_val.shape[1])
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f'{x_mesh[0, i]:.1f}' for i in x_indices])

    y_indices = np.arange(Z_val.shape[0])
    ax.set_yticks(y_indices)
    ax.set_yticklabels([f'{y_mesh[i, 0]:.1f}' for i in y_indices])

    # Set labels and title
    ax.set_xlabel(x_label, fontsize=18)
    ax.set_ylabel(y_label, fontsize=18)
    ax.set_title(title, fontsize=15)

    # Customize the colorbar
    plt.colorbar(ScalarMappable(norm=norm, cmap='Reds'), ax=ax)

    # Highlight specific point
    highlight_x_index = np.where(x_mesh[0,:] == highlight_x_value)[0][0]
    highlight_y_index = np.where(y_mesh[:,0] == highlight_y_value)[0][0]
    ax.plot(highlight_x_index, highlight_y_index, 
            's', markersize=30, markerfacecolor='none', markeredgecolor='black', markeredgewidth=2)

    # Annotate each cell with its value
    for i in range(Z_val.shape[0]):
        for j in range(Z_val.shape[1]):
            ax.text(j, i, f'{Z_val[i, j]:.1f}', ha='center', va='center', color='black', fontsize=12)

    full_path = f'{folder_path}/{filename}'
    fig.savefig(full_path, format='jpg', dpi=300, bbox_inches='tight', pad_inches=0.55)
    # Save the figure
    #fig.savefig(f'{folder_path}/m_p_predict.jpg', format='jpg', dpi=300, bbox_inches='tight', pad_inches=0.55)


def calculate_correlations(X, y, kmeans, n_clusters):
    labels= kmeans.predict(X)  # see they correspond to what index
    correlations = []  # List to store correlations for each cluster
    for i in range(n_clusters):  # Assuming 10 clusters
        indices = np.where(labels == i)[0] # Indices of samples in the current cluster
        group_X = X[indices]  # Extracting feature values of the current cluster
        group_y = y[indices]  # Extracting target values of the current cluster
        
        # Compute Pearson correlation for each feature in the cluster
        group_correlations = [pearsonr(group_X[:, j], group_y)[0] for j in range(group_X.shape[1])]
        correlations.append(group_correlations)
    return correlations

def objfunc(xdict, X_scale_func, Y_scale_func):
    D_Main = xdict["D-Main"]
    D_Side = xdict["D-Side"]

    Feature = pd.DataFrame()
    Feature['D_main']=np.array([D_Main[0]])
    Feature['D_side']=np.array([D_Side[0]])
    #Feature=np.array([[D_Main[0], D_Side[0]]])
    Feature_s= X_scale_func.transform(Feature[['D_main', 'D_side']])

    Outputs_s =sm.predict_values(Feature_s)
    Outputs=Y_scale_func.inverse_transform(Outputs_s)

    mp=Outputs[0][0] *1e-6 # mp [KTonns]
    tp_max=Outputs[0][1]  #tp_max [deg]

    funcs = {}
    funcs["obj"] = mp
    conval = [0] * 1
    conval[0] = tp_max
    funcs["con"] = conval
    fail = False

    return funcs, fail 

def objfunc_Cobyla(x, X_scale_func, Y_scale_func,sm):
    #x=np.round(x * 10) / 10.0  # Adjusting for step size 0.1
    #x0 = [D_main, D_side, pnt_low_D, pnt_up_D, ps, Z_keel,  Z_frbd, Clmn_spcing]
    D_Main = x[0]
    D_Side = x[1]
    pnt_low_D = x[2]
    pnt_up_D = x[3]
    ps = x[4]
    Z_keel = x[5]
    Z_frbd = x[6]
    Clmn_spcing = x[7]
    Clmn_lngth=Z_frbd-Z_keel

    Feature = pd.DataFrame()
    Feature['D_main']=np.array([D_Main])
    Feature['D_side']=np.array([D_Side])
    Feature['pnt_low_D']=np.array([pnt_low_D])
    Feature['pnt_up_D']=np.array([pnt_up_D])
    Feature['ps']=np.array([ps])
    Feature['Z_keel']=np.array([Z_keel])
    Feature['Z_frbd']=np.array([Z_frbd])
    Feature['Clmn_spcing']=np.array([Clmn_spcing])
    Feature['Clmn_lngth']=np.array([Clmn_lngth])

    Feature_s=pd.DataFrame()
    
    Feature_s[['D_main_s', 'D_side_s', 'pnt_low_D_s', 'pnt_up_D_s', 'ps_s','Z_keel_s', 'Z_frbd_s', 'Clmn_spcing_s', 'Clmn_lngth_s' ]] = X_scale_func.transform(Feature[['D_main', 'D_side', 'pnt_low_D', 'pnt_up_D', 'ps','Z_keel', 'Z_frbd', 'Clmn_spcing', 'Clmn_lngth' ]])    
    Outputs_s = sm.predict_values(Feature_s[['D_main_s', 'D_side_s', 'pnt_low_D_s', 'pnt_up_D_s', 'ps_s','Z_keel_s', 'Z_frbd_s', 'Clmn_spcing_s']].values)
    Outputs = Y_scale_func.inverse_transform(Outputs_s)

    #Outputs : ['m_p', 'tp_max','max_nac_acc','twr_f1', 'twr_f2', 'twr_f3', 'twr_f4', 'twr_f5', 'twr_f6' ,'rft_f1', 'rft_f2', 'rft_f3', 'rft_f4', 'rft_f5', 'rft_f6', 'AEP', 'LCOE']
    mp=Outputs[0][0] *1e-6 # mp [KTonns]
    tp_max=Outputs[0][1]  #tp_max [deg]
    max_nac_acc=Outputs[0][2]  
    rft_f1=Outputs[0][9]  
    rft_f2=Outputs[0][10]  
    rft_f3=Outputs[0][11]  
    rft_f4=Outputs[0][12]  
    rft_f5=Outputs[0][13]  
    rft_f6=Outputs[0][14]  
    AEP=Outputs[0][15] * 1e-6
    LCOE=Outputs[0][16] * 1000.0 

    return mp

def const_Cobyla(x, X_scale_func, Y_scale_func,sm):
    #x=np.round(x * 10) / 10.0  # Adjusting for step size 0.1
    #x0 = [D_main, D_side, pnt_low_D, pnt_up_D, ps, Z_keel,  Z_frbd, Clmn_spcing]
    D_Main = x[0]
    D_Side = x[1]
    pnt_low_D = x[2]
    pnt_up_D = x[3]
    ps = x[4]
    Z_keel = x[5]
    Z_frbd = x[6]
    Clmn_spcing = x[7]
    Clmn_lngth=Z_frbd-Z_keel

    Feature = pd.DataFrame()
    Feature['D_main']=np.array([D_Main])
    Feature['D_side']=np.array([D_Side])
    Feature['pnt_low_D']=np.array([pnt_low_D])
    Feature['pnt_up_D']=np.array([pnt_up_D])
    Feature['ps']=np.array([ps])
    Feature['Z_keel']=np.array([Z_keel])
    Feature['Z_frbd']=np.array([Z_frbd])
    Feature['Clmn_spcing']=np.array([Clmn_spcing])
    Feature['Clmn_lngth']=np.array([Clmn_lngth])

    Feature_s=pd.DataFrame()
    
    Feature_s[['D_main_s', 'D_side_s', 'pnt_low_D_s', 'pnt_up_D_s', 'ps_s','Z_keel_s', 'Z_frbd_s', 'Clmn_spcing_s', 'Clmn_lngth_s' ]] = X_scale_func.transform(Feature[['D_main', 'D_side', 'pnt_low_D', 'pnt_up_D', 'ps','Z_keel', 'Z_frbd', 'Clmn_spcing', 'Clmn_lngth' ]])    
    Outputs_s = sm.predict_values(Feature_s[['D_main_s', 'D_side_s', 'pnt_low_D_s', 'pnt_up_D_s', 'ps_s','Z_keel_s', 'Z_frbd_s', 'Clmn_spcing_s']].values)
    Outputs = Y_scale_func.inverse_transform(Outputs_s)

    #Outputs : ['m_p', 'tp_max','max_nac_acc','twr_f1', 'twr_f2', 'twr_f3', 'twr_f4', 'twr_f5', 'twr_f6' ,'rft_f1', 'rft_f2', 'rft_f3', 'rft_f4', 'rft_f5', 'rft_f6', 'AEP', 'LCOE']
    mp=Outputs[0][0] *1e-6 # mp [KTonns]
    tp_max=Outputs[0][1]  #tp_max [deg]
    max_nac_acc=Outputs[0][2]  
    rft_f1=Outputs[0][9]  
    rft_f2=Outputs[0][10]  
    rft_f3=Outputs[0][11]  
    rft_f4=Outputs[0][12]  
    rft_f5=Outputs[0][13]  
    rft_f6=Outputs[0][14]  
    AEP=Outputs[0][15] * 1e-6
    LCOE=Outputs[0][16] * 1000.0 

    #return [6.9-tp_max, 0.7- max_nac_acc] #6.8-tp_max
    #return [6.74303-tp_max, 0.608581- max_nac_acc] #same as baseline
    return [5.74303-tp_max, 0.608581- max_nac_acc] 

def constraint1(x, X_scale_func, Y_scale_func,sm):
    """
    Returns the first value from G(x).
    """
    return const_Cobyla(x, X_scale_func, Y_scale_func,sm)[0]

def constraint2(x, X_scale_func, Y_scale_func,sm):
    """
    Returns the second value from G(x).
    """
    return const_Cobyla(x, X_scale_func, Y_scale_func,sm)[1]



def func_Plot_Rgeresion(df,tn='m_p', scale=1.0, folder_name='column_side_and_main',
                         x_label_1='$D_\mathrm{Main- Clmn}\,\, \mathrm{[m]}$', 
                         y_label_1='$D_\mathrm{Side- Clmn}\,\, \mathrm{[m]}$',
                         x_label_2='$e= t_p - \hat{t}_p \,\, \mathrm{[deg]}$', 
                         y_label_2='$\mathrm{Density}$'): #tn: target name
    folder_path = os.getcwd()+'/outputs/'+ folder_name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    df[tn]=df[tn]*scale
    df[tn+'_prdict']=df[tn+'_prdict']*scale
    fig, ax = plt.subplots(1,1)
    ax.plot(df[tn], df[tn+'_prdict'],'o',markersize=1.0)
    ax.plot([df[tn].min(), df[tn].max()], [df[tn].min(), df[tn].max()],color='k', alpha=0.3)
    mae = mean_absolute_error(df[tn], df[tn+'_prdict'])#/df[tn].mean() # I normalized that !!!!!!!!!----I removed normalization-------!!!!!!!!!!!!
    mse = mean_squared_error(df[tn], df[tn+'_prdict'])#/df[tn].mean()**2 # I normalized that !!!!!!!!!----I removed normalization-------!!!!!!!!!!!!
    rmse = np.sqrt(mse)
    r2 = r2_score(df[tn], df[tn+'_prdict'])
    ax.set_xlabel
    ax.set_xlabel(x_label_1, fontsize=18)
    ax.set_ylabel(y_label_1, fontsize=18)
    #ax.set_xlabel(rf'${tn} \,\, [\mathrm{{{unit}}}]$', fontsize=18)
    #ax.set_ylabel(rf'$\hat{{{tn}}} \,\, [\mathrm{{{unit}}}]$', fontsize=18)
    if any('train' in col for col in df.columns):
        ax.set_title(rf'$\mathrm{{Training \,\, Data}} \,\,\, , \,\, R^2 \, : \, {r2:.2f} $', fontsize=18)
        fig.savefig(folder_path +'/'+tn+'_r2_train.jpg', format='jpg', dpi=300, bbox_inches='tight',pad_inches=0.1)  
    elif any('test' in col for col in df.columns):
        ax.set_title(rf'$\mathrm{{Test \,\, Data}} \,\,\, , \,\, R^2 \, : \, {r2:.2f} $', fontsize=18)
        fig.savefig(folder_path +'/'+tn+'_r2_test.jpg', format='jpg', dpi=300, bbox_inches='tight',pad_inches=0.1)  
    else:
        ax.set_title(rf'$\mathrm{{All \,\, Data}} \,\,\, , \,\, R^2 \, : \, {r2:.2f} $', fontsize=18)
        fig.savefig(folder_path +'/'+tn+'_r2_all.jpg', format='jpg', dpi=300, bbox_inches='tight',pad_inches=0.1)  
    plt.close(fig)

    fig, ax = plt.subplots(1,1)
    e = (df[tn] - df[tn+'_prdict'])#/df[tn].mean() !!!!!!!!!----I removed normalization-------!!!!!!!!!!!!
    mean_e = np.mean(e)
    var_e = np.var(e)
    ax.hist(e, bins=40, density=True, alpha=0.6, color='g', label='Histogram')
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mean_e, np.sqrt(var_e))
    ax.plot(x, p, 'k', linewidth=2, label='Fit')
    ax.set_xlabel(x_label_2, fontsize=18)
    ax.set_ylabel(y_label_2, fontsize=18)
    #ax.set_xlabel(rf'$e=({tn}-\hat{{{tn}}}) / \, \overline{{{tn}}}  $', fontsize=18)
    #ax.set_ylabel('$\mathrm{Density}$', fontsize=18)
    if any('train' in col for col in df.columns):
        ax.set_title(rf'$\mathrm{{Training \,\, Data}} \,\,\, , \,\, e_{{\mathrm{{mean}}}} \, : \, {mean_e:.2f} \,\,\, , \,\, e_{{\mathrm{{var}}}} \, : \, {var_e:.2f} $', fontsize=14)
        fig.savefig(folder_path +'/'+tn+'_gauss_train.jpg', format='jpg', dpi=300, bbox_inches='tight',pad_inches=0.1)  
    elif any('test' in col for col in df.columns):
        ax.set_title(rf'$\mathrm{{Test \,\, Data}} \,\,\, , \,\, e_{{\mathrm{{mean}}}} \, : \, {mean_e:.2f} \,\,\, , \,\, e_{{\mathrm{{var}}}} \, : \, {var_e:.2f} $', fontsize=14)
        fig.savefig(folder_path +'/'+tn+'_gauss_test.jpg', format='jpg', dpi=300, bbox_inches='tight',pad_inches=0.1)  
    else:
        ax.set_title(rf'$\mathrm{{All \,\, Data}} \,\,\, , \,\, e_{{\mathrm{{mean}}}} \, : \, {mean_e:.2f} \,\,\, , \,\, e_{{\mathrm{{var}}}} \, : \, {var_e:.2f} $', fontsize=14)
        fig.savefig(folder_path +'/'+tn+'_gauss_all.jpg', format='jpg', dpi=300, bbox_inches='tight',pad_inches=0.1)  
    plt.close(fig)

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
    output_dir = os.path.join(run_dir, "outputs/15_RAFT_All_1500") # 15_RAFT_optimization_freq_step1_clmnSpacing_sideColmnsdiam_lowPntoon_diam") #opt coByla
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
    Clmns_keel_vec=np.squeeze(outdata[0]['floating.jointdv_0'])
    Clmns_freeboard_vec=np.squeeze(outdata[0]['floating.jointdv_1'])
    Clmns_spacing_vec=np.squeeze(outdata[0]['floating.jointdv_2'])
    ps_percent_vec=np.squeeze(outdata[0]['tune_rosco_ivc.ps_percent'])
    main_column_outer_diameter_vec=np.squeeze(outdata[0]['floating.memgrp0.outer_diameter_in'])
    side_columns_outer_diameter_vec=np.squeeze(outdata[0]['floating.memgrp1.outer_diameter_in'])
    Y_pntoon_upper_vec=np.squeeze(outdata[0]['floating.memgrp2.outer_diameter_in'])
    Y_pntoon_lower_vec=np.squeeze(outdata[0]['floating.memgrp3.outer_diameter_in'])
    Clmns_Length_vec=Clmns_freeboard_vec-Clmns_keel_vec
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
    df['D_main'] = main_column_outer_diameter_vec
    df['D_side'] = side_columns_outer_diameter_vec
    df['pnt_low_D'] = Y_pntoon_lower_vec
    df['pnt_up_D'] = Y_pntoon_upper_vec
    df['ps'] = ps_percent_vec
    df['Z_keel'] = Clmns_keel_vec
    df['Z_frbd'] = Clmns_freeboard_vec
    df['Clmn_spcing'] = Clmns_spacing_vec
    df['Clmn_lngth'] = Clmns_Length_vec

    df['m_p'] = floatingse_platform_mass_vec
    df['tp_max'] = Max_PtfmPitch_vec
    df['max_nac_acc'] = max_nac_accel_vec
    df[['twr_f1','twr_f2', 'twr_f3','twr_f4','twr_f5','twr_f6']] = tower_structural_frequencies_vec
    df[['rft_f1','rft_f2', 'rft_f3','rft_f4','rft_f5','rft_f6']] = raft_freq_Hz
    df['AEP'] = AEP_vec
    df['LCOE'] = LCOE_vec

    #df['tp_max'] = df['tp_max'].fillna(12).clip(upper=12)
    df = df[df['tp_max'] <= 12.0]

    #Features = df[['Ypntoon_lower_Raw', 'Clmns_spacing_Raw', 'D_side']]
    Features = df[['D_main', 'D_side', 'pnt_low_D', 'pnt_up_D', 'ps', 'Z_keel', 'Z_frbd', 'Clmn_spcing']]
    Targets = df[['m_p', 'tp_max', 'max_nac_acc', 'twr_f1','twr_f2','twr_f3','twr_f4','twr_f5','twr_f6', 'rft_f1', 'rft_f2','rft_f3','rft_f4','rft_f5','rft_f6','AEP', 'LCOE']]


    X_train, X_test, y_train, y_test = train_test_split(Features, Targets, test_size=0.2, random_state=42, shuffle=True)
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    df_train['D_main']=X_train.values[:,0]
    df_train['D_side']=X_train.values[:,1]
    df_train['pnt_low_D'] = X_train.values[:,2]
    df_train['pnt_up_D'] = X_train.values[:,3]
    df_train['ps'] = X_train.values[:,4]
    df_train['Z_keel'] = X_train.values[:,5]
    df_train['Z_frbd'] = X_train.values[:,6]
    df_train['Clmn_spcing'] = X_train.values[:,7]
    df_train['Clmn_lngth'] = X_train.values[:,6]-X_train.values[:,5]

    df_train['m_p']=y_train.values[:,0]
    df_train['tp_max']=y_train.values[:,1]
    df_train['max_nac_acc']=y_train.values[:,2]
    df_train['twr_f1']=y_train.values[:,3]
    df_train['twr_f2']=y_train.values[:,4]
    df_train['twr_f3']=y_train.values[:,5]
    df_train['twr_f4']=y_train.values[:,6]
    df_train['twr_f5']=y_train.values[:,7]
    df_train['twr_f6']=y_train.values[:,8]
    df_train['rft_f1']=y_train.values[:,9]
    df_train['rft_f2']=y_train.values[:,10]
    df_train['rft_f3']=y_train.values[:,11]
    df_train['rft_f4']=y_train.values[:,12]
    df_train['rft_f5']=y_train.values[:,13]
    df_train['rft_f6']=y_train.values[:,14]
    df_train['AEP']=y_train.values[:,15]
    df_train['LCOE']=y_train.values[:,16]

    df_train['train_indices']=X_train.index.values

    df_test['D_main']=X_test.values[:,0]
    df_test['D_side']=X_test.values[:,1]
    df_test['pnt_low_D'] = X_test.values[:,2]
    df_test['pnt_up_D'] = X_test.values[:,3]
    df_test['ps'] = X_test.values[:,4]
    df_test['Z_keel'] = X_test.values[:,5]
    df_test['Z_frbd'] = X_test.values[:,6]
    df_test['Clmn_spcing'] = X_test.values[:,7]
    df_test['Clmn_lngth'] = X_test.values[:,6]-X_test.values[:,5]

    df_test['m_p']=y_test.values[:,0]
    df_test['tp_max']=y_test.values[:,1]
    df_test['max_nac_acc']=y_test.values[:,2]
    df_test['twr_f1']=y_test.values[:,3]
    df_test['twr_f2']=y_test.values[:,4]
    df_test['twr_f3']=y_test.values[:,5]
    df_test['twr_f4']=y_test.values[:,6]
    df_test['twr_f5']=y_test.values[:,7]
    df_test['twr_f6']=y_test.values[:,8]
    df_test['rft_f1']=y_test.values[:,9]
    df_test['rft_f2']=y_test.values[:,10]
    df_test['rft_f3']=y_test.values[:,11]
    df_test['rft_f4']=y_test.values[:,12]
    df_test['rft_f5']=y_test.values[:,13]
    df_test['rft_f6']=y_test.values[:,14]
    df_test['AEP']=y_test.values[:,15]
    df_test['LCOE']=y_test.values[:,16]

    df_test['test_indices']=X_test.index.values

    X_scale_func = StandardScaler()
    Y_scale_func = StandardScaler()

    X_scale_func.fit_transform(df_train[['D_main', 'D_side', 'pnt_low_D', 'pnt_up_D', 'ps','Z_keel', 'Z_frbd', 'Clmn_spcing', 'Clmn_lngth' ]])
    Y_scale_func.fit_transform(df_train[['m_p', 'tp_max','max_nac_acc','twr_f1', 'twr_f2', 'twr_f3', 'twr_f4', 'twr_f5', 'twr_f6' ,'rft_f1', 'rft_f2', 'rft_f3', 'rft_f4', 'rft_f5', 'rft_f6', 'AEP', 'LCOE']])

    #scaling features and targets to make them ready for training
    df_train[['D_main_s', 'D_side_s', 'pnt_low_D_s', 'pnt_up_D_s', 'ps_s','Z_keel_s', 'Z_frbd_s', 'Clmn_spcing_s', 'Clmn_lngth_s' ]] = X_scale_func.transform(df_train[['D_main', 'D_side', 'pnt_low_D', 'pnt_up_D', 'ps','Z_keel', 'Z_frbd', 'Clmn_spcing', 'Clmn_lngth' ]])
    df_train[['m_p_s', 'tp_max_s','max_nac_acc_s','twr_f1_s', 'twr_f2_s', 'twr_f3_s', 'twr_f4_s', 'twr_f5_s', 'twr_f6_s' ,'rft_f1_s', 'rft_f2_s', 'rft_f3_s', 'rft_f4_s', 'rft_f5_s', 'rft_f6_s', 'AEP_s', 'LCOE_s']] = Y_scale_func.transform(df_train[['m_p', 'tp_max','max_nac_acc','twr_f1', 'twr_f2', 'twr_f3', 'twr_f4', 'twr_f5', 'twr_f6' ,'rft_f1', 'rft_f2', 'rft_f3', 'rft_f4', 'rft_f5', 'rft_f6', 'AEP', 'LCOE']])
    df_test[['D_main_s', 'D_side_s', 'pnt_low_D_s', 'pnt_up_D_s', 'ps_s','Z_keel_s', 'Z_frbd_s', 'Clmn_spcing_s', 'Clmn_lngth_s' ]]  = X_scale_func.transform(df_test[['D_main', 'D_side', 'pnt_low_D', 'pnt_up_D', 'ps','Z_keel', 'Z_frbd', 'Clmn_spcing', 'Clmn_lngth' ]])
    df_test[['m_p_s', 'tp_max_s','max_nac_acc_s','twr_f1_s', 'twr_f2_s', 'twr_f3_s', 'twr_f4_s', 'twr_f5_s', 'twr_f6_s' ,'rft_f1_s', 'rft_f2_s', 'rft_f3_s', 'rft_f4_s', 'rft_f5_s', 'rft_f6_s', 'AEP_s', 'LCOE_s']]  = Y_scale_func.transform(df_test[['m_p', 'tp_max','max_nac_acc','twr_f1', 'twr_f2', 'twr_f3', 'twr_f4', 'twr_f5', 'twr_f6' ,'rft_f1', 'rft_f2', 'rft_f3', 'rft_f4', 'rft_f5', 'rft_f6', 'AEP', 'LCOE']])

    sm = KPLSK(theta0=[1e-2],print_training=False, nugget=1e-1)
    #sm = KRG(theta0=[1e-2],print_training=False, nugget=1e-1)
    sm.options['print_prediction'] = False
    sm.set_training_values(df_train[['D_main_s', 'D_side_s', 'pnt_low_D_s', 'pnt_up_D_s', 'ps_s','Z_keel_s', 'Z_frbd_s', 'Clmn_spcing_s']].values, df_train[['m_p_s', 'tp_max_s','max_nac_acc_s','twr_f1_s', 'twr_f2_s', 'twr_f3_s', 'twr_f4_s', 'twr_f5_s', 'twr_f6_s' ,'rft_f1_s', 'rft_f2_s', 'rft_f3_s', 'rft_f4_s', 'rft_f5_s', 'rft_f6_s', 'AEP_s', 'LCOE_s']].values)
    sm.train()
    #filtered_df = df[(df['D_main'] == 10) & (df['D_side'] == 12.5)]
    # esimate tagregts and tehir variance
    df_train[['m_p_s_prdict', 'tp_max_s_prdict','max_nac_acc_s_prdict','twr_f1_s_prdict', 'twr_f2_s_prdict', 'twr_f3_s_prdict', 'twr_f4_s_prdict', 'twr_f5_s_prdict', 'twr_f6_s_prdict' ,'rft_f1_s_prdict', 'rft_f2_s_prdict', 'rft_f3_s_prdict', 'rft_f4_s_prdict', 'rft_f5_s_prdict', 'rft_f6_s_prdict', 'AEP_s_prdict', 'LCOE_s_prdict']] =sm.predict_values(df_train[['D_main_s', 'D_side_s', 'pnt_low_D_s', 'pnt_up_D_s', 'ps_s','Z_keel_s', 'Z_frbd_s', 'Clmn_spcing_s']].values)
    #df_train[['var_m_p_s_prdict','var_tp_max_s_prdict']] = sm.predict_variances(df_train[['D_main_s', 'D_side_s', 'pnt_low_D_s', 'pnt_up_D_s', 'ps_s','Z_keel_s', 'Z_frbd_s', 'Clmn_spcing_s']].values) # estimated variance
    df_test[['m_p_s_prdict', 'tp_max_s_prdict','max_nac_acc_s_prdict','twr_f1_s_prdict', 'twr_f2_s_prdict', 'twr_f3_s_prdict', 'twr_f4_s_prdict', 'twr_f5_s_prdict', 'twr_f6_s_prdict' ,'rft_f1_s_prdict', 'rft_f2_s_prdict', 'rft_f3_s_prdict', 'rft_f4_s_prdict', 'rft_f5_s_prdict', 'rft_f6_s_prdict', 'AEP_s_prdict', 'LCOE_s_prdict']] =sm.predict_values(df_test[['D_main_s', 'D_side_s', 'pnt_low_D_s', 'pnt_up_D_s', 'ps_s','Z_keel_s', 'Z_frbd_s', 'Clmn_spcing_s']].values)
    #df_test[['var_m_p_s_prdict','var_tp_max_s_prdict']] = sm.predict_variances(df_test[['D_main_s', 'D_side_s', 'pnt_low_D_s', 'pnt_up_D_s', 'ps_s','Z_keel_s', 'Z_frbd_s', 'Clmn_spcing_s']].values) # estimated variance

    #unscale targets
    df_train[['m_p_prdict', 'tp_max_prdict','max_nac_acc_prdict','twr_f1_prdict', 'twr_f2_prdict', 'twr_f3_prdict', 'twr_f4_prdict', 'twr_f5_prdict', 'twr_f6_prdict' ,'rft_f1_prdict', 'rft_f2_prdict', 'rft_f3_prdict', 'rft_f4_prdict', 'rft_f5_prdict', 'rft_f6_prdict', 'AEP_prdict', 'LCOE_prdict']]=Y_scale_func.inverse_transform(df_train[['m_p_s_prdict', 'tp_max_s_prdict','max_nac_acc_s_prdict','twr_f1_s_prdict', 'twr_f2_s_prdict', 'twr_f3_s_prdict', 'twr_f4_s_prdict', 'twr_f5_s_prdict', 'twr_f6_s_prdict' ,'rft_f1_s_prdict', 'rft_f2_s_prdict', 'rft_f3_s_prdict', 'rft_f4_s_prdict', 'rft_f5_s_prdict', 'rft_f6_s_prdict', 'AEP_s_prdict', 'LCOE_s_prdict']])
    df_test[['m_p_prdict', 'tp_max_prdict','max_nac_acc_prdict','twr_f1_prdict', 'twr_f2_prdict', 'twr_f3_prdict', 'twr_f4_prdict', 'twr_f5_prdict', 'twr_f6_prdict' ,'rft_f1_prdict', 'rft_f2_prdict', 'rft_f3_prdict', 'rft_f4_prdict', 'rft_f5_prdict', 'rft_f6_prdict', 'AEP_prdict', 'LCOE_prdict']]=Y_scale_func.inverse_transform(df_test[['m_p_s_prdict', 'tp_max_s_prdict','max_nac_acc_s_prdict','twr_f1_s_prdict', 'twr_f2_s_prdict', 'twr_f3_s_prdict', 'twr_f4_s_prdict', 'twr_f5_s_prdict', 'twr_f6_s_prdict' ,'rft_f1_s_prdict', 'rft_f2_s_prdict', 'rft_f3_s_prdict', 'rft_f4_s_prdict', 'rft_f5_s_prdict', 'rft_f6_s_prdict', 'AEP_s_prdict', 'LCOE_s_prdict']])


    # all data
    df[['D_main_s', 'D_side_s', 'pnt_low_D_s', 'pnt_up_D_s', 'ps_s','Z_keel_s', 'Z_frbd_s', 'Clmn_spcing_s', 'Clmn_lngth_s' ]] = X_scale_func.transform(df[['D_main', 'D_side', 'pnt_low_D', 'pnt_up_D', 'ps','Z_keel', 'Z_frbd', 'Clmn_spcing', 'Clmn_lngth' ]])
    df[['m_p_s', 'tp_max_s','max_nac_acc_s','twr_f1_s', 'twr_f2_s', 'twr_f3_s', 'twr_f4_s', 'twr_f5_s', 'twr_f6_s' ,'rft_f1_s', 'rft_f2_s', 'rft_f3_s', 'rft_f4_s', 'rft_f5_s', 'rft_f6_s', 'AEP_s', 'LCOE_s']] = Y_scale_func.transform(df[['m_p', 'tp_max','max_nac_acc','twr_f1', 'twr_f2', 'twr_f3', 'twr_f4', 'twr_f5', 'twr_f6' ,'rft_f1', 'rft_f2', 'rft_f3', 'rft_f4', 'rft_f5', 'rft_f6', 'AEP', 'LCOE']])
    df[['m_p_s_prdict', 'tp_max_s_prdict','max_nac_acc_s_prdict','twr_f1_s_prdict', 'twr_f2_s_prdict', 'twr_f3_s_prdict', 'twr_f4_s_prdict', 'twr_f5_s_prdict', 'twr_f6_s_prdict' ,'rft_f1_s_prdict', 'rft_f2_s_prdict', 'rft_f3_s_prdict', 'rft_f4_s_prdict', 'rft_f5_s_prdict', 'rft_f6_s_prdict', 'AEP_s_prdict', 'LCOE_s_prdict']] =sm.predict_values(df[['D_main_s', 'D_side_s', 'pnt_low_D_s', 'pnt_up_D_s', 'ps_s','Z_keel_s', 'Z_frbd_s', 'Clmn_spcing_s']].values)
    df[['m_p_prdict', 'tp_max_prdict','max_nac_acc_prdict','twr_f1_prdict', 'twr_f2_prdict', 'twr_f3_prdict', 'twr_f4_prdict', 'twr_f5_prdict', 'twr_f6_prdict' ,'rft_f1_prdict', 'rft_f2_prdict', 'rft_f3_prdict', 'rft_f4_prdict', 'rft_f5_prdict', 'rft_f6_prdict', 'AEP_prdict', 'LCOE_prdict']]=Y_scale_func.inverse_transform(df[['m_p_s_prdict', 'tp_max_s_prdict','max_nac_acc_s_prdict','twr_f1_s_prdict', 'twr_f2_s_prdict', 'twr_f3_s_prdict', 'twr_f4_s_prdict', 'twr_f5_s_prdict', 'twr_f6_s_prdict' ,'rft_f1_s_prdict', 'rft_f2_s_prdict', 'rft_f3_s_prdict', 'rft_f4_s_prdict', 'rft_f5_s_prdict', 'rft_f6_s_prdict', 'AEP_s_prdict', 'LCOE_s_prdict']])

    folder_path = os.getcwd()+'/outputs/'+ 'All_1500'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    #func_Plot_Rgeresion(df=df_train, tn='m_p', unit='KTonns', folder_name='All_1500')
    func_Plot_Rgeresion(df=df_train.copy(),tn='m_p', scale=1e-6, folder_name='All_1500',
                         x_label_1='$m_{\mathrm{p}}\,\, \mathrm{[KTonns]}$', 
                         y_label_1='$\hat{m}_{\mathrm{p}}\,\, \mathrm{[KTonns]}$',
                         x_label_2='$e= m_p - \hat{m}_p \,\, \mathrm{[KTonns]}$', 
                         y_label_2='$\mathrm{Density}$')
    func_Plot_Rgeresion(df=df_train.copy(),tn='tp_max', scale=1.0, folder_name='All_1500',
                         x_label_1='$t_{\mathrm{p}}^{\mathrm{max}}\,\, \mathrm{[deg]}$', 
                         y_label_1='$\hat{t}_{\mathrm{p}}^{\mathrm{max}}\,\, \mathrm{[KTonns]}$',
                         x_label_2='$e= t_{\mathrm{p}}^{\mathrm{max}} - \hat{t}_{\mathrm{p}}^{\mathrm{max}} \,\, \mathrm{[deg]}$', 
                         y_label_2='$\mathrm{Density}$')
    func_Plot_Rgeresion(df=df_train.copy(),tn='max_nac_acc', scale=1.0, folder_name='All_1500',
                         x_label_1='$acc_{\mathrm{nac}}^{\mathrm{max}}\,\, \mathrm{[m/s]}$', 
                         y_label_1='$\hat{acc}_{\mathrm{nac}}^{\mathrm{max}}\,\, \mathrm{[m/s]}$',
                         x_label_2='$e= acc_{\mathrm{nac}}^{\mathrm{max}} - \hat{acc}_{\mathrm{nac}}^{\mathrm{max}} \,\, \mathrm{[m/s]}$', 
                         y_label_2='$\mathrm{Density}$')
    func_Plot_Rgeresion(df=df_train.copy(),tn='rft_f1', scale=1.0, folder_name='All_1500',
                         x_label_1='$f_{\mathrm{raft}}^{\mathrm{1}}\,\, \mathrm{[Hz]}$', 
                         y_label_1='$\hat{f}_{\mathrm{raft}}^{\mathrm{1}}\,\, \mathrm{[Hz]}$',
                         x_label_2='$e= f_{\mathrm{raft}}^{\mathrm{1}} - \hat{f}_{\mathrm{raft}}^{\mathrm{1}} \,\, \mathrm{[Hz]}$', 
                         y_label_2='$\mathrm{Density}$')
    func_Plot_Rgeresion(df=df_train.copy(),tn='rft_f2', scale=1.0, folder_name='All_1500',
                         x_label_1='$f_{\mathrm{raft}}^{\mathrm{2}}\,\, \mathrm{[Hz]}$', 
                         y_label_1='$\hat{f}_{\mathrm{raft}}^{\mathrm{2}}\,\, \mathrm{[Hz]}$',
                         x_label_2='$e= f_{\mathrm{raft}}^{\mathrm{2}} - \hat{f}_{\mathrm{raft}}^{\mathrm{2}} \,\, \mathrm{[Hz]}$', 
                         y_label_2='$\mathrm{Density}$')
    func_Plot_Rgeresion(df=df_train.copy(),tn='rft_f3', scale=1.0, folder_name='All_1500',
                         x_label_1='$f_{\mathrm{raft}}^{\mathrm{3}}\,\, \mathrm{[Hz]}$', 
                         y_label_1='$\hat{f}_{\mathrm{raft}}^{\mathrm{3}}\,\, \mathrm{[Hz]}$',
                         x_label_2='$e= f_{\mathrm{raft}}^{\mathrm{3}} - \hat{f}_{\mathrm{raft}}^{\mathrm{3}} \,\, \mathrm{[Hz]}$', 
                         y_label_2='$\mathrm{Density}$')
    func_Plot_Rgeresion(df=df_train.copy(),tn='rft_f4', scale=1.0, folder_name='All_1500',
                         x_label_1='$f_{\mathrm{raft}}^{\mathrm{4}}\,\, \mathrm{[Hz]}$', 
                         y_label_1='$\hat{f}_{\mathrm{raft}}^{\mathrm{4}}\,\, \mathrm{[Hz]}$',
                         x_label_2='$e= f_{\mathrm{raft}}^{\mathrm{4}} - \hat{f}_{\mathrm{raft}}^{\mathrm{4}} \,\, \mathrm{[Hz]}$', 
                         y_label_2='$\mathrm{Density}$')
    func_Plot_Rgeresion(df=df_train.copy(),tn='rft_f5', scale=1.0, folder_name='All_1500',
                         x_label_1='$f_{\mathrm{raft}}^{\mathrm{5}}\,\, \mathrm{[Hz]}$', 
                         y_label_1='$\hat{f}_{\mathrm{raft}}^{\mathrm{5}}\,\, \mathrm{[Hz]}$',
                         x_label_2='$e= f_{\mathrm{raft}}^{\mathrm{5}} - \hat{f}_{\mathrm{raft}}^{\mathrm{5}} \,\, \mathrm{[Hz]}$', 
                         y_label_2='$\mathrm{Density}$')
    func_Plot_Rgeresion(df=df_train.copy(),tn='rft_f6', scale=1.0, folder_name='All_1500',
                         x_label_1='$f_{\mathrm{raft}}^{\mathrm{6}}\,\, \mathrm{[Hz]}$', 
                         y_label_1='$\hat{f}_{\mathrm{raft}}^{\mathrm{6}}\,\, \mathrm{[Hz]}$',
                         x_label_2='$e= f_{\mathrm{raft}}^{\mathrm{6}} - \hat{f}_{\mathrm{raft}}^{\mathrm{6}} \,\, \mathrm{[Hz]}$', 
                         y_label_2='$\mathrm{Density}$')
    func_Plot_Rgeresion(df=df_train.copy(),tn='AEP', scale=1.e-6, folder_name='All_1500',
                         x_label_1='$\mathrm{AEP}\,\, \mathrm{[MWh]}$', 
                         y_label_1='$\hat{\mathrm{AEP}}\,\, \mathrm{[MWH]}$',
                         x_label_2='$e= \mathrm{AEP} - \hat{\mathrm{AEP}}\,\, \mathrm{[MWh]}$', 
                         y_label_2='$\mathrm{Density}$')
    func_Plot_Rgeresion(df=df_train.copy(),tn='LCOE', scale=1000.0, folder_name='All_1500',
                         x_label_1='$\mathrm{LCOE}\,\, \mathrm{[USD/MWh]}$', 
                         y_label_1='$\hat{\mathrm{LCOE}}\,\, \mathrm{[USD/MWh]}$',
                         x_label_2='$e= \mathrm{LCOE} - \hat{\mathrm{LCOE}}\,\, \mathrm{[USD/MWh]}$', 
                         y_label_2='$\mathrm{Density}$')

    
    func_Plot_Rgeresion(df=df_test.copy(),tn='m_p', scale=1e-6, folder_name='All_1500',
                         x_label_1='$m_{\mathrm{p}}\,\, \mathrm{[KTonns]}$', 
                         y_label_1='$\hat{m}_{\mathrm{p}}\,\, \mathrm{[KTonns]}$',
                         x_label_2='$e= m_p - \hat{m}_p \,\, \mathrm{[KTonns]}$', 
                         y_label_2='$\mathrm{Density}$')
    func_Plot_Rgeresion(df=df_test.copy(),tn='tp_max', scale=1.0, folder_name='All_1500',
                         x_label_1='$t_{\mathrm{p}}^{\mathrm{max}}\,\, \mathrm{[deg]}$', 
                         y_label_1='$\hat{t}_{\mathrm{p}}^{\mathrm{max}}\,\, \mathrm{[KTonns]}$',
                         x_label_2='$e= t_{\mathrm{p}}^{\mathrm{max}} - \hat{t}_{\mathrm{p}}^{\mathrm{max}} \,\, \mathrm{[deg]}$', 
                         y_label_2='$\mathrm{Density}$')
    func_Plot_Rgeresion(df=df_test.copy(),tn='max_nac_acc', scale=1.0, folder_name='All_1500',
                         x_label_1='$acc_{\mathrm{nac}}^{\mathrm{max}}\,\, \mathrm{[m/s]}$', 
                         y_label_1='$\hat{acc}_{\mathrm{nac}}^{\mathrm{max}}\,\, \mathrm{[m/s]}$',
                         x_label_2='$e= acc_{\mathrm{nac}}^{\mathrm{max}} - \hat{acc}_{\mathrm{nac}}^{\mathrm{max}} \,\, \mathrm{[m/s]}$', 
                         y_label_2='$\mathrm{Density}$')
    func_Plot_Rgeresion(df=df_test.copy(),tn='rft_f1', scale=1.0, folder_name='All_1500',
                         x_label_1='$f_{\mathrm{raft}}^{\mathrm{1}}\,\, \mathrm{[Hz]}$', 
                         y_label_1='$\hat{f}_{\mathrm{raft}}^{\mathrm{1}}\,\, \mathrm{[Hz]}$',
                         x_label_2='$e= f_{\mathrm{raft}}^{\mathrm{1}} - \hat{f}_{\mathrm{raft}}^{\mathrm{1}} \,\, \mathrm{[Hz]}$', 
                         y_label_2='$\mathrm{Density}$')
    func_Plot_Rgeresion(df=df_test.copy(),tn='rft_f2', scale=1.0, folder_name='All_1500',
                         x_label_1='$f_{\mathrm{raft}}^{\mathrm{2}}\,\, \mathrm{[Hz]}$', 
                         y_label_1='$\hat{f}_{\mathrm{raft}}^{\mathrm{2}}\,\, \mathrm{[Hz]}$',
                         x_label_2='$e= f_{\mathrm{raft}}^{\mathrm{2}} - \hat{f}_{\mathrm{raft}}^{\mathrm{2}} \,\, \mathrm{[Hz]}$', 
                         y_label_2='$\mathrm{Density}$')
    func_Plot_Rgeresion(df=df_test.copy(),tn='rft_f3', scale=1.0, folder_name='All_1500',
                         x_label_1='$f_{\mathrm{raft}}^{\mathrm{3}}\,\, \mathrm{[Hz]}$', 
                         y_label_1='$\hat{f}_{\mathrm{raft}}^{\mathrm{3}}\,\, \mathrm{[Hz]}$',
                         x_label_2='$e= f_{\mathrm{raft}}^{\mathrm{3}} - \hat{f}_{\mathrm{raft}}^{\mathrm{3}} \,\, \mathrm{[Hz]}$', 
                         y_label_2='$\mathrm{Density}$')
    func_Plot_Rgeresion(df=df_test.copy(),tn='rft_f4', scale=1.0, folder_name='All_1500',
                         x_label_1='$f_{\mathrm{raft}}^{\mathrm{4}}\,\, \mathrm{[Hz]}$', 
                         y_label_1='$\hat{f}_{\mathrm{raft}}^{\mathrm{4}}\,\, \mathrm{[Hz]}$',
                         x_label_2='$e= f_{\mathrm{raft}}^{\mathrm{4}} - \hat{f}_{\mathrm{raft}}^{\mathrm{4}} \,\, \mathrm{[Hz]}$', 
                         y_label_2='$\mathrm{Density}$')
    func_Plot_Rgeresion(df=df_test.copy(),tn='rft_f5', scale=1.0, folder_name='All_1500',
                         x_label_1='$f_{\mathrm{raft}}^{\mathrm{5}}\,\, \mathrm{[Hz]}$', 
                         y_label_1='$\hat{f}_{\mathrm{raft}}^{\mathrm{5}}\,\, \mathrm{[Hz]}$',
                         x_label_2='$e= f_{\mathrm{raft}}^{\mathrm{5}} - \hat{f}_{\mathrm{raft}}^{\mathrm{5}} \,\, \mathrm{[Hz]}$', 
                         y_label_2='$\mathrm{Density}$')
    func_Plot_Rgeresion(df=df_test.copy(),tn='rft_f6', scale=1.0, folder_name='All_1500',
                         x_label_1='$f_{\mathrm{raft}}^{\mathrm{6}}\,\, \mathrm{[Hz]}$', 
                         y_label_1='$\hat{f}_{\mathrm{raft}}^{\mathrm{6}}\,\, \mathrm{[Hz]}$',
                         x_label_2='$e= f_{\mathrm{raft}}^{\mathrm{6}} - \hat{f}_{\mathrm{raft}}^{\mathrm{6}} \,\, \mathrm{[Hz]}$', 
                         y_label_2='$\mathrm{Density}$')
    func_Plot_Rgeresion(df=df_test.copy(),tn='AEP', scale=1.e-6, folder_name='All_1500',
                         x_label_1='$\mathrm{AEP}\,\, \mathrm{[MWh]}$', 
                         y_label_1='$\hat{\mathrm{AEP}}\,\, \mathrm{[MWH]}$',
                         x_label_2='$e= \mathrm{AEP} - \hat{\mathrm{AEP}}\,\, \mathrm{[MWh]}$', 
                         y_label_2='$\mathrm{Density}$')
    func_Plot_Rgeresion(df=df_test.copy(),tn='LCOE', scale=1000.0, folder_name='All_1500',
                         x_label_1='$\mathrm{LCOE}\,\, \mathrm{[USD/MWh]}$', 
                         y_label_1='$\hat{\mathrm{LCOE}}\,\, \mathrm{[USD/MWh]}$',
                         x_label_2='$e= \mathrm{LCOE} - \hat{\mathrm{LCOE}}\,\, \mathrm{[USD/MWh]}$', 
                         y_label_2='$\mathrm{Density}$')    

    
    func_Plot_Rgeresion(df=df.copy(),tn='m_p', scale=1e-6, folder_name='All_1500',
                         x_label_1='$m_{\mathrm{p}}\,\, \mathrm{[KTonns]}$', 
                         y_label_1='$\hat{m}_{\mathrm{p}}\,\, \mathrm{[KTonns]}$',
                         x_label_2='$e= m_p - \hat{m}_p \,\, \mathrm{[KTonns]}$', 
                         y_label_2='$\mathrm{Density}$')
    func_Plot_Rgeresion(df=df.copy(),tn='tp_max', scale=1.0, folder_name='All_1500',
                         x_label_1='$t_{\mathrm{p}}^{\mathrm{max}}\,\, \mathrm{[deg]}$', 
                         y_label_1='$\hat{t}_{\mathrm{p}}^{\mathrm{max}}\,\, \mathrm{[KTonns]}$',
                         x_label_2='$e= t_{\mathrm{p}}^{\mathrm{max}} - \hat{t}_{\mathrm{p}}^{\mathrm{max}} \,\, \mathrm{[deg]}$', 
                         y_label_2='$\mathrm{Density}$')
    func_Plot_Rgeresion(df=df.copy(),tn='max_nac_acc', scale=1.0, folder_name='All_1500',
                         x_label_1='$acc_{\mathrm{nac}}^{\mathrm{max}}\,\, \mathrm{[m/s]}$', 
                         y_label_1='$\hat{acc}_{\mathrm{nac}}^{\mathrm{max}}\,\, \mathrm{[m/s]}$',
                         x_label_2='$e= acc_{\mathrm{nac}}^{\mathrm{max}} - \hat{acc}_{\mathrm{nac}}^{\mathrm{max}} \,\, \mathrm{[m/s]}$', 
                         y_label_2='$\mathrm{Density}$')
    func_Plot_Rgeresion(df=df.copy(),tn='rft_f1', scale=1.0, folder_name='All_1500',
                         x_label_1='$f_{\mathrm{raft}}^{\mathrm{1}}\,\, \mathrm{[Hz]}$', 
                         y_label_1='$\hat{f}_{\mathrm{raft}}^{\mathrm{1}}\,\, \mathrm{[Hz]}$',
                         x_label_2='$e= f_{\mathrm{raft}}^{\mathrm{1}} - \hat{f}_{\mathrm{raft}}^{\mathrm{1}} \,\, \mathrm{[Hz]}$', 
                         y_label_2='$\mathrm{Density}$')
    func_Plot_Rgeresion(df=df.copy(),tn='rft_f2', scale=1.0, folder_name='All_1500',
                         x_label_1='$f_{\mathrm{raft}}^{\mathrm{2}}\,\, \mathrm{[Hz]}$', 
                         y_label_1='$\hat{f}_{\mathrm{raft}}^{\mathrm{2}}\,\, \mathrm{[Hz]}$',
                         x_label_2='$e= f_{\mathrm{raft}}^{\mathrm{2}} - \hat{f}_{\mathrm{raft}}^{\mathrm{2}} \,\, \mathrm{[Hz]}$', 
                         y_label_2='$\mathrm{Density}$')
    func_Plot_Rgeresion(df=df.copy(),tn='rft_f3', scale=1.0, folder_name='All_1500',
                         x_label_1='$f_{\mathrm{raft}}^{\mathrm{3}}\,\, \mathrm{[Hz]}$', 
                         y_label_1='$\hat{f}_{\mathrm{raft}}^{\mathrm{3}}\,\, \mathrm{[Hz]}$',
                         x_label_2='$e= f_{\mathrm{raft}}^{\mathrm{3}} - \hat{f}_{\mathrm{raft}}^{\mathrm{3}} \,\, \mathrm{[Hz]}$', 
                         y_label_2='$\mathrm{Density}$')
    func_Plot_Rgeresion(df=df.copy(),tn='rft_f4', scale=1.0, folder_name='All_1500',
                         x_label_1='$f_{\mathrm{raft}}^{\mathrm{4}}\,\, \mathrm{[Hz]}$', 
                         y_label_1='$\hat{f}_{\mathrm{raft}}^{\mathrm{4}}\,\, \mathrm{[Hz]}$',
                         x_label_2='$e= f_{\mathrm{raft}}^{\mathrm{4}} - \hat{f}_{\mathrm{raft}}^{\mathrm{4}} \,\, \mathrm{[Hz]}$', 
                         y_label_2='$\mathrm{Density}$')
    func_Plot_Rgeresion(df=df.copy(),tn='rft_f5', scale=1.0, folder_name='All_1500',
                         x_label_1='$f_{\mathrm{raft}}^{\mathrm{5}}\,\, \mathrm{[Hz]}$', 
                         y_label_1='$\hat{f}_{\mathrm{raft}}^{\mathrm{5}}\,\, \mathrm{[Hz]}$',
                         x_label_2='$e= f_{\mathrm{raft}}^{\mathrm{5}} - \hat{f}_{\mathrm{raft}}^{\mathrm{5}} \,\, \mathrm{[Hz]}$', 
                         y_label_2='$\mathrm{Density}$')
    func_Plot_Rgeresion(df=df.copy(),tn='rft_f6', scale=1.0, folder_name='All_1500',
                         x_label_1='$f_{\mathrm{raft}}^{\mathrm{6}}\,\, \mathrm{[Hz]}$', 
                         y_label_1='$\hat{f}_{\mathrm{raft}}^{\mathrm{6}}\,\, \mathrm{[Hz]}$',
                         x_label_2='$e= f_{\mathrm{raft}}^{\mathrm{6}} - \hat{f}_{\mathrm{raft}}^{\mathrm{6}} \,\, \mathrm{[Hz]}$', 
                         y_label_2='$\mathrm{Density}$')
    func_Plot_Rgeresion(df=df.copy(),tn='AEP', scale=1.e-6, folder_name='All_1500',
                         x_label_1='$\mathrm{AEP}\,\, \mathrm{[MWh]}$', 
                         y_label_1='$\hat{\mathrm{AEP}}\,\, \mathrm{[MWH]}$',
                         x_label_2='$e= \mathrm{AEP} - \hat{\mathrm{AEP}}\,\, \mathrm{[MWh]}$', 
                         y_label_2='$\mathrm{Density}$')
    func_Plot_Rgeresion(df=df.copy(),tn='LCOE', scale=1000.0, folder_name='All_1500',
                         x_label_1='$\mathrm{LCOE}\,\, \mathrm{[USD/MWh]}$', 
                         y_label_1='$\hat{\mathrm{LCOE}}\,\, \mathrm{[USD/MWh]}$',
                         x_label_2='$e= \mathrm{LCOE} - \hat{\mathrm{LCOE}}\,\, \mathrm{[USD/MWh]}$', 
                         y_label_2='$\mathrm{Density}$') 

    D_main = np.arange(6,14+1,1)
    D_side = np.arange(10.5, 14.5+0.5, 0.5)
    n=len(D_main)
    D_main_mesh, D_side_mesh = np.meshgrid(D_main, D_side)
    n_data=n**2
    pnt_low_D=9.6148*np.ones(n_data)
    pnt_up_D=0.91*np.ones(n_data)
    ps=0.85*np.ones(n_data)
    Z_keel=-20.0*np.ones(n_data)
    Z_frbd=15.0*np.ones(n_data)
    Clmn_spcing=51.57*np.ones(n_data)
    Clmn_lngth=Z_frbd-Z_keel

    inputs=np.array([D_main_mesh.flatten(), D_side_mesh.flatten(), pnt_low_D, pnt_up_D, ps, Z_keel,  Z_frbd, Clmn_spcing, Clmn_lngth ]).T
    inputs_scaled=X_scale_func.transform(inputs)
    predicted_scaled=sm.predict_values(inputs_scaled[:,:-1])
    predicted_unscaled=Y_scale_func.inverse_transform(predicted_scaled)

    mp=predicted_unscaled[:,0].reshape(n,n)*1e-6
    tp_max=predicted_unscaled[:,1].reshape(n,n)
    max_nac_acc=predicted_unscaled[:,2].reshape(n,n)
    rft_f1=predicted_unscaled[:,9].reshape(n,n)
    rft_f2=predicted_unscaled[:,10].reshape(n,n)
    rft_f3=predicted_unscaled[:,11].reshape(n,n)
    rft_f4=predicted_unscaled[:,12].reshape(n,n)
    rft_f5=predicted_unscaled[:,13].reshape(n,n)
    rft_f6=predicted_unscaled[:,14].reshape(n,n)
    AEP=predicted_unscaled[:,15].reshape(n,n)*1e-6
    LCOE=predicted_unscaled[:,16].reshape(n,n)*1000

    plot_Mesh(Z_val=mp, x_mesh=D_main_mesh, y_mesh = D_side_mesh, folder_path=folder_path, filename='m_p_predict.jpg',
                         x_label='$D_\mathrm{Main- Clmn}\,\, \mathrm{[m]}$', 
                         y_label='$D_\mathrm{Side- Clmn}\,\, \mathrm{[m]}$', 
                         title='$\mathrm{platform\,mass} \,\, [\mathrm{Kilo \,\,Tons}]$',
                         highlight_x_value=10, highlight_y_value=12.5)
    plot_Mesh(Z_val=tp_max, x_mesh=D_main_mesh, y_mesh = D_side_mesh, folder_path=folder_path, filename='tp_max_predict.jpg',
                         x_label='$D_\mathrm{Main- Clmn}\,\, \mathrm{[m]}$', 
                         y_label='$D_\mathrm{Side- Clmn}\,\, \mathrm{[m]}$', 
                         title='$\mathrm{Max\,\,Ptfm\,\,Pitch\,\,[deg]}$',
                         highlight_x_value=10, highlight_y_value=12.5)
    plot_Mesh(Z_val=max_nac_acc, x_mesh=D_main_mesh, y_mesh = D_side_mesh, folder_path=folder_path, filename='max_nac_acc.jpg',
                         x_label='$D_\mathrm{Main- Clmn}\,\, \mathrm{[m]}$', 
                         y_label='$D_\mathrm{Side- Clmn}\,\, \mathrm{[m]}$', 
                         title='$\mathrm{Max\,\,nac\,\,acc\,\,[ms]}$',
                         highlight_x_value=10, highlight_y_value=12.5)
    plot_Mesh(Z_val=rft_f1*1000, x_mesh=D_main_mesh, y_mesh = D_side_mesh, folder_path=folder_path, filename='rft_f1.jpg',
                         x_label='$D_\mathrm{Main- Clmn}\,\, \mathrm{[m]}$', 
                         y_label='$D_\mathrm{Side- Clmn}\,\, \mathrm{[m]}$', 
                         title='$\mathrm{Raft\,\,Surge\,\,freq\,\,[mHz]}$',
                         highlight_x_value=10, highlight_y_value=12.5)
    plot_Mesh(Z_val=rft_f2*1000, x_mesh=D_main_mesh, y_mesh = D_side_mesh, folder_path=folder_path, filename='rft_f2.jpg',
                         x_label='$D_\mathrm{Main- Clmn}\,\, \mathrm{[m]}$', 
                         y_label='$D_\mathrm{Side- Clmn}\,\, \mathrm{[m]}$', 
                         title='$\mathrm{Raft\,\,Sway\,\,freq\,\,[mHz]}$',
                         highlight_x_value=10, highlight_y_value=12.5)
    plot_Mesh(Z_val=rft_f3*1000, x_mesh=D_main_mesh, y_mesh = D_side_mesh, folder_path=folder_path, filename='rft_f3.jpg',
                         x_label='$D_\mathrm{Main- Clmn}\,\, \mathrm{[m]}$', 
                         y_label='$D_\mathrm{Side- Clmn}\,\, \mathrm{[m]}$', 
                         title='$\mathrm{Raft\,\,Heave\,\,freq\,\,[mHz]}$',
                         highlight_x_value=10, highlight_y_value=12.5)
    plot_Mesh(Z_val=rft_f4*1000, x_mesh=D_main_mesh, y_mesh = D_side_mesh, folder_path=folder_path, filename='rft_f4.jpg',
                         x_label='$D_\mathrm{Main- Clmn}\,\, \mathrm{[m]}$', 
                         y_label='$D_\mathrm{Side- Clmn}\,\, \mathrm{[m]}$', 
                         title='$\mathrm{Raft\,\,Roll\,\,freq\,\,[mHz]}$',
                         highlight_x_value=10, highlight_y_value=12.5)
    plot_Mesh(Z_val=rft_f5*1000, x_mesh=D_main_mesh, y_mesh = D_side_mesh, folder_path=folder_path, filename='rft_f5.jpg',
                         x_label='$D_\mathrm{Main- Clmn}\,\, \mathrm{[m]}$', 
                         y_label='$D_\mathrm{Side- Clmn}\,\, \mathrm{[m]}$', 
                         title='$\mathrm{Raft\,\,Pitch\,\,freq\,\,[mHz]}$',
                         highlight_x_value=10, highlight_y_value=12.5)
    plot_Mesh(Z_val=rft_f6*1000, x_mesh=D_main_mesh, y_mesh = D_side_mesh, folder_path=folder_path, filename='rft_f6.jpg',
                         x_label='$D_\mathrm{Main- Clmn}\,\, \mathrm{[m]}$', 
                         y_label='$D_\mathrm{Side- Clmn}\,\, \mathrm{[m]}$', 
                         title='$\mathrm{Raft\,\,Yaw\,\,freq\,\,[mHz]}$',
                         highlight_x_value=10, highlight_y_value=12.5)
    plot_Mesh(Z_val=AEP, x_mesh=D_main_mesh, y_mesh = D_side_mesh, folder_path=folder_path, filename='AEP.jpg',
                         x_label='$D_\mathrm{Main- Clmn}\,\, \mathrm{[m]}$', 
                         y_label='$D_\mathrm{Side- Clmn}\,\, \mathrm{[m]}$', 
                         title='$\mathrm{AEP\,\,[MWh]}$',
                         highlight_x_value=10, highlight_y_value=12.5)
    plot_Mesh(Z_val=LCOE, x_mesh=D_main_mesh, y_mesh = D_side_mesh, folder_path=folder_path, filename='AEP.jpg',
                         x_label='$D_\mathrm{Main- Clmn}\,\, \mathrm{[m]}$', 
                         y_label='$D_\mathrm{Side- Clmn}\,\, \mathrm{[m]}$', 
                         title='$\mathrm{LCOE\,\,[USD/MWh]}$',
                         highlight_x_value=10, highlight_y_value=12.5)
    


    
    """norm = Normalize(vmin=mp.min(), vmax=mp.max())
    rgba_colors = ScalarMappable(norm=norm, cmap='Reds').to_rgba(mp, alpha=1)
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(rgba_colors, aspect='auto',origin= 'lower')
    # Set x ticks to side_clmns_mesh[0,:]
    x_indices = np.arange(0, mp.shape[1], 1)
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f'{D_main_mesh[0, i]:.1f}' for i in x_indices])
    # Set y ticks to side_clmns_mesh[:,0]
    y_indices = np.arange(0, mp.shape[0], 1)
    #y_indices = np.append(y_indices, y_indices[-1]+1)
    ax.set_yticks(y_indices)
    ax.set_yticklabels([f'{D_side_mesh[i, 0]:.1f}' for i in y_indices])
    ax.set_ylabel('$D_\mathrm{Side- Clmn}\,\, \mathrm{[m]}$', fontsize=18)
    ax.set_xlabel('$D_\mathrm{Main- Clmn}\,\, \mathrm{[m]}$', fontsize=18)
    ax.set_title('$\mathrm{platfrom\,mass} \,\, [\mathrm{Kilo \,\,Tons}]$', fontsize=15)
    # Customize the colorbar ticks
    cbar = plt.colorbar(ScalarMappable(norm=norm, cmap='Reds'), ax=ax)
    ax.plot(np.where(D_main_mesh[0,:]==10)[0][0],np.where(D_side_mesh[:,0]==12.5)[0][0] , 's', markersize=25, markerfacecolor='none', markeredgecolor='black', markeredgewidth =2)
    for i in range(mp.shape[0]):
        for j in range(mp.shape[1]):
            ax.text(j, i, f'{mp[i, j]:.1f}', ha='center', va='center', color='black', fontsize=12)
    fig.savefig(folder_path + '/' +'m_p_predict.jpg', format='jpg', dpi=300, bbox_inches='tight',pad_inches=0.55 )  

    norm = Normalize(vmin=tp_max.min(), vmax=tp_max.max())
    rgba_colors = ScalarMappable(norm=norm, cmap='Reds').to_rgba(tp_max, alpha=1)
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(rgba_colors, aspect='auto',origin= 'lower')
    # Set x ticks to side_clmns_mesh[0,:]
    x_indices = np.arange(0, tp_max.shape[1], 1)
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f'{D_main_mesh[0, i]:.1f}' for i in x_indices])
    # Set y ticks to side_clmns_mesh[:,0]
    y_indices = np.arange(0, tp_max.shape[0], 1)
    #y_indices = np.append(y_indices, y_indices[-1]+1)
    ax.set_yticks(y_indices)
    ax.set_yticklabels([f'{D_side_mesh[i, 0]:.1f}' for i in y_indices])
    ax.set_ylabel('$D_\mathrm{Side- Clmn}\,\, \mathrm{[m]}$', fontsize=18)
    ax.set_xlabel('$D_\mathrm{Main- Clmn}\,\, \mathrm{[m]}$', fontsize=18)
    ax.set_title('$\mathrm{Max\,\,Ptfm\,\,Pitch\,\,[deg]}$', fontsize=15)
    # Customize the colorbar ticks
    cbar = plt.colorbar(ScalarMappable(norm=norm, cmap='Reds'), ax=ax)
    ax.plot(np.where(D_main_mesh[0,:]==10)[0][0],np.where(D_side_mesh[:,0]==12.5)[0][0] , 's', markersize=25, markerfacecolor='none', markeredgecolor='black', markeredgewidth =2)
    for i in range(tp_max.shape[0]):
        for j in range(tp_max.shape[1]):
            ax.text(j, i, f'{tp_max[i, j]:.1f}', ha='center', va='center', color='black', fontsize=12)
    fig.savefig(folder_path + '/' +'tp_max_predict.jpg', format='jpg', dpi=300, bbox_inches='tight',pad_inches=0.55 )  
    """
    # Correlation study
    # Perform K-Means clustering to divide the data into 10 groups
    """n_clusters=80
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(df_train[['D_main', 'D_side']])
    kmeans_labels = kmeans.labels_
    kmeans_cluster_centers=kmeans.cluster_centers_
    correlations = calculate_correlations(df_train[['D_main', 'D_side']].values, df_train['tp_max'].values, kmeans, n_clusters)
    for i, group_correlation in enumerate(correlations, 1):
        print(f"Group {i}: Feature correlations with Y: {group_correlation}")

    correlations=np.array(correlations)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(df_train['D_main'], df_train['D_side'], alpha=0.5, label='Data Points')
    ax.scatter(kmeans_cluster_centers[:,0], kmeans_cluster_centers[:,1], c='red', label='Cluster Centers')
    # Add vectors for each cluster center
    for i in range(len(kmeans_cluster_centers)):
        ax.quiver(kmeans_cluster_centers[i, 0], kmeans_cluster_centers[i, 1],
              -correlations[i, 0], -correlations[i, 1],
              angles='xy', scale_units='xy', scale=1.5, color='blue', alpha=0.5 ,headwidth=3, headlength=5)
    # Set labels and legend
    ax.set_xlabel('D-Main')
    ax.set_ylabel('D-Side')
    ax.set_title('Cluster Centers and Correlations')
    ax.legend()
    ax.grid(True)"""

    # ----------------------------------- ------------------------------------------------
    # Run Optimizaiton to see Design Coupling 
    """# Pyoptsparse
    my_objfunc=lambda xdict: objfunc(xdict, X_scale_func, Y_scale_func)
    optProb = Optimization("OPT Problem Objective", my_objfunc)

    optProb.addVarGroup("D-Main", 1, "c", lower=[6.4], upper=[13.6], value=[10.0], scale=np.array([1.0/13.6]))
    optProb.addVarGroup("D-Side", 1, "c", lower=[10.7], upper=[16.1], value=[12.5], scale=np.array([1.0/16.1]))
    optProb.addConGroup("con", 1, lower=None, upper=6.8)
    optProb.addObj("obj")
    print(optProb)
    #optProb.printSparsity()
    #optOptions = {"IPRINT": -1, 'MAXIT': 1} # for SLSQP
    #optOptions = {"max_wall_time": 10} # for ipopt
    #opt=IPOPT()
    #optOptions = {"IPRINT": -1}
    opt = SLSQP()#options=optOptions)

    #max_wall_time
    #opt = OPT("IPOPT")
    #opt = SLSQP(options=optOptions)
    histroy_address = folder_path+ os.sep +  f"opt_history.hst"
    sol = opt(optProb, sens="FD", storeHistory= histroy_address)

    print(sol)
    print(sol.fStar)
    print(sol.xStar)"""

    # Use the COBYLA optimizer with constraints
    """my_objfunc=lambda x: objfunc_Cobyla(x, X_scale_func, Y_scale_func,sm)
    constraint = {'type': 'ineq', 'fun': lambda x: const_Cobyla(x, X_scale_func, Y_scale_func,sm)}
    constraints = [
    {'type': 'ineq', 'fun': lambda x: constraint1(x, X_scale_func, Y_scale_func,sm)},
    {'type': 'ineq', 'fun': lambda x: constraint2(x, X_scale_func, Y_scale_func,sm)}
]
    #x0 = [D_main, D_side, pnt_low_D, pnt_up_D, ps, Z_keel,  Z_frbd, Clmn_spcing]
    x0 = [10.0, 12.5, 9.6148, 0.91, 0.85, -20.0, 15.0, 51.57 ] # Initial guess
    #bounds = [(6.4, 13.6), (10.5, 14.5), (6.6148, 13.6148) , (0.71, 1.11) , (0.75, 1.0) , (-24.0, -16.0) , (7.0, 21.0) , (41.57, 61.57)]
    bounds = [(10.0, 10.0), (10.5, 14.5), (6.6148, 13.6148) , (0.71, 1.11) , (0.75, 1.0) , (-24.0, -16.0) , (7.0, 21.0) , (41.57, 61.57)]
    iter_details = [] # List to store iteration details
    # Callback function to store iteration details
    def store_iteration(x):
        iter_details.append(x)

    # Multi-start local optimization setup
    num_starts = 10  # Number of random starts
    best_solution = None
    best_objective = np.inf 

    for _ in range(num_starts):
        # Generate a random starting point within the bounds
        x0 = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds])
    
        # Perform local optimization from this starting point
        result = minimize(my_objfunc, x0, method='COBYLA', bounds=bounds, constraints=constraint, options={'disp': True, 'maxiter': 90000},callback=store_iteration) # 'maxiter': 1000
    
        # Update the best found solution if this result is better
        if result.fun < best_objective:
            #if const_Cobyla(result.x, X_scale_func, Y_scale_func,sm)>=0:
            if constraint1(result.x, X_scale_func, Y_scale_func,sm)>=0:
                if constraint2(result.x, X_scale_func, Y_scale_func,sm)>=0:
                    best_solution = result
                    best_objective = result.fun

    #result = minimize(my_objfunc, x0, method='COBYLA', bounds=bounds, constraints=constraint,callback=store_iteration)
    #constraint_value = const_Cobyla(result.x, X_scale_func, Y_scale_func,sm)
    # Print the result
    #print("Optimal value:", result.fun)
    #print("Optimal point:", result.x)
    #for i, x in enumerate(iter_details):
    #    print(f"Iteration {i+1}: x = {x}, f(x) = {my_objfunc(x)}")
    print(f"Best solution: {best_solution.x}")
    #print(f"Best solution: {np.round(best_solution.x * 10) / 10.0 }")
    print(f"Best objective value: {best_solution.fun}")
    constraint_value = const_Cobyla(best_solution.x, X_scale_func, Y_scale_func,sm)
    print("Constraint value at optimal point:", constraint_value)
    k=1"""

    # Use pulp
    # Define the bounds for each variable
    # Assuming the objective function, constraints, and other dependencies are defined:
    def my_objfunc(x):
        return (objfunc_Cobyla(x, X_scale_func, Y_scale_func, sm),) # Your objective function as tuple
    def constraint1_func(x):
        return constraint1(x, X_scale_func, Y_scale_func, sm)  # Your first constraint
    def constraint2_func(x):
        return constraint2(x, X_scale_func, Y_scale_func, sm)  # Your second constraint
    
    #variable_bounds = [(6.0+1.0, 14.0-0.1), (10.5, 14.5-0.1), (6.6148, 13.6148-0.1) , (0.71, 1.11-0.01) , (0.75, 1.0-0.06) , (-24.0, -16.0-.1) , (7.0, 21.0-0.1) , (41.57, 61.57-0.1)] #step 1
    variable_bounds = [(6.0+1.0, 14.0-0.01), (10.5, 14.5-0.01), (6.6148, 13.6148-0.01) , (0.71, 1.11-0.001) , (0.75, 1.0-0.006) , (-24.0, -16.0-.01) , (7.0, 21.0-0.01) , (41.57, 61.57-0.01)] #step 0.1
    #variable_bounds = [(6.0+1.0, 14.0-0.01), (10.5, 14.5-0.01), (6.6148, 13.6148-0.01) , (0.71, 1.11-0.001) , (0.75, 1.0-0.006) , (-24.0, -16.0-.01) , (7.0, 21.0-0.01)  , (51.57, 51.57)] 
    #variable_bounds = [(6.0+1.0, 14.0-0.1), (10.5, 14.5-0.1), (6.6148, 13.6148-0.1) , (0.71, 1.11-0.01) , (0.75, 1.0-0.06) , (-24.0, -16.0-.1) , (7.0, 21.0-0.1), (51.57, 51.57)]
    #step_sizes = [1.0, 1.0, 1.0, 0.1, 0.1, 2.0, 2.0, 2.0]  # Define step sizes for each variable #step 1.0
    step_sizes = [0.1, 0.1, 0.1, 0.01, 0.01, 0.1, 0.1, 0.1]  # Define step sizes for each variable #step 0.1

    # Define the GA's fitness and individual
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    # Define bounds and step sizes for your floating-point variables
    
    # Helper function to create an individual
    def create_individual():
        individual = []
        for (low, up), step in zip(variable_bounds, step_sizes):
            # Generate a value within the bounds using the step size
            val_range = np.arange(low, up + step, step)
            individual.append(random.choice(val_range))
        return creator.Individual(individual)
    
    variable_sets=[]
    for (low, up), step in zip(variable_bounds, step_sizes):
        variable_sets.append(np.arange(low, up + step, step))
    
    """variable_sets = [
    np.array([7, 8, 9, 10, 11, 12, 13]),
    np.array([11, 12, 13, 14, 15]),
    np.array([7, 8, 9, 10, 11, 12, 13, 14]),
    np.array([0.71, 0.81, 0.91, 1.01, 1.11]),
    np.array([0.75, 0.85, 0.95]),
    np.array([-24, -22, -20, -18, -16]),
    np.array([7, 9, 11, 13, 15, 17, 19, 21]),
    np.array([41.57, 43.57, 45.57, 47.57, 49.57, 51.57, 53.57, 55.57, 57.57, 59.57, 61.57])
    ]"""
    def custom_mutate(individual, indpb):
        """Mutate an individual by picking a new value from its set with probability indpb."""
        for i in range(len(individual)):
            if random.random() < indpb:
                # Set the new value from the variable set
                individual[i] = random.choice(variable_sets[i])
        return individual,
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # Genetic operators
    toolbox.register("evaluate", my_objfunc)
    toolbox.register("mate", tools.cxUniform, indpb=0.0) #0.5)  # Uniform crossover--!!!!!!! changed this
    # Convert your variable sets to numpy arrays for efficient distance calculation
    variable_sets_np = [np.array(var_set) for var_set in variable_sets]
    #toolbox.register("mate", custom_crossover, variable_sets=variable_sets_np)
    #toolbox.register("mutate", tools.mutPolynomialBounded, low=[b[0] for b in variable_bounds], up=[b[1] for b in variable_bounds], eta=1.0, indpb=0.2)
    toolbox.register("mutate", custom_mutate, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    # Check for feasibility of constraints
    def check_constraints(individual):
        return constraint1_func(individual) >= 0 and constraint2_func(individual) >= 0
    toolbox.decorate("evaluate", tools.DeltaPenalty(check_constraints, 1e6, my_objfunc))
    # Optimization process
    def optimize():
        pop = toolbox.population(n=200)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.1, ngen=100, stats=stats, halloffame=hof, verbose=True)
        return pop, stats, hof
    pop, stats, hof = optimize()
    print("Best individual is:", hof[0], "with fitness:", hof[0].fitness.values)
    print("Const1:", constraint1_func(hof[0]), "Const2:", constraint1_func(hof[0]))