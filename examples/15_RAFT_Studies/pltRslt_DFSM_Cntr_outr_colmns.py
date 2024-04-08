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
from smt.surrogate_models import KPLSK
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


import numpy as np
import pandas as pd 
import multiprocessing as mp 
import pickle as pkl
import matplotlib.pyplot as plt

import openmdao.api as om
from weis.aeroelasticse import FileTools

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
    x=np.round(x * 10) / 10.0  # Adjusting for step size 0.1
    D_Main = x[0]
    D_Side = x[1]

    Feature = pd.DataFrame()
    Feature['D_main']=np.array([D_Main])
    Feature['D_side']=np.array([D_Side])
    #Feature=np.array([[D_Main[0], D_Side[0]]])
    Feature_s= X_scale_func.transform(Feature[['D_main', 'D_side']])

    Outputs_s =sm.predict_values(Feature_s)
    Outputs=Y_scale_func.inverse_transform(Outputs_s)

    mp=Outputs[0][0] *1e-6 # mp [KTonns]
    tp_max=Outputs[0][1]  #tp_max [deg]

    return mp

def const_Cobyla(x, X_scale_func, Y_scale_func,sm):
    x=np.round(x * 10) / 10.0  # Adjusting for step size 0.1
    D_Main = x[0]
    D_Side = x[1]

    Feature = pd.DataFrame()
    Feature['D_main']=np.array([D_Main])
    Feature['D_side']=np.array([D_Side])
    #Feature=np.array([[D_Main[0], D_Side[0]]])
    Feature_s= X_scale_func.transform(Feature[['D_main', 'D_side']])

    Outputs_s =sm.predict_values(Feature_s)
    Outputs=Y_scale_func.inverse_transform(Outputs_s)

    mp=Outputs[0][0] *1e-6 # mp [KTonns]
    tp_max=Outputs[0][1]  #tp_max [deg]

    return 6.8-tp_max



def func_Plot_Rgeresion(df,tn='m_p', unit='KTonns', folder_name='column_side_and_main'): #tn: target name
    folder_path = os.getcwd()+'/outputs/'+ folder_name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    fig, ax = plt.subplots(1,1)
    ax.plot(df[tn], df[tn+'_prdict'],'o',markersize=1.0)
    ax.plot([df[tn].min(), df[tn].max()], [df[tn].min(), df[tn].max()],color='k', alpha=0.3)
    mae = mean_absolute_error(df[tn], df[tn+'_prdict'])#/df[tn].mean() # I normalized that !!!!!!!!!----I removed normalization-------!!!!!!!!!!!!
    mse = mean_squared_error(df[tn], df[tn+'_prdict'])#/df[tn].mean()**2 # I normalized that !!!!!!!!!----I removed normalization-------!!!!!!!!!!!!
    rmse = np.sqrt(mse)
    r2 = r2_score(df[tn], df[tn+'_prdict'])
    ax.set_xlabel(rf'${tn} \,\, [\mathrm{{{unit}}}]$', fontsize=18)
    ax.set_ylabel(rf'$\hat{{{tn}}} \,\, [\mathrm{{{unit}}}]$', fontsize=18)
    if any('train' in col for col in df.columns):
        ax.set_title(rf'$\mathrm{{Training \,\, Data}} \,\,\, , \,\, R^2 \, : \, {r2:.2f} $', fontsize=18)
        fig.savefig(folder_path +'/'+tn+'_r2_train.jpg', format='jpg', dpi=300, bbox_inches='tight',pad_inches=0.1)  
    elif any('test' in col for col in df.columns):
        ax.set_title(rf'$\mathrm{{Test \,\, Data}} \,\,\, , \,\, R^2 \, : \, {r2:.2f} $', fontsize=18)
        fig.savefig(folder_path +'/'+tn+'_r2_test.jpg', format='jpg', dpi=300, bbox_inches='tight',pad_inches=0.1)  
    else:
        ax.set_title(rf'$\mathrm{{All \,\, Data}} \,\,\, , \,\, R^2 \, : \, {r2:.2f} $', fontsize=18)
        fig.savefig(folder_path +'/'+tn+'_r2_all.jpg', format='jpg', dpi=300, bbox_inches='tight',pad_inches=0.1)  
    fig, ax = plt.subplots(1,1)
    e = (df[tn] - df[tn+'_prdict'])#/df[tn].mean() !!!!!!!!!----I removed normalization-------!!!!!!!!!!!!
    mean_e = np.mean(e)
    var_e = np.var(e)
    ax.hist(e, bins=40, density=True, alpha=0.6, color='g', label='Histogram')
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mean_e, np.sqrt(var_e))
    ax.plot(x, p, 'k', linewidth=2, label='Fit')
    ax.set_xlabel(rf'$e=({tn}-\hat{{{tn}}}) / \, \overline{{{tn}}}  $', fontsize=18)
    ax.set_ylabel('$\mathrm{Density}$', fontsize=18)
    if any('train' in col for col in df.columns):
        ax.set_title(rf'$\mathrm{{Training \,\, Data}} \,\,\, , \,\, e_{{\mathrm{{mean}}}} \, : \, {mean_e:.2f} \,\,\, , \,\, e_{{\mathrm{{var}}}} \, : \, {var_e:.2f} $', fontsize=14)
        fig.savefig(folder_path +'/'+tn+'_gauss_train.jpg', format='jpg', dpi=300, bbox_inches='tight',pad_inches=0.1)  
    elif any('test' in col for col in df.columns):
        ax.set_title(rf'$\mathrm{{Test \,\, Data}} \,\,\, , \,\, e_{{\mathrm{{mean}}}} \, : \, {mean_e:.2f} \,\,\, , \,\, e_{{\mathrm{{var}}}} \, : \, {var_e:.2f} $', fontsize=14)
        fig.savefig(folder_path +'/'+tn+'_gauss_test.jpg', format='jpg', dpi=300, bbox_inches='tight',pad_inches=0.1)  
    else:
        ax.set_title(rf'$\mathrm{{All \,\, Data}} \,\,\, , \,\, e_{{\mathrm{{mean}}}} \, : \, {mean_e:.2f} \,\,\, , \,\, e_{{\mathrm{{var}}}} \, : \, {var_e:.2f} $', fontsize=14)
        fig.savefig(folder_path +'/'+tn+'_gauss_all.jpg', format='jpg', dpi=300, bbox_inches='tight',pad_inches=0.1)  


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
    output_dir = os.path.join(run_dir, "outputs/15_RAFT_optimization_freq_step1_Cntr_Clmns_and_Outr_Clmns_Reduced_included") # 15_RAFT_optimization_freq_step1_clmnSpacing_sideColmnsdiam_lowPntoon_diam") #opt coByla
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
    main_column_outer_diameter_vec=np.squeeze(outdata[0]['floating.memgrp0.outer_diameter_in'])
    side_columns_outer_diameter_vec=np.squeeze(outdata[0]['floating.memgrp1.outer_diameter_in'])
    #--------------------------------------------------------------------------------------

    #-------------------------------------Constraints------------------------------------
    Max_PtfmPitch_vc=np.squeeze(outdata[0]['raft.Max_PtfmPitch'])
    #Max_PtfmPitch_vc[Max_PtfmPitch_vc > 10] = 10
    Std_PtfmPitch_vec=np.squeeze(outdata[0]['raft.Std_PtfmPitch'])
    #Std_PtfmPitch_vec[Std_PtfmPitch_vec > 2] = 2.0
    height_constraint_vec=np.squeeze(outdata[0]['towerse.height_constraint'])
    tower_slope_vec=np.squeeze(outdata[0]['towerse.slope'])
    nacelle_acceleration_vec=np.squeeze(outdata[0]['raft.response_nacelle acceleration'])
    #--------------------------------------------------------------------------------------

    #-------------------------------------Objectives------------------------------------
    floatingse_platform_mass_vec=np.squeeze(outdata[0]['floatingse.platform_mass'])

    #--------------------------------------------------------------------------------------
    df = pd.DataFrame()
    df['D_main'] = main_column_outer_diameter_vec
    df['D_side'] = side_columns_outer_diameter_vec
    #df['Ypntoon_lower_Raw'] = Y_pntoon_lower_vec
    #df['Clmns_spacing_Raw'] = Clmns_spacing_vec
    #df['D_side'] = side_columns_outer_diameter_vec
    df['m_p'] = floatingse_platform_mass_vec
    df['tp_max'] = Max_PtfmPitch_vc
    #df['tp_max'] = df['tp_max'].fillna(12).clip(upper=12)
    df = df[df['tp_max'] <= 12.0]

    #Features = df[['Ypntoon_lower_Raw', 'Clmns_spacing_Raw', 'D_side']]
    Features = df[['D_main', 'D_side']]
    Targets = df[['m_p', 'tp_max']]


    X_train, X_test, y_train, y_test = train_test_split(Features, Targets, test_size=0.3, random_state=42, shuffle=True)
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    df_train['D_main']=X_train.values[:,0]
    df_train['D_side']=X_train.values[:,1]
    df_train['m_p']=y_train.values[:,0]
    df_train['tp_max']=y_train.values[:,1]
    df_train['train_indices']=X_train.index.values

    df_test['D_main']=X_test.values[:,0]
    df_test['D_side']=X_test.values[:,1]
    df_test['m_p']=y_test.values[:,0]
    df_test['tp_max']=y_test.values[:,1]
    df_test['test_indices']=X_test.index.values

    X_scale_func = StandardScaler()
    Y_scale_func = StandardScaler()

    X_scale_func.fit_transform(df_train[['D_main', 'D_side']])
    Y_scale_func.fit_transform(df_train[['m_p', 'tp_max']])

    #scaling features and targets to make them ready for training
    df_train[['D_main_s','D_side_s']] = X_scale_func.transform(df_train[['D_main', 'D_side']])
    df_train[['m_p_s','tp_max_s']] = Y_scale_func.transform(df_train[['m_p', 'tp_max']])
    df_test[['D_main_s','D_side_s']]  = X_scale_func.transform(df_test[['D_main', 'D_side']])
    df_test[['m_p_s','tp_max_s']]  = Y_scale_func.transform(df_test[['m_p', 'tp_max']])

    sm = KPLSK(theta0=[1e-2],print_training=False)
    sm.options['print_prediction'] = False
    sm.set_training_values(df_train[['D_main_s', 'D_side_s']].values, df_train[['m_p_s', 'tp_max_s']].values)
    sm.train()
    #filtered_df = df[(df['D_main'] == 10) & (df['D_side'] == 12.5)]
    # esimate tagregts and tehir variance
    df_train[['m_p_s_prdict','tp_max_s_prdict']] =sm.predict_values(df_train[['D_main_s','D_side_s']].values)
    df_train[['var_m_p_s_prdict','var_tp_max_s_prdict']] = sm.predict_variances(df_train[['D_main_s','D_side_s']].values) # estimated variance
    df_test[['m_p_s_prdict','tp_max_s_prdict']] =sm.predict_values(df_test[['D_main_s','D_side_s']].values)
    df_test[['var_m_p_s_prdict','var_tp_max_s_prdict']] = sm.predict_variances(df_test[['D_main_s','D_side_s']].values) # estimated variance

    #unscale targets
    df_train[['m_p_prdict','tp_max_prdict']]=Y_scale_func.inverse_transform(df_train[['m_p_s_prdict','tp_max_s_prdict']])
    df_test[['m_p_prdict','tp_max_prdict']]=Y_scale_func.inverse_transform(df_test[['m_p_s_prdict','tp_max_s_prdict']])


    # all data
    df[['D_main_s','D_side_s']] = X_scale_func.transform(df[['D_main', 'D_side']])
    df[['m_p_s','tp_max_s']] = Y_scale_func.transform(df[['m_p', 'tp_max']])
    df[['m_p_s_prdict','tp_max_s_prdict']] =sm.predict_values(df[['D_main_s','D_side_s']].values)
    df[['m_p_prdict','tp_max_prdict']]=Y_scale_func.inverse_transform(df[['m_p_s_prdict','tp_max_s_prdict']])

    folder_path = os.getcwd()+'/outputs/'+ 'column_side_and_main'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    func_Plot_Rgeresion(df=df_train, tn='m_p', unit='KTonns', folder_name='column_side_and_main')
    func_Plot_Rgeresion(df=df_train, tn='tp_max', unit='deg', folder_name='column_side_and_main')
    func_Plot_Rgeresion(df=df_test, tn='m_p', unit='KTonns', folder_name='column_side_and_main')
    func_Plot_Rgeresion(df=df_test, tn='tp_max', unit='deg', folder_name='column_side_and_main')
    func_Plot_Rgeresion(df=df, tn='m_p', unit='KTonns', folder_name='column_side_and_main')
    func_Plot_Rgeresion(df=df, tn='tp_max', unit='deg', folder_name='column_side_and_main')

    main_clmns_mesh=main_column_outer_diameter_vec.reshape((41,41))[11::3, 2::3]#[11:,:]
    side_clmns_mesh=side_columns_outer_diameter_vec.reshape((41,41))[11::3, 2::3]
    Max_PtfmPitch_vc_mesh=Max_PtfmPitch_vc.reshape((41,41))[11::3, 2::3]
    floatingse_platform_mass_vec_mesh=floatingse_platform_mass_vec.reshape((41,41))[11::3, 2::3]*1e-6

    inputs=np.array([main_clmns_mesh.flatten(), side_clmns_mesh.flatten()]).T
    inputs_scaled=X_scale_func.transform(inputs)
    predicted_scaled=sm.predict_values(inputs_scaled)
    predicted_unscaled=Y_scale_func.inverse_transform(predicted_scaled)
    max_ptf_pitch_mesh_predicted=predicted_unscaled[:,1].reshape(Max_PtfmPitch_vc_mesh.shape)
    floatingse_platform_mass_vec_mesh_predicted=predicted_unscaled[:,0].reshape(floatingse_platform_mass_vec_mesh.shape)*1e-6

    # Normalize the data
    norm = Normalize(vmin=max_ptf_pitch_mesh_predicted.min(), vmax=max_ptf_pitch_mesh_predicted.max())
    rgba_colors = ScalarMappable(norm=norm, cmap='Reds').to_rgba(max_ptf_pitch_mesh_predicted, alpha=1)
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(rgba_colors, aspect='auto',origin= 'lower')
    # Set x ticks to side_clmns_mesh[0,:]
    x_indices = np.arange(0, max_ptf_pitch_mesh_predicted.shape[1], 2)
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f'{main_clmns_mesh[0, i]:.1f}' for i in x_indices])
    # Set y ticks to side_clmns_mesh[:,0]
    y_indices = np.arange(0, max_ptf_pitch_mesh_predicted.shape[0], 2)
    y_indices = np.append(y_indices, y_indices[-1]+1)
    ax.set_yticks(y_indices)
    ax.set_yticklabels([f'{side_clmns_mesh[i, 0]:.1f}' for i in y_indices])
    ax.set_ylabel('$D_\mathrm{Side- Clmn}\,\, \mathrm{[m]}$', fontsize=18)
    ax.set_xlabel('$D_\mathrm{Main- Clmn}\,\, \mathrm{[m]}$', fontsize=18)
    ax.set_title('$\mathrm{Max\,\,Ptfm\,\,Pitch\,\,[deg]}$', fontsize=15)
    # Customize the colorbar ticks
    cbar = plt.colorbar(ScalarMappable(norm=norm, cmap='Reds'), ax=ax)
    ax.plot(np.where(main_clmns_mesh[0,:]==10)[0][0],np.where(side_clmns_mesh[:,0]==12.5)[0][0] , 's', markersize=25, markerfacecolor='none', markeredgecolor='black', markeredgewidth =2)
    for i in range(max_ptf_pitch_mesh_predicted.shape[0]):
        for j in range(max_ptf_pitch_mesh_predicted.shape[1]):
            ax.text(j, i, f'{max_ptf_pitch_mesh_predicted[i, j]:.1f}', ha='center', va='center', color='black', fontsize=12)
    fig.savefig(folder_path + '/' +'tp_max_predict.jpg', format='jpg', dpi=300, bbox_inches='tight',pad_inches=0.55 )  

    # Normalize the data
    norm = Normalize(vmin=floatingse_platform_mass_vec_mesh_predicted.min(), vmax=floatingse_platform_mass_vec_mesh_predicted.max())
    rgba_colors = ScalarMappable(norm=norm, cmap='Reds').to_rgba(floatingse_platform_mass_vec_mesh_predicted, alpha=1)
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(rgba_colors, aspect='auto',origin= 'lower')
    # Set x ticks to side_clmns_mesh[0,:]
    x_indices = np.arange(0, floatingse_platform_mass_vec_mesh_predicted.shape[1], 2)
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f'{main_clmns_mesh[0, i]:.1f}' for i in x_indices])
    # Set y ticks to side_clmns_mesh[:,0]
    y_indices = np.arange(0, floatingse_platform_mass_vec_mesh_predicted.shape[0], 2)
    y_indices = np.append(y_indices, y_indices[-1]+1)
    ax.set_yticks(y_indices)
    ax.set_yticklabels([f'{side_clmns_mesh[i, 0]:.1f}' for i in y_indices])
    ax.set_ylabel('$D_\mathrm{Side- Clmn}\,\, \mathrm{[m]}$', fontsize=15)
    ax.set_xlabel('$D_\mathrm{Main- Clmn}\,\, \mathrm{[m]}$', fontsize=15)
    ax.set_title('$\mathrm{platfrom\,mass} \,\, [\mathrm{Kilo \,\,Tons}]$', fontsize=15)
    # Customize the colorbar ticks
    cbar = plt.colorbar(ScalarMappable(norm=norm, cmap='Reds'), ax=ax)
    ax.plot(np.where(main_clmns_mesh[0,:]==10)[0][0],np.where(side_clmns_mesh[:,0]==12.5)[0][0] , 's', markersize=30, markerfacecolor='none', markeredgecolor='black', markeredgewidth =2)
    for i in range(floatingse_platform_mass_vec_mesh_predicted.shape[0]):
        for j in range(floatingse_platform_mass_vec_mesh_predicted.shape[1]):
            ax.text(j, i, f'{floatingse_platform_mass_vec_mesh_predicted[i, j]:.1f}', ha='center', va='center', color='black', fontsize=12)
    fig.savefig(folder_path + '/'+'m_p_predict.jpg', format='jpg', dpi=300, bbox_inches='tight' ,pad_inches=0.55)


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
    my_objfunc=lambda x: objfunc_Cobyla(x, X_scale_func, Y_scale_func,sm)
    constraint = {'type': 'ineq', 'fun': lambda x: const_Cobyla(x, X_scale_func, Y_scale_func,sm)}
    x0 = [10.0, 12.5] # Initial guess
    #bounds = [(6.4, 13.6), (10.7, 16.1)]
    bounds = [(6.4, 13.6), (10.7, 16.1)]
    iter_details = [] # List to store iteration details
    # Callback function to store iteration details
    def store_iteration(x):
        iter_details.append(x)

    # Multi-start local optimization setup
    num_starts = 70  # Number of random starts
    best_solution = None
    best_objective = np.inf 

    for _ in range(num_starts):
        # Generate a random starting point within the bounds
        x0 = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds])
    
        # Perform local optimization from this starting point
        result = minimize(my_objfunc, x0, method='COBYLA', bounds=bounds, constraints=constraint, options={'disp': True},callback=store_iteration)
    
        # Update the best found solution if this result is better
        if result.fun < best_objective:
            if const_Cobyla(result.x, X_scale_func, Y_scale_func,sm)>=0:
                best_solution = result
                best_objective = result.fun

    """result = minimize(my_objfunc, x0, method='COBYLA', bounds=bounds, constraints=constraint,callback=store_iteration)
    constraint_value = const_Cobyla(result.x, X_scale_func, Y_scale_func,sm)
    # Print the result
    print("Optimal value:", result.fun)
    print("Optimal point:", result.x)"""
    for i, x in enumerate(iter_details):
        print(f"Iteration {i+1}: x = {x}, f(x) = {my_objfunc(x)}")
    print(f"Best solution: {best_solution.x}")
    print(f"Best solution: {np.round(best_solution.x * 10) / 10.0 }")
    print(f"Best objective value: {best_solution.fun}")
    constraint_value = const_Cobyla(best_solution.x, X_scale_func, Y_scale_func,sm)
    print("Constraint value at optimal point:", constraint_value)
    k=1