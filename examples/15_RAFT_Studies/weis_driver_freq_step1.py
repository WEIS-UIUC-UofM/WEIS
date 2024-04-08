import os
import time
import sys

from weis.glue_code.runWEIS     import run_weis
from wisdem.commonse.mpi_tools  import MPI
import numpy as np
import pickle

## File management
run_dir                = os.path.dirname( os.path.realpath(__file__) ) + os.sep
fname_wt_input         = os.path.join(run_dir,"..","06_IEA-15-240-RWT", "IEA-15-240-RWT_VolturnUS-S.yaml")
fname_modeling_options = run_dir + "modelling_options_Freq_step1.yaml"
fname_analysis_options = run_dir + "analysis_options_Freq_step1.yaml"

#load all possible names to override the design
"""all_Possible_Pvars_names_address=os.path.dirname(os.path.dirname( os.path.realpath(__file__) )) + os.sep + 'all_Possible_Pvars_names.pkl'
with open(all_Possible_Pvars_names_address, 'rb') as f:
    all_Possible_Pvars_names = pickle.load(f)
# supopse we want to change rotor diamaeter--> we need to dinf the corrpsoing name
rotor_elements =[element for element in all_Possible_Pvars_names if 'rotor_diameter' in element]"""

overridden_values = {}
#overridden_values['wisdem.wt.wt_init.floating.memgrp0.outer_diameter_in'] = np.array([7.0]) #main column diametr #10 7
#overridden_values['wisdem.wt.wt_init.floating.memgrp1.outer_diameter_in'] = np. array([11.1]) #1st side collumn diameter # 12.5 11.1
#overridden_values['wisdem.wt.wt_init.floating.joints.jointdv_0'] = np. array([51.75]) # 51.75 columns spacing

#overridden_values['tune_rosco_ivc.ps_percent'] = np.array([0.85])  #0.85
#overridden_values['tune_rosco_ivc.omega_pc']=np.array([0.2]) # 0.2
#overridden_values['tune_rosco_ivc.zeta_pc']=np.array([0.7]) # 1.0
#overridden_values['tune_rosco_ivc.omega_vs']=np.array([0.05]) # 0.12
#overridden_values['tune_rosco_ivc.zeta_vs']=np.array([0.6]) # 0.85

#1500 case:
"""overridden_values['wisdem.wt.wt_init.floating.joints.jointdv_0']= np.array([-16.0])  #Clmns_keel
overridden_values['wisdem.wt.wt_init.floating.joints.jointdv_1']= np.array([13.2])  #Clmns_frbd
overridden_values['wisdem.wt.wt_init.floating.joints.jointdv_2']= np.array([59.27])  #Clmns_spc
overridden_values['tune_rosco_ivc.ps_percent'] = np.array([1.0])  #ps
overridden_values['wisdem.wt.wt_init.floating.memgrp0.outer_diameter_in']= np.array([7.0])  #main_clmn_D
overridden_values['wisdem.wt.wt_init.floating.memgrp1.outer_diameter_in']= np.array([11.1])  #side_clmn_D
overridden_values['floating.memgrp2.outer_diameter_in']= np.array([0.87])  #Y_pntoon_upper_D
overridden_values['floating.memgrp3.outer_diameter_in']= np.array([6.71])  #Y_pntoon_lower_D"""

#1500 case-  tp and max_acc constarint the same as the baseline:
"""overridden_values['wisdem.wt.wt_init.floating.joints.jointdv_0']= np.array([-16.7])  #Clmns_keel
overridden_values['wisdem.wt.wt_init.floating.joints.jointdv_1']= np.array([13.5])  #Clmns_frbd
overridden_values['wisdem.wt.wt_init.floating.joints.jointdv_2']= np.array([59.07])  #Clmns_spc
overridden_values['tune_rosco_ivc.ps_percent'] = np.array([0.99])  #ps
overridden_values['wisdem.wt.wt_init.floating.memgrp0.outer_diameter_in']= np.array([7.1])  #main_clmn_D
overridden_values['wisdem.wt.wt_init.floating.memgrp1.outer_diameter_in']= np.array([10.9])  #side_clmn_D
overridden_values['floating.memgrp2.outer_diameter_in']= np.array([0.86])  #Y_pntoon_upper_D
overridden_values['floating.memgrp3.outer_diameter_in']= np.array([6.7148])  #Y_pntoon_lower_D"""

#test controller
#overridden_values['tune_rosco_ivc.omega_pc'] = 0.9 #0.2
#overridden_values['tune_rosco_ivc.zeta_pc'] = 0.5 #1.0
#overridden_values['tune_rosco_ivc.omega_vs'] = 0.6 #0.12
#overridden_values['tune_rosco_ivc.zeta_vs'] =0.1 #0.85
#overridden_values['tune_rosco_ivc.ps_percent'] = 0.85 #0.85
#overridden_values['tune_rosco_ivc.Kp_float'] = -5.0 #-10
#overridden_values['tune_rosco_ivc.ptfm_freq'] = 0.2 #0.2

##overridden_values['sse_tune.tune_rosco.VS_Kp'] = -63647384.51368386 #these are updated automatically
##overridden_values['sse_tune.tune_rosco.VS_Ki'] = -5039983.48064195
##overridden_values['sse_tune.tune_rosco.PC_Kp'] = -1.45477175
##overridden_values['sse_tune.tune_rosco.PC_Ki'] = -0.14824348



tt = time.time()
wt_opt, modeling_options, opt_options = run_weis(fname_wt_input, fname_modeling_options, fname_analysis_options, overridden_values)

"""model = wt_opt.model
all_Possible_Pvars=model.list_outputs()
all_Possible_Pvars_names=[all_Possible_Pvars[i][0] for i in np.arange(len(all_Possible_Pvars))]
[element for element in all_Possible_Pvars_names if 'rosco' in element]"""

Clmns_keel=wt_opt['wisdem.wt.wt_init.floating.joints.jointdv_0']
Clmns_frbd=wt_opt['wisdem.wt.wt_init.floating.joints.jointdv_1']
Clmns_spc=wt_opt['wisdem.wt.wt_init.floating.joints.jointdv_2']
ps=wt_opt['tune_rosco_ivc.ps_percent'] 
main_clmn_D=wt_opt['wisdem.wt.wt_init.floating.memgrp0.outer_diameter_in']
side_clmn_D=wt_opt['wisdem.wt.wt_init.floating.memgrp1.outer_diameter_in']
Y_pntoon_upper_D=wt_opt['floating.memgrp2.outer_diameter_in']
Y_pntoon_lower_D=wt_opt['floating.memgrp3.outer_diameter_in']
Clmns_length=Clmns_frbd-Clmns_keel

platform_mass= wt_opt['wisdem.wt.floatingse.sys.mux.platform_mass']
tp_max= wt_opt['raft.raft.Max_PtfmPitch']
max_nac_acc=wt_opt['raft.raft.max_nac_accel']


#wt_opt['wisdem.wt.towerse.tower.structural_frequencies']
#wt_opt['wisdem.wt.floatingse.load.tower.structural_frequencies']
#wt_opt['wisdem.wt.rotorse.rs.frame.freqs']
#'tune_rosco_ivc.omega_pc'
#01:
#'tune_rosco_ivc.zeta_pc'
#02:
#'tune_rosco_ivc.omega_vs'
#03:
#'tune_rosco_ivc.zeta_vs'

if MPI:
    rank = MPI.COMM_WORLD.Get_rank()
else:
    rank = 0
if rank == 0:
    print('Run time: %f'%(time.time()-tt))
    sys.stdout.flush()
