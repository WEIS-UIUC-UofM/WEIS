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
#overridden_values['tune_rosco_ivc.omega_pc']=np.array([0.1]) # 0.2
#overridden_values['tune_rosco_ivc.zeta_pc']=np.array([0.7]) # 1.0
#overridden_values['tune_rosco_ivc.omega_vs']=np.array([0.05]) # 0.12
#overridden_values['tune_rosco_ivc.zeta_vs']=np.array([0.6]) # 0.85

tt = time.time()
wt_opt, modeling_options, opt_options = run_weis(fname_wt_input, fname_modeling_options, fname_analysis_options, overridden_values)

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
