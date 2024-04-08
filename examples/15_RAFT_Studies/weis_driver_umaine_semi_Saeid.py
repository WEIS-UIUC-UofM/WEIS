import os
import time
import sys

from weis.glue_code.runWEIS     import run_weis
from wisdem.commonse.mpi_tools  import MPI
import pickle
import numpy as np
import concurrent.futures

## File management
run_dir                = os.path.dirname( os.path.realpath(__file__) ) + os.sep
fname_wt_input         = os.path.realpath(os.path.join(run_dir,"IEA-15-floating.yaml"))
fname_modeling_options = run_dir + "modeling_options_umaine_semi_Saeid.yaml"
fname_analysis_options = run_dir + "analysis_options_Saeid.yaml"

#load all possible names to override the design
all_Possible_Pvars_names_address=os.path.dirname(os.path.dirname( os.path.realpath(__file__) )) + os.sep + 'all_Possible_Pvars_names.pkl'
with open(all_Possible_Pvars_names_address, 'rb') as f:
    all_Possible_Pvars_names = pickle.load(f)
# supopse we want to change rotor diamaeter--> we need to dinf the corrpsoing name
rotor_elements =[element for element in all_Possible_Pvars_names if 'rotor_diameter' in element]
peak_shaving=[element for element in all_Possible_Pvars_names if 'ps_percent' in element]

overridden_values = {}
overridden_values['wisdem.wt.wt_init.configuration.rotor_diameter_user']=262.0
overridden_values['wisdem.wt.wt_init.floating.joints.location_in']=\
np.array([[   0.        ,    0.        ,  -20.        ],
          [   0.        ,    0.        ,   15.        ],
          [  51.75*1.2      ,    3.14159265,  -20.        ],
          [  51.75*1.2      ,    3.14159265,   15.        ],
          [  51.75*1.2      ,    1.0471976 ,  -20.        ],
          [  51.75*1.2      ,    1.0471975 ,   15.        ],
          [  51.75*1.2      ,   -1.0471976 ,  -20.        ],
          [  51.75*1.2      ,   -1.0471976 ,   15.        ],
          [ 837.8       ,    3.14159265, -200.        ],
          [ 837.8       ,    1.0471976 , -200.        ],
          [ 837.8       ,   -1.0471976 , -200.        ]]) # also check this: 'wisdem.wt.wt_init.floating.alljoints.joints_xyz'

overridden_values['wisdem.wt.wt_init.floating.memgrp0.outer_diameter_in'] = np.array([10.*1.2, 10.*1.2]) #main column diametr
overridden_values['wisdem.wt.wt_init.floating.memgrp1.outer_diameter_in'] = np. array([12.5*1.2, 12.5*1.2]) #1st side collumn diameter
overridden_values['wisdem.wt.wt_init.floating.memgrp2.outer_diameter_in'] = np. array([12.5*1.2, 12.5*1.2]) #2nd side collumn diameter
overridden_values['wisdem.wt.wt_init.floating.memgrp3.outer_diameter_in'] = np. array([12.5*1.2, 12.5*1.2]) #3d side collumn diameter
overridden_values['tune_rosco_ivc.ps_percent'] = np.array([0.85*1.1])  # peak shaving 

overridden_values['control.max_pitch_rate'] = np.array([0.03490659*0.9])
overridden_values['tune_rosco_ivc.max_pitch'] = np.array([1.57*0.9])
overridden_values['raft.max_nac_accel'] = np.array([1.70673512*0.9])
overridden_values['configuration.hub_height_user'] = np.array([150.*1.1])
overridden_values['tower.diameter'] = \
       np.array([10.        , 10.        , 10.        , 10.        , 10.        ,
                 10.        ,  9.9050755 ,  9.9050755 ,  9.14777571,  9.14777571,
                 8.3311378 ,  8.3311378 ,  7.44055742,  7.44055742,  6.5       ,
                 6.5       ,  6.5       ,  6.5       ,  6.5       ,  6.5       ,
                 6.5       ])*0.9

tt = time.time()

wt_opt, modeling_options, opt_options = run_weis(fname_wt_input, fname_modeling_options, fname_analysis_options, overridden_values=overridden_values)

if MPI:
    rank = MPI.COMM_WORLD.Get_rank()
else:
    rank = 0
if rank == 0:
    print('Run time: %f'%(time.time()-tt))
    sys.stdout.flush()

model = wt_opt.model
all_Possible_Pvars=model.list_outputs()
all_Possible_Pvars_names=[all_Possible_Pvars[i][0] for i in np.arange(len(all_Possible_Pvars))]

# Save the list as a pickle file
all_Possible_Pvars_names_address=os.path.dirname(os.path.dirname( os.path.realpath(__file__) )) + os.sep + 'all_Possible_Pvars_names.pkl'
with open(all_Possible_Pvars_names_address, 'wb') as f:
    pickle.dump(all_Possible_Pvars_names, f)

"""model = wt_opt.model
all_Possible_Pvars=model.list_outputs()
all_Possible_Pvars_names=[all_Possible_Pvars[i][0] for i in np.arange(len(all_Possible_Pvars))]

rotor_elements =[element for element in all_Possible_Pvars_names if 'diameter' in element]
#wisdem.wt.wt_init.configuration.rotor_diameter_user


#print(all_Possible_Pvars[0])
#print(f"Tip deflection: {wt_opt[all_Possible_Pvars_names[508]][0]} meters")

# Construct a dict with values to overwrite
overridden_values = {}
#overridden_values["rotorse.wt_class.V_mean"] = 11.5
#overridden_values['wisdem.wt.wt_init.configuration.hub_height_user']=150.0*1.4
overridden_values['wisdem.wt.wt_init.tower.diameter']=wt_opt['wisdem.wt.wt_init.tower.diameter']*5.5 # LCOE will increase
"""