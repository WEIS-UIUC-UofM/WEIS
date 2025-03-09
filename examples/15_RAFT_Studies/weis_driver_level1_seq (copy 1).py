import os
import time
import sys

from weis.glue_code.runWEIS     import run_weis
from openmdao.utils.mpi  import MPI


run_dir                = os.path.dirname( os.path.realpath(__file__) ) + os.sep
fname_wt_input         = os.path.join(run_dir,"..","06_IEA-15-240-RWT", "IEA-15-240-RWT_VolturnUS-S.yaml")
fname_modeling_options = run_dir + "modeling_options_level1_doe.yaml"
fname_analysis_options1 = run_dir + "analysis_options_level1_seq1 (copy 1).yaml"
overridden_values = {}

tt = time.time()
wt_opt1, modeling_options1, opt_options1 = run_weis(fname_wt_input, fname_modeling_options, fname_analysis_options1, overridden_values)

if MPI:
    rank = MPI.COMM_WORLD.Get_rank()
else:
    rank = 0
if rank == 0:
    print('Run time: %f'%(time.time()-tt))
    sys.stdout.flush()

print('rank = {:}, exiting'.format(rank))

dv = 'floating.jointdv_2'

x_star1 = wt_opt1.get_val(dv)

objective_st1 = wt_opt1.get_val('raft.platform_mass')
# exit_flag1 = wt_opt1.driver._scipy_optimize_result['status']
print('raft.heave_avg, ', wt_opt1.get_val('raft.heave_avg'))
print('raft.Std_PtfmPitch, ', wt_opt1.get_val('raft.Std_PtfmPitch'))
print('raft.heave_period, ', wt_opt1.get_val('raft.heave_period'))
print('raft.pitch_period, ', wt_opt1.get_val('raft.pitch_period'))
print(wt_opt1.driver.result)
# %%
import numpy as np

overridden_values = {}
overridden_values[dv] = np.array(x_star1)
fname_analysis_options2 = run_dir + "analysis_options_level1_seq2 (copy 1).yaml"
wt_opt2, modeling_options2, opt_options2 = run_weis(fname_wt_input, fname_modeling_options, fname_analysis_options2, overridden_values)
x_star2 = wt_opt2.get_val(dv)

objective_st2 = wt_opt2.get_val('raft.platform_mass')
exit_flag2 = wt_opt2.driver._scipy_optimize_result['status']

