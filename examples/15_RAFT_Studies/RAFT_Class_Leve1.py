import os
import time
import sys

from weis.glue_code.runWEIS     import run_weis
from wisdem.commonse.mpi_tools  import MPI
import pickle
import numpy as np

class RAFT_Class_Level1:
    def __init__(self):
        pass

    def run_Level1(self,fname_wt_input, fname_modeling_options, fname_analysis_options, overridden_values):
        wt_opt, modeling_options, opt_options = run_weis(fname_wt_input, fname_modeling_options, fname_analysis_options, overridden_values)
        return wt_opt, modeling_options, opt_options 
    
    def 


if __name__ == "__main__":
