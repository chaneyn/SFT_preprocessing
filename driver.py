import sys
import getopt
import os
#import numpy as np
#import json
#import pickle
import preprocessing
#import gc
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def main(argv):
   try:
      opts, args = getopt.getopt(argv,"hf:s:",["jfile=","pstage="])
   except getopt.GetoptError:
      print(argv)
      print('SFTs_preprocessor -f <jfile> -s <pstage>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('SFTs_preprocessor -f <jfile> -s <pstage>')
         sys.exit()
      elif opt in ("-f", "--jfile"):
         jfile = arg
      elif opt in ("-s", "--pstage"):
         pstage = arg

   if pstage == 'define_domain':
        print("(r%d) SFTs: Assemble and screen domain" % rank,flush=True)
        preprocessing.assemble_domain(comm,jfile)
   elif pstage == 'preprocess_datasets':
        print("(r%d) SFTs: Preprocessing datasets" % rank,flush=True)
        preprocessing.preprocess_datasets(comm,jfile)
   elif pstage == 'compute_sfts':
        print("(r%d) SFTs: Compute SFTs" % rank,flush=True)
        preprocessing.compute_sfts(comm,jfile)
   elif pstage == 'finalize_netcdf':
        if rank == 0:
         print("(r%d) SFTs: Finalize NetCDF file" % rank,flush=True)
         preprocessing.prepare_netcdf_file(comm,jfile)
        comm.Barrier()
   else:
        print("The pstage provided is not an option")

if __name__ == "__main__":
   main(sys.argv[1:])