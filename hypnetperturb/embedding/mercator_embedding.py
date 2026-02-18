import numpy as np
import tempfile
import os
import mercator

def run_mercator_temp(edgelist, fast_mode=True):

    with tempfile.TemporaryDirectory() as tmpdir:

        base_path = os.path.join(tmpdir, "edgelist")

        # 1Save temporary edgelist
        np.savetxt(base_path + ".txt", edgelist, fmt="%d")

        # Run mercator
        mercator.embed(base_path + ".txt",
                       output_name=base_path,
                       fast_mode=fast_mode)

        # 3Load outputs directly (no need to copy)
        coords_tr = np.loadtxt(base_path + ".inf_coord",
                               usecols=(2,3))

        log_data = open(base_path + ".inf_log").read()

        # 4Return what you need
        return coords_tr, log_data
    

def run_mercator(edgelist, folder_path, fast_mode=True): #totest

    # 1Save edgelist
    edgelist_path = os.path.join(folder_path, "edgelist.txt")
    np.savetxt(edgelist_path, edgelist, fmt="%d")

    # Run mercator
    mercator.embed(edgelist_path, output_name=edgelist_path, fast_mode=fast_mode)

    # 3Load outputs
    coords_tr = np.loadtxt(edgelist_path + ".inf_coord", usecols=(2,3))
    log_data = open(edgelist_path + ".inf_log").read()

    # 4Return what you need
    return coords_tr, log_data

    


