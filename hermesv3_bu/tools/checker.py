#!/usr/bin/env python

import os
import sys
from mpi4py import MPI


def check_file(file_path_list, warning=False):
    if isinstance(file_path_list, str):
        file_path_list = [file_path_list]

    files_not_found = []
    for file_path in file_path_list:
        if not os.path.exists(file_path):
            files_not_found.append(file_path)

    if len(files_not_found) > 0:
        error_message = "*ERROR* (Rank {0}) File/s not found:".format(MPI.COMM_WORLD.Get_rank())
        for file_path in files_not_found:
            error_message += "\n\t{0}".format(file_path)
        print(error_message, file=sys.stderr)
        MPI.COMM_WORLD.Abort()
