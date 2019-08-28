#!/usr/bin/env python

import os
import sys
from mpi4py import MPI
from warnings import warn


def check_files(file_path_list, warning=False):
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

        if warning:
            warn(error_message.replace('ERROR', 'WARNING'))
            return False
        else:
            error_exit(error_message)
    return True


def error_exit(error_message):
    if not error_message[:7] == "*ERROR*":
        error_message = "*ERROR* (Rank {0}) ".format(MPI.COMM_WORLD.Get_rank()) + error_message
    print(error_message, file=sys.stderr)
    MPI.COMM_WORLD.Abort()
