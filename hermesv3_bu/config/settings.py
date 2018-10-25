#!/usr/bin/env python

# Copyright 2018 Earth Sciences Department, BSC-CNS
#
# This file is part of HERMESv3_GR.
#
# HERMESv3_GR is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HERMESv3_GR is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HERMESv3_GR. If not, see <http://www.gnu.org/licenses/>.


import os
import numpy as np

global refresh_log

global precision
precision = np.float64

global writing_serial
writing_serial = False

global compressed_netcdf
compressed_netcdf = True

if not writing_serial:
    compressed_netcdf = False

global icomm
global comm
global rank
global size

global log_level
global log_file
global df_times


def define_global_vars(in_log_level):
    # TODO Documentation
    from mpi4py import MPI

    global icomm
    global comm
    global rank
    global size

    icomm = MPI.COMM_WORLD
    comm = icomm.Split(color=0, key=0)
    rank = comm.Get_rank()
    size = comm.Get_size()

    global log_level
    log_level = in_log_level


def define_log_file(log_path, date):
    # TODO Documentation
    log_path = os.path.join(log_path, 'logs')
    if not os.path.exists(log_path):
        if rank == 0:
            os.makedirs(log_path)
        comm.Barrier()
    log_path = os.path.join(log_path, 'HERMESv3_{0}_Rank{1}_Procs{2}.log'.format(
        date.strftime('%Y%m%d%H'), str(rank).zfill(4), str(size).zfill(4)))
    if os.path.exists(log_path):
        os.remove(log_path)

    global log_file

    log_file = open(log_path, mode='w')


def define_times_file():
    # TODO Documentation
    import pandas as pd
    global df_times

    df_times = pd.DataFrame(columns=['Class', 'Function', rank])


def write_log(msg, level=1):
    # TODO Documentation
    if log_level >= level:
        log_file.write(msg + '\n')
        log_file.flush()


def write_time(module, func, time, level=1):
    # TODO Documentation
    global df_times
    if log_level >= level:
        df_times = df_times.append({'Class': module, 'Function': func, rank: time}, ignore_index=True)


def finish_logs(output_dir, date):
    # TODO Documentation
    import pandas as pd
    from functools import reduce
    log_file.close()

    global df_times
    df_times = df_times.groupby(['Class', 'Function']).sum().reset_index()
    data_frames = comm.gather(df_times, root=0)
    if rank == 0:
        times_path = os.path.join(output_dir, 'logs', 'HERMESv3_{0}_times_Procs{1}.csv'.format(
            date.strftime('%Y%m%d%H'), str(size).zfill(4)))
        if os.path.exists(times_path):
            os.remove(times_path)
        df_merged = reduce(lambda left, right: pd.merge(left, right, on=['Class', 'Function'], how='outer'),
                           data_frames)
        df_merged['min'] = df_merged.loc[:, range(size)].min(axis=1)
        df_merged['max'] = df_merged.loc[:, range(size)].max(axis=1)
        df_merged['mean'] = df_merged.loc[:, range(size)].mean(axis=1)

        df_merged.to_csv(times_path)
    comm.Barrier()
