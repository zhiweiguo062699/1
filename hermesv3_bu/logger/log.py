#!/usr/bin/env python

import os
import numpy as np
import pandas as pd


class Log(object):
    def __init__(self, comm, arguments, log_refresh=1, time_log_refresh=0):
        """
        Initialise the Log class.

        :param comm: MPI communicator

        :param arguments: Complete argument NameSpace.
        :type arguments: NameSpace

        :param log_refresh:
        :param time_log_refresh:
        """
        self.comm = comm

        self.refresh_rate = (log_refresh, time_log_refresh)
        self.log_refresh = self.refresh_rate[0]
        self.time_log_refresh = self.refresh_rate[1]

        self.log_level = arguments.log_level
        self.log_path = os.path.join(arguments.output_dir, 'logs', 'Log_r{0:04d}_p{1:04d}_{2}.log'.format(
            comm.Get_rank(), comm.Get_size(), os.path.basename(arguments.output_name).replace('.nc', '')))
        self.time_log_path = os.path.join(arguments.output_dir, 'logs', 'Times_p{0:04d}_{1}.csv'.format(
            comm.Get_size(), os.path.basename(arguments.output_name).replace('.nc', '')))

        if comm.Get_rank() == 0:
            if not os.path.exists(os.path.dirname(self.log_path)):
                os.makedirs(os.path.dirname(self.log_path))
            else:
                if os.path.exists(self.time_log_path):
                    os.remove(self.time_log_path)
            self.time_log = open(self.time_log_path, mode='w')
        else:
            # Time log only writed by master process
            self.time_log = None
        comm.Barrier()

        if os.path.exists(self.log_path):
            os.remove(self.log_path)

        self.log = open(self.log_path, mode='w')

        self.df_times = pd.DataFrame(columns=['Class', 'Function', comm.Get_rank()])

    def write_log(self, message, message_level=1):
        """
        Write the log message.

        The log will be refresh every log_refresh value messages.

        :param message: Message to write.
        :type message: str

        :param message_level: Importance of the message. From 1 (bottom) to 3 (top). Default 1
        :type message_level: int

        :return: True if everything is ok.
        :rtype: bool
        """
        if message_level <= self.log_level:
            self.log.write("{0}\n".format(message))

            if self.log_refresh > 0:
                self.log_refresh -= 1
            if self.log_refresh == 0:
                self.log.flush()
                self.log_refresh = self.refresh_rate[0]
        return True

    def _write_csv_times_log_file(self, rank=0):
        """
        Write the times log CSV file.

        :param rank: Process to write.
        :type rank: int

        :return: True if everything is ok.
        :rtype: bool
        """
        print self.df_times
        self.df_times = self.df_times.groupby(['Class', 'Function']).sum().reset_index()
        data_frames = self.comm.gather(self.df_times, root=0)
        if self.comm.Get_rank() == rank:
            df_merged = reduce(lambda left, right: pd.merge(left, right, on=['Class', 'Function'], how='outer'),
                               data_frames)
            df_merged = df_merged.groupby(['Class', 'Function']).sum()
            df_merged['min'] = df_merged.loc[:, range(self.comm.Get_size())].min(axis=1)
            df_merged['max'] = df_merged.loc[:, range(self.comm.Get_size())].max(axis=1)
            df_merged['mean'] = df_merged.loc[:, range(self.comm.Get_size())].mean(axis=1)

            df_merged = df_merged.replace(0.0, np.NaN)
            df_merged.to_csv(self.time_log_path)

        self.comm.Barrier()
        return True

    def write_time_log(self, class_name, function_name, time, message_level=1):
        """
        Add times to be written. Master process will write that log every times_log_refresh received messages.

        :param class_name: Name of the class.
        :type class_name: str

        :param function_name: Name of the function.
        :type function_name: str

        :param time: Time spent in the function.
        :type time: float

        :param message_level: Importance of the message. From 1 (bottom) to 3 (top). Default 1
        :type message_level: int

        :return: True if everything is ok.
        :rtype: bool
        """
        if message_level <= self.log_level:
            self.df_times = self.df_times.append(
                {'Class': class_name, 'Function': function_name, self.comm.Get_rank(): time}, ignore_index=True)
            # if self.time_log_refresh > 0:
            #     self.time_log_refresh -= 1
            # if self.time_log_refresh == 0:
            #
            #     self._write_csv_times_log_file()
            #     self.time_log_refresh = self.refresh_rate[0]
        return True

    def finish_logs(self):
        """
        Finalize the log files.

        :return:
        """
        self._write_csv_times_log_file()
        self.log.flush()
        self.log.close()
