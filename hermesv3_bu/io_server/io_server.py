#!/usr/bin/env python

from mpi4py import MPI


class IoServer(object):
    """
    :param comm: Communicator object
    :type comm: MPI.Comm
    """
    def __init__(self, comm):
        self.__comm = comm
