# coding=utf-8
#-------------------------------------------------------------------------------

#Author: Alexandre Manhaes Savio <alexsavio@gmail.com>
#Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
#Universidad del Pais Vasco UPV/EHU
#
#2013, Alexandre Manhaes Savio
#Use this at your own risk!
#-------------------------------------------------------------------------------

import numpy as np


def takespread(sequence, num):
    """
    Generator for sequence. Will return num equally spaced
    items from the sequence.

    @param sequence: container

    @param num: int
    Number of items to be retrieved from sequence.

    @return: container item
    """
    length = float(len(sequence))
    for i in range(num):
        yield sequence[int(np.ceil(i * length / num))]


def makespread(sequence, num):
    """
    Return a sequence with num equally spaced items
    from the sequence.

    @param sequence: container

    @param num: int
    Number of items to be retrieved from sequence

    @return: container
    Subset of the given sequence
    """
    length = float(len(sequence))
    seq = np.array(sequence)
    return seq[np.ceil(np.arange(num) * length / num).astype(int)]