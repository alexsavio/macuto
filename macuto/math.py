# coding=utf-8
#-------------------------------------------------------------------------------
#License GNU/GPL v3
#Author: Alexandre Manhaes Savio <alexsavio@gmail.com>
#Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
#Universidad del Pais Vasco UPV/EHU
#
#2013, Alexandre Manhaes Savio
#Use this at your own risk!
#-------------------------------------------------------------------------------

import numpy as np


def takespread (sequence, num):
    """
    @param sequence:
    @param num:
    @return:
    """
    length = float(len(sequence))
    for i in range(num):
        yield sequence[int(np.ceil(i * length / num))]


def makespread (sequence, num):
    """
    @param sequence:
    @param num:
    @return:
    """
    length = float(len(sequence))
    seq = np.array(sequence)
    return seq[np.ceil(np.arange(num) * length / num).astype(int)]