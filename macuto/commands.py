# coding=utf-8
#-------------------------------------------------------------------------------

#Author: Alexandre Manhaes Savio
#Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
#Universidad del Pais Vasco UPV/EHU

#License: 3-Clause BSD

#2013, Alexandre Manhaes Savio
#Use this at your own risk!
#-------------------------------------------------------------------------------

import subprocess
import logging

log = logging.getLogger(__name__)


def condor_call(cmd, shell=True):
    """
    Tries to submit cmd to HTCondor, if it does not succeed, it will
    be called with subprocess.call.

    @param cmd: string
     Command to be submitted

    @return:
    """
    print(cmd)
    ret = condor_submit(cmd)
    if ret != 0:
        subprocess.call(cmd, shell=shell)


def condor_submit(cmd):
    """
    Submits cmd to HTCondor queue

    @param cmd: string
     Command to be submitted

    @return: int
    returncode value from calling the submission command.
    """
    try:
        is_running = subprocess.call('condor_status', shell=True) == 0
    except:
        return -1

    if is_running:
        sub_cmd = 'condor_qsub -shell n -b y -r y -N ' \
                  + cmd.split()[0] + ' -m n'

        log.info('Calling: ' + sub_cmd)

        return subprocess.call(sub_cmd + ' ' + cmd, shell=True)
    else:
        return -1



# if [ $scriptmode -ne 1 ] ; then
#     sge_command="$qsub_cmd -V -cwd -shell n -b y -r y $queueCmd $pe_options -M $mailto -N $JobName -m $MailOpts $LogOpts $sge_arch $sge_hol
# d"
# else
#     sge_command="$qsub_cmd $LogOpts $sge_arch $sge_hold"
# fi
# if [ $verbose -eq 1 ] ; then
#     echo sge_command: $sge_command >&2
#     echo executing: $@ >&2
# fi
# exec $sge_command $@ | awk '{print $3}'

