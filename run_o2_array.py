import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('-cmd', type=str)
parser.add_argument('-days', type=int, default=0)
parser.add_argument('-hours', type=int, default=1)
parser.add_argument('-mem', type=int, default=500)
parser.add_argument('-cores', type=int, default=1)
parser.add_argument('-partition', type=str, default='short', choices=['short', 'medium','long'])
parser.add_argument('-mail', action='store_true')
parser.add_argument('-expt', type=str)
parser.add_argument('-mock', action='store_true')

parser.add_argument('-paramfile', type=argparse.FileType('r'))

settings = parser.parse_args(); 

N = sum(1 for line in settings.paramfile)

fo = open("array_job.sbatch", "w")

jobarrayname = "d%s_%d" % (settings.expt, N)
jobname = "d%s_%%a_%d" % (settings.expt, N)
fo.write("#!/bin/bash\n\n")
fo.write("#SBATCH --job-name=%s\n" % jobarrayname)
fo.write("#SBATCH --output=/n/groups/datta/guitchounts/ephys_results/spike_sorting_results/%s/logs/%s.out\n" % (settings.expt, jobname))
fo.write("#SBATCH --error=/n/groups/datta/guitchounts/ephys_results/spike_sorting_results/%s/logs/%s.err\n" % (settings.expt,jobname))
fo.write("#SBATCH -t %d-%d:00\n"% (settings.days,settings.hours))
fo.write("#SBATCH -p %s\n" % settings.partition)
fo.write("#SBATCH -c %d\n" % settings.cores)
fo.write("#SBATCH -N 1\n")
fo.write("#SBATCH --mem=%d\n" % settings.mem)
if settings.mail:
    fo.write('#SBATCH --mail-type=FAIL\n')
    fo.write('#SBATCH --mail-user=guitchounts@fas.harvard.edu\n')
fo.write("\ncd %s\n" %  os.getcwd())
fo.write("conda activate spikeinterface \n")
fo.write("python %s -paramfile %s -line ${SLURM_ARRAY_TASK_ID}\n" % (settings.cmd,settings.paramfile.name))

fo.close()

if not settings.mock:
    os.system("sbatch --array=0-%d %s"% (N-1,fo.name))