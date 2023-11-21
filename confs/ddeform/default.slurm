nodes: 1
# set max wallclock time
time: 15:00:00
# set name of job
job_name: 'jade_sweep'
# set number of GPUs
gres: gpu:1

# Partition dedicated to jobs that utilise a single GPUs each.
partition: small
# Output file dir
output: ../../exp_bash/output/seq004_neus_%j.out