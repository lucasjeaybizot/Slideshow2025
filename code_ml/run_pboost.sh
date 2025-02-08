module load singularity

for i in {1..15}; do
    subj=$(printf "SUBJ%02d" $i)
    singularity exec ubuPy2_05_.img python -c "import utils; utils.transform(indir='DATA', outdir='EXPERIMENTS', subj='$subj')"
done

singularity exec ubuPy2_05_.img python -c "import utils; utils.cfg_generator_taskbased(cfg_fname='taskbased.cfg', working_dir='EXPERIMENTS/taskbased/eeg', subjs=['SUBJ01', 'SUBJ02', 'SUBJ03', 'SUBJ04', 'SUBJ05', 'SUBJ06', 'SUBJ07', 'SUBJ08', 'SUBJ09', 'SUBJ10', 'SUBJ11', 'SUBJ12', 'SUBJ13', 'SUBJ14', 'SUBJ15'])"
singularity exec ubuPy2_05_.img python -c "import utils; utils.cfg_generator_timebased(cfg_fname='timebased.cfg', working_dir='EXPERIMENTS/timebased/eeg', subjs=['SUBJ01', 'SUBJ02', 'SUBJ03', 'SUBJ04', 'SUBJ05', 'SUBJ06', 'SUBJ07', 'SUBJ08', 'SUBJ09', 'SUBJ10', 'SUBJ11', 'SUBJ12', 'SUBJ13', 'SUBJ14', 'SUBJ15'])"

singularity exec ubuPy2_05_.img mpirun -np 16 python run.py -cp taskbased.cfg 1-2265
singularity exec ubuPy2_05_.img mpirun -np 16 python run.py -cp timebased.cfg 1-2265