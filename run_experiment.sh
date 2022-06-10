# Create symbolic link to folder containing the *.nc ERA5 files
ln -s /home/zgoussea/scratch/nas_raw_datasets/ERA5 /home/zgoussea/projects/def-ka3scott/zgoussea/sifnet_public/datasets/raw

# Run experiment
python run_experiment.py \
    --region "Hudson" \
    --results_dir "results" \
    --month "Sep" \
    --year 1980 \
    --forecast_length 90 \
    --model_enum 1 \
    --raw_data_source "datasets/raw/" \
    --pre_computed_vars "datasets/pre_computed/"