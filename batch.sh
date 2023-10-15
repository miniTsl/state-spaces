#!/bin/bash

horizons=(1 5 10 20 30 50 70 100 120 140 150 160 170 180 190)

for horizon in ${horizons[@]}
do
    output_path=train_output_input200/forecasting/$horizon.txt
    python -m train pipeline=server model=s4 trainer.max_epochs=500 wandb=null task.loss=mae dataset.prediction_window=200 dataset.forecast_horizon=$horizon > $output_path
done

# for horizon in ${horizons[@]}
# do
#     output_path=train_output/regression/$horizon.txt
#     python -m train pipeline=server pipeline.task=regression model=s4 wandb=null task.loss=mae task.metrics=mae dataset.forecast_horizon=$horizon > $output_path
# done