#!/bin/bash

horizons=(140 150 160 170)

for horizon in ${horizons[@]}
do
    output_path=train_input300/$horizon.txt
    python -m train pipeline=server model=s4 trainer.max_epochs=1000 trainer.devices=2 wandb=null task.loss=mae dataset.prediction_window=300 dataset.forecast_horizon=$horizon > $output_path
done