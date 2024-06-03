# Encoder-Decoder Training
deepspeed --include localhost:0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 run.py --config run_configs/train/13b/sample_ko.json

# Decoder Training / Change the model_id in the config file!
# deepspeed --include localhost:0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 run_decoder.py --config run_configs/train/13b/sample_ko.json