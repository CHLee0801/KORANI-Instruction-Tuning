##### Encoder-Decoder Model Inferece
deepspeed --include localhost:0 run.py --config run_configs/eval/basic_ko.json # Not specifiying cluster of dataset will evaluate every datasets.

# deepspeed --include localhost:0 run.py --config run_configs/eval/basic_ko.json --target_cluster natural_language_inference,sentiment
# deepspeed --include localhost:0 run.py --config run_configs/eval/basic_ko.json --target_dataset kobest_wic,curse_detection
# deepspeed --include localhost:0 run.py --config run_configs/eval/basic_ko.json --target_cluster natural_language_inference,sentiment --target_dataset kobest_wic,curse_detection

##### Decoder Model Inferece / Change the model_id in the config file!
# deepspeed --include localhost:0 run_decoder.py --config run_configs/eval/basic_ko.json --target_dataset klue_nli