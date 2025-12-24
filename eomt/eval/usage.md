Call extract logits first which takes the path to the dataset and the model name, default eomt L 1024, trianed on cityspaces.

usage example:
eval % python extract_logits.py \
 --input "datapath/RoadAnomaly21/images/\*.jpg" \
 --model_id tue-mps/cityscapes_semantic_eomt_large_1024 \
 --save_dir ./saved_logits

The logits will be saved under eval/saved_logits/RoadAnomaly

Call evaluate_logits to calculate benchmarks.

python evaluate_logits.py \
 --logits_dir ./saved_logits/RoadAnomaly21

Sample output (just the summary part):

Final Summary:
msp: AUPRC 14.80 | FPR95 95.00
max_logit: AUPRC 18.53 | FPR95 93.11
entropy: AUPRC 14.80 | FPR95 94.99
