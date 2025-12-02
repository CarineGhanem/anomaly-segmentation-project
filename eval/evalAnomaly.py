# Copyright (c) OpenMMLab. All rights reserved.
import os
import glob
import torch
import random
from PIL import Image
import numpy as np
from erfnet import ERFNet
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 20
# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

input_transform = Compose(
    [
        Resize((512, 1024), Image.BILINEAR),
        ToTensor(),
        # Normalize([.485, .456, .406], [.229, .224, .225]),
    ]
)

target_transform = Compose(
    [
        Resize((512, 1024), Image.NEAREST),
    ]
)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )  
    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--anomaly_score', default='max_logit')  #other option is entropy
    args = parser.parse_args()
    anomaly_score_list = []
    ood_gts_list = []

    # Extract dataset and method names for result identification
    input_path = str(args.input[0])
    dataset_name = "Unknown"
    if "RoadAnomaly21" in input_path:
        dataset_name = "RoadAnomaly21"
    elif "RoadObsticle21" in input_path:
        dataset_name = "RoadObsticle21"
    elif "FS_LostFound_full" in input_path or "LostFound" in input_path:
        dataset_name = "FS_LostFound_full"
    elif "fs_static" in input_path:
        dataset_name = "fs_static"
    elif "RoadAnomaly" in input_path and "RoadAnomaly21" not in input_path:
        dataset_name = "RoadAnomaly"
    method_name = args.anomaly_score

    # Remove old result for same dataset+method if exists
    results_file = 'results.txt'
    if os.path.exists(results_file):
        with open(results_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        with open(results_file, 'w', encoding='utf-8') as f:
            for line in lines:
                exact_dataset_match = f'Dataset: {dataset_name} |' in line
                method_match = f'Method: {method_name}' in line
                # Keep line if it doesn't match current dataset+method exactly
                if not (exact_dataset_match and method_match):
                    f.write(line)

    file = open(results_file, 'a')

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    model = ERFNet(NUM_CLASSES)

    if (not args.cpu):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print("Using device:", device)
        model = model.to(device)


    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    print ("Model and weights LOADED successfully")
    model.eval()
    
    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        print(path)
        images = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float().to(device)
        #images = images.permute(0,3,1,2)
        with torch.no_grad():
            result = model(images)
        if args.anomaly_score == 'entropy':
            # compute softmax probabilities
            probabilities = torch.nn.functional.softmax(result, dim=1)
            # compute entropy
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-6), dim=1)
            anomaly_result = entropy.squeeze(0).data.cpu().numpy()
        
        elif args.anomaly_score == 'msp':
            probabilities = torch.nn.functional.softmax(result, dim=1)
            anomaly_result = 1.0 - np.max(probabilities.squeeze(0).data.cpu().numpy(), axis=0)      
        else:
            # compute max logit score
            max_logit, _ = torch.max(result, dim=1)
            anomaly_result = -max_logit.squeeze(0).data.cpu().numpy()
            
        pathGT = path.replace("images", "labels_masks")                
        if "RoadObsticle21" in pathGT:
           pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
           pathGT = pathGT.replace("jpg", "png")                
        if "RoadAnomaly" in pathGT:
           pathGT = pathGT.replace("jpg", "png")  

        mask = Image.open(pathGT)
        mask = target_transform(mask)
        ood_gts = np.array(mask)

        if "RoadAnomaly" in pathGT:
            ood_gts = np.where((ood_gts==2), 1, ood_gts)
        if "LostAndFound" in pathGT:
            ood_gts = np.where((ood_gts==0), 255, ood_gts)
            ood_gts = np.where((ood_gts==1), 0, ood_gts)
            ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)

        if "Streethazard" in pathGT:
            ood_gts = np.where((ood_gts==14), 255, ood_gts)
            ood_gts = np.where((ood_gts<20), 0, ood_gts)
            ood_gts = np.where((ood_gts==255), 1, ood_gts)

        if 1 not in np.unique(ood_gts):
            continue              
        else:
             ood_gts_list.append(ood_gts)
             anomaly_score_list.append(anomaly_result)
        del result, anomaly_result, ood_gts, mask
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    file.write( "\n")

    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)

    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))
    
    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))

    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)

    print(f'AUPRC score: {prc_auc*100.0}')
    print(f'FPR@TPR95: {fpr*100.0}')

    # Write result with dataset and method identifier
    result_line = f'Dataset: {dataset_name} | Method: {method_name} | AUPRC: {prc_auc*100.0:.2f} | FPR@TPR95: {fpr*100.0:.2f}\n'
    file.write(result_line)
    file.close()
    print(f"Result saved: {result_line.strip()}")

if __name__ == '__main__':
    main()
