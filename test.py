


import os
import yaml
import torch
import argparse
import glob
from torch.utils import data

from tqdm import tqdm
from yaml.parser import Parser

from models import get_model
from dataloaders import get_loader
from optimizers import get_optimizer
from metrics.metrics import runningScore
from utils import ( CLASS_NAMES, VALID_CLASSES)
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms




def test(model, testloader,device, running_metrics, out_filename, folder, output_folder):
    
    labels_directory = output_folder
    if not os.path.exists(labels_directory):
        os.makedirs(labels_directory)
    # test loop
    model.eval()
    with torch.no_grad():
        for i, (images, labels, img_name) in tqdm(enumerate(testloader)):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            pred = outputs.data.max(1)[1].cpu().numpy()
            gt = labels.data.cpu().numpy()
            running_metrics.update(gt, pred)
            output_label = transforms.ToPILImage()(pred[0].astype('uint8'))
            output_label.save(os.path.join(labels_directory,img_name[0]+'.png'))
            
        # extracting results
        score, per_class_iou = running_metrics.get_scores()
        miou = score.get("Mean IoU : \t")
        weighted_iou = score.get("FreqW Acc : \t")
        class_precision = score.get("Class precision : \t")
        class_recall = score.get("Class recall : \t")
        # writing results to file
        with open(out_filename,'w') as f:
            # miou+weighted iou
            f.write("best weighted_iou: %.3f\n" % weighted_iou)
            f.write("best miou: %.3f\n" % miou)
            f.write("per class iou: \n")
            for k, v in per_class_iou.items():
                f.write("class %2d: %.3f\n" % (k,v))
            # precision per class
            f.write("per class precision: \n")
            for i, (row) in enumerate(class_precision):
                f.write("class %2d: %.3f\n" % (VALID_CLASSES[i],row[i]/row.sum()))
            
            # recall per class
            f.write("per class recall: \n")
            for i, (row) in enumerate(class_recall):
                f.write("class %2d: %.3f\n" % (VALID_CLASSES[i],row[i]/row.sum()))

            
        


def test_models(cfg, train_folder):

    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Dataloader
    data_loader = get_loader(cfg["data"]["dataset"])
    data_path = cfg["data"]["path"]
    image_path = os.path.join(data_path, cfg["data"]["image_split"])
    label_path = os.path.join(data_path, cfg["data"]["label_split"])
    n_channels = cfg["data"]["n_channels"]

    test_data = data_loader(
        image_path = image_path,
        label_path = label_path,
        img_list=cfg["data"]["test_img_list"],
        img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
        n_channels=n_channels,
    )


    testloader = data.DataLoader(
        test_data,
        batch_size=1,
        num_workers=cfg["training"]["n_workers"],
        shuffle=False,
    )


    # Setup Models
    model = get_model(cfg["model"]).to(device)

    # Setup optimizer, lr_scheduler and loss function
    optimizer_class = get_optimizer(cfg)
    optimizer_params = {k: v for k, v in cfg["training"]["optimizer"].items() if k != "name"}
    optimizer = optimizer_class(model.parameters(), **optimizer_params)

    # Setup Metrics
    n_classes = cfg["model"]["n_classes"]
    ignore_indices = cfg["training"]["loss"]["ignore_indices"]
    running_metrics_test = runningScore(n_classes, ignore_indices)

    # loading pretrained model
    pretraind_miou_path = os.path.join(train_folder, 'segmentation_best_iou_model.pkl')
    miou_output = os.path.join(train_folder, 'test_miou_res.txt')
    miou_label_path = os.path.join(train_folder, 'output_labels_miou')
    pretraind_weighted_iou_path = os.path.join(train_folder, 'segmentation_best_weighted_iou_model.pkl')
    weighted_iou_output = os.path.join(train_folder, 'test_weighted_iou_res.txt')
    weighted_iou_label_path = os.path.join(train_folder, 'output_labels_weighted_iou')

    for model_path, out_path, output_folder in [(pretraind_miou_path, miou_output, miou_label_path), (pretraind_weighted_iou_path, weighted_iou_output, weighted_iou_label_path)]:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        test(model, testloader, device, running_metrics_test, out_path, train_folder, output_folder)
        running_metrics_test.reset()
    




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--train_folder",
        nargs="?",
        type=str,
        default="",
        help="the folder containing output of training, e.g: /home/segmentation/2000_10_10_18_00_00_model_name",
    )
    

    args = parser.parse_args()

    # loading training configs
    cfg_paths = glob.glob(os.path.join(args.train_folder, '*.yml'))
    assert(len(cfg_paths) == 1)
    with open(cfg_paths[0]) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
    





    test_models(cfg, args.train_folder)






