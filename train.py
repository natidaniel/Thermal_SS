


import os
import yaml
import torch
import argparse

from tqdm import tqdm

from models import get_model
from losses import get_loss_function
from optimizers import get_optimizer
from metrics.metrics import runningScore
from utils import (save_train_loss_info, save_val_loss_info, save_best_score, 
setupDataloaders, setupRandomSeeds, logdircreateLog)
from statistics import mean



def print_loss_info(loss, logger, i, training_iterations):
    fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}"
    print_str = fmt_str.format(
        i + 1,
        training_iterations,
        loss.item(),
    )
    print(print_str)
    logger.info(print_str)

def evaluate(model, valloader,device, loss_fn):
    model.eval()
    with torch.no_grad():
        val_round_loss = []
        for i_val, (images_val, _) in tqdm(enumerate(valloader)):
            images_val = images_val.to(device)
            outputs = model(images_val)
            val_loss= loss_fn(*outputs, config = cfg)
            val_round_loss.append(val_loss.item())               
        mean_round_loss = mean(val_round_loss)
        return mean_round_loss



def train(cfg):

    # creating log directory and file
    logger, logdir = logdircreateLog(cfg)
    # Setup seeds
    random_seed = cfg["misc"]["random_seed"]
    setupRandomSeeds(random_seed)
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("using device: {}".format(device))
    # TODO: Setup Augmentations


    # Setup Dataloaders
    trainloader, valloader, num_train_images, num_val_images = setupDataloaders(cfg)

    # Setup Model
    model = get_model(cfg["model"]).to(device)

    # Setup optimizer, lr_scheduler and loss function
    optimizer_class = get_optimizer(cfg)
    optimizer_params = {k: v for k, v in cfg["training"]["optimizer"].items() if k != "name"}
    optimizer = optimizer_class(model.parameters(), **optimizer_params)
    logger.info("Using optimizer {}".format(optimizer))

    #scheduler = get_scheduler(optimizer, cfg["training"]["lr_schedule"])

    # setup loss function
    loss_fn = get_loss_function(cfg)
    logger.info("Using loss {}".format(loss_fn))

    # Setup Metrics
    n_classes = cfg["model"]["n_classes"]
    ignore_indices = cfg["training"]["loss"]["ignore_indices"]
    running_metrics_val = runningScore(n_classes, ignore_indices)

    begin_iter = 0
    # loading pretrained model if relevant
    if cfg["training"]["pretrained_model"] is not None:
        pretrained_model_path = cfg["training"]["pretrained_model"]
        if os.path.isfile(pretrained_model_path):
            logger.info("Loading model and optimizer from checkpoint '{}'".format(pretrained_model_path))
            checkpoint = torch.load(pretrained_model_path)
            begin_iter = checkpoint["epoch"]
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            start_iter = checkpoint["epoch"]
            logger.info("Loaded checkpoint '{}' (iter {})".format(pretrained_model_path, checkpoint["epoch"]))
            print("Loaded checkpoint '{}' (iter {})".format(pretrained_model_path, checkpoint["epoch"]))
        else:
            logger.info("No checkpoint found at '{}'".format(pretrained_model_path))
            print("No checkpoint found at: {}".format(pretrained_model_path))
            return

    best_miou = 0
    best_weighted_iou = 0
    i = begin_iter
    flag = True
    train_loss_info = []
    val_loss_info = []
    epochs = cfg["training"]["train_epochs"]
    num_train_images = cfg["data"]["num_train_images"]
    batch_size = cfg["training"]["batch_size"]
    training_iterations = epochs*num_train_images//batch_size
    
    
    train_print_interval = cfg["training"]["print_interval"]
    val_interval = cfg["training"]["val_interval"] 
    
    
    # train loop
    optimizer.zero_grad()
    while i <= training_iterations and flag:

        for images, labels, _ in trainloader:
            i += 1
            model.train()
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = loss_fn(input = outputs, target = labels)  
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


            # print loss info
            if (i + 1) % train_print_interval == 0:
                print_loss_info(loss, logger, i, training_iterations)
                curr_loss_info = [(i + 1), loss.item()]
                train_loss_info.append(curr_loss_info)
                save_train_loss_info(logdir, train_loss_info)
                
            # validate every 'val_interval' iterations
            if (i + 1) % val_interval == 0:
                model.eval()
                with torch.no_grad():
                    val_round_loss = []
                    for i_val, (images_val, labels_val, _) in tqdm(enumerate(valloader)):
                        images_val = images_val.to(device)
                        labels_val = labels_val.to(device)
                        outputs = model(images_val)
                        val_loss = loss_fn(input = outputs, target = labels_val)   
                        val_round_loss.append(val_loss.item())
                        pred = outputs.data.max(1)[1].cpu().numpy()
                        gt = labels_val.data.cpu().numpy()
                        running_metrics_val.update(gt, pred)

                    score, per_class_iou = running_metrics_val.get_scores()
                    mean_round_loss = mean(val_round_loss)
                    curr_loss_info = [(i+1), mean_round_loss, score.get("Mean IoU : \t"), score.get("FreqW Acc : \t")]
                    val_loss_info.append(curr_loss_info)
                    save_val_loss_info(logdir, val_loss_info)
                    running_metrics_val.reset()
                    curr_miou = score["Mean IoU : \t"]
                    curr_weighted_iou = score["FreqW Acc : \t"]

                    
                
                    # saving best miou model
                    if curr_miou > best_miou:
                        best_miou = curr_miou
                        temp_best_weighted_iou = best_weighted_iou
                        if curr_weighted_iou > temp_best_weighted_iou:
                            temp_best_weighted_iou = curr_weighted_iou
                        state = {
                            "epoch": i + 1,
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "best_iou": best_miou,
                            "best_weighted_iou": temp_best_weighted_iou, 
                            "per_class_iou": per_class_iou,
                        }
                        save_best_score(logdir, best_miou, best_weighted_iou, per_class_iou)
                        save_path = os.path.join(logdir, cfg["training"]["saved_iou_model"])
                        torch.save(state, save_path)
                    
                    # saving best weighted iou model
                    if curr_weighted_iou > best_weighted_iou:
                        best_weighted_iou = curr_weighted_iou
                        state = {
                            "epoch": i + 1,
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "best_iou": best_miou,
                            "best_weighted_iou": best_weighted_iou, 
                            "per_class_iou": per_class_iou,
                        }
                        save_best_score(logdir, best_miou, best_weighted_iou, per_class_iou)
                        save_path = os.path.join(logdir, cfg["training"]["saved_weighted_iou_model"])
                        torch.save(state, save_path)

            # making sure to stop after the wanted number of iterations
            if (i + 1) == training_iterations:
                
                flag = False
                break
    




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/template.yml",
        help="Configuration file to use for train+val",
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
    cfg['config_file'] = args.config


    train(cfg)






