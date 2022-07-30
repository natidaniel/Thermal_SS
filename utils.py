#from _typeshed import OpenTextModeReading
import logging
import datetime
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from dataloaders import get_loader
from torch.utils import data
from tensorboardX import SummaryWriter
import shutil


VALID_CLASSES = [0,1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17]

CLASS_NAMES = {
    0: 'Terrain',
    1: 'Unpaved route',
    2: 'Paved road',
    3: 'Tree trunk',
    4: 'Tree foliage',
    5: 'Rocks',
    6: 'Large shrubs',
    7: 'Low vegetation',
    8: 'Wire fence',
    9: 'Sky',
    10: 'Person',
    11: 'Vehicle',
    12: 'Building',
    13: 'Tree branches',
    14: 'Misc',
    15: 'Water',
    16: 'Animal',
    17: 'Ignore',
    18: 'Ignore',
}





def logdircreateLog(cfg):
    timestamp = str(datetime.datetime.now()).split(".")[0]
    timestamp = timestamp.replace(" ", "_").replace(":", "_").replace("-", "_")
    logdir = cfg["training"]["run_dir"]
    logdir = os.path.join(logdir, timestamp + "_" + os.path.basename(cfg['config_file']).split(".")[0])
    writer = SummaryWriter(log_dir=logdir)
    print("RUNDIR: {}".format(logdir))
    # copying config file to log directory
    shutil.copy(cfg['config_file'], logdir)
    logger = get_logger(logdir)
    logger.info("Let the games begin")
    return (logger, logdir)







def setupRandomSeeds(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)




def setupDataloaders(cfg):
    data_loader = get_loader(cfg["data"]["dataset"])
    data_path = cfg["data"]["path"]
    image_path = os.path.join(data_path, cfg["data"]["image_split"])
    label_path = os.path.join(data_path, cfg["data"]["label_split"])
    n_channels = cfg["data"]["n_channels"]

    t_loader = data_loader(
        image_path = image_path,
        label_path = label_path,
        img_list=cfg["data"]["train_img_list"],
        img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
        n_channels=n_channels,
    )
    num_train_images = len(t_loader)

    v_loader = data_loader(
        image_path = image_path,
        label_path = label_path,
        img_list=cfg["data"]["val_img_list"],
        img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
        n_channels=n_channels,
    )
    num_val_images = len(v_loader)


    trainloader = data.DataLoader(
        t_loader,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["n_workers"],
        shuffle=True,
    )

    valloader = data.DataLoader(
        v_loader,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["n_workers"],
    )
    return (trainloader, valloader, num_train_images, num_val_images)






def get_logger(logdir):
    logger = logging.getLogger("ptsemseg")
    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    file_path = os.path.join(logdir, "run_{}.log".format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger





def save_train_loss_info(res_dir, loss_info):
    acc_txt_name = os.path.join(res_dir,r'train_acc.txt')
    acc_fig_name = os.path.join(res_dir, r'train_acc.png')
    acc_txt = np.array(loss_info)

    np.savetxt(acc_txt_name, acc_txt, fmt='%.3f', delimiter='\t')

    iters = acc_txt[:, 0]
    train_loss = acc_txt[:,1]
    plt.plot(iters, train_loss, 'r', label = 'loss')
    plt.title('train loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(acc_fig_name)
    plt.close()


def save_val_loss_info(res_dir, loss_info):
    acc_txt_name = os.path.join(res_dir,r'val_acc.txt')
    acc_fig_name = os.path.join(res_dir, r'val_acc.png')
    acc_txt = np.array(loss_info)
    iters = acc_txt[:,0]
    val_loss = acc_txt[:,1]
    miou = acc_txt[:,2]
    fwiou = acc_txt[:,3]
    np.savetxt(acc_txt_name, acc_txt, fmt='%.3f', delimiter='\t')

    plt.plot(iters, val_loss, 'b', label='Validation loss')
    plt.plot(iters, miou, 'g', label='Validation iou')
    plt.plot(iters, fwiou, 'r', label='Validation weighted_iou')
    

    plt.title('validation loss')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(acc_fig_name)
    plt.close()


def save_best_score(res_dir, best_iou, best_weighted_iou, class_iou):
    file_name = os.path.join(res_dir, 'best_score.txt')
    with open(file_name,'w') as f:
        f.write("best weighted_iou: %.3f\n" % best_weighted_iou)
        f.write("best iou: %.3f\n" % best_iou)
        for k, v in class_iou.items():
            f.write("class %2d: %.3f\n" % (k,v))





    