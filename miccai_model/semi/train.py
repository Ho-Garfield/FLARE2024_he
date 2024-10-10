import logging
from tqdm import tqdm
import os 
import sys
import shutil
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss,BCELoss
import torch.backends.cudnn as cudnn
import SimpleITK as sitk
import copy
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
from config import *
from data_transforms import  SampleRandomFlip, SampleRandomRotateZ, SampleResizeyx, \
    Sample_Normalize, SampleRandomCrop, SampleToTensor, Sample_Random_Cutout,Sample_Add_Noise,\
    Sample_Brightness_Multiply,Sample_Gussian_Blur,\
    SampleRandomScale,Sample_Adjust_contrast,SampleLowRes,SampleGama, CurrentLabelBatch
from dataset import TwoStreamBatchSampler
from dataset import MedicalImageDataset as mydataset
import ramps
import losses
import random
from net import Net
import monai.transforms as mt
import monai.data as ma

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

def save_checkpoint(model, ema_model, optimizer, scaler,iter_num, epoch_num, best_tar_mean_dice,snapshot_path,best_model):
    checkpoint_path = os.path.join(snapshot_path, 'model_checkpoint.pth.tar')
    torch.save({
        'iter_num': iter_num,
        'epoch_num': epoch_num,
        "best_tar_mean_dice":best_tar_mean_dice,
        'model_state_dict': model.state_dict(),
        'ema_model_state_dict': ema_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict':scaler.state_dict() if scaler is not None else None
    }, checkpoint_path)
    logging.info("Saved checkpoint to '{}'".format(checkpoint_path))
    save_model_path = os.path.join(snapshot_path, f'iter_{iter_num}.pth')
    if best_model is not None:
        torch.save(best_model.state_dict(), save_model_path)
        logging.info("Saved model to '{}'".format(save_model_path))

def show_image(volume_batch, label_batch, main_ouput_soft, writer, iter_num, tar_class, show_class_list=None):
    B,C,Z,X,Y = main_ouput_soft.size()
    _, out_labels = torch.max(main_ouput_soft, dim=1, keepdim=True)                            
    zs = np.unique(np.where(label_batch[0, :, :, :].cpu().numpy() == tar_class)[0])
    max_z = label_batch[0, :, :, :].cpu().numpy().shape[0]-1
    zs = np.sort(zs)  
    num_slices = 5
    if len(zs) > 5:
        slices = np.linspace(0, len(zs) - 1, num_slices, dtype=int)
        zs = zs[slices]
    else: 
        zs = np.linspace(0, max_z, num_slices, dtype=int)
    
    # image
    image = volume_batch[0, 0:1, zs, :, :].permute(
        1, 0, 2, 3).repeat(1, 3, 1, 1)
    grid_image = make_grid(image, 5, normalize=True)
    writer.add_image('train/Image', grid_image, iter_num)
    if tar_class is not None:
        # sofmax ouput
        image = main_ouput_soft[0, tar_class:tar_class+1, zs, :, :].permute(
            1, 0, 2, 3).repeat(1, 3, 1, 1)
        grid_image = make_grid(image, 5, normalize=False)
        writer.add_image('train/Predicted_label_softmax',
                            grid_image,iter_num)
        #Groundtruth
        image = (label_batch[0, zs, :, :] == tar_class).unsqueeze(
            0).permute(1, 0, 2, 3).repeat(1, 3, 1, 1)
        grid_image = make_grid(image, 5, normalize=False)
        writer.add_image('train/Groundtruth_label',
                            grid_image, iter_num)

        # predict target label 
        image = (out_labels[0, :, zs, :, :] == tar_class).permute(
            1, 0, 2, 3).repeat(1, 3, 1, 1)
        grid_image = make_grid(image, 5, normalize=False)
        writer.add_image(f'train/Predicted_label',
                            grid_image, iter_num)
    
    if show_class_list is not None:
        for c in show_class_list:
            if c != tar_class:
                # predict label GLAND
                image = (out_labels[0, :, zs, :, :] == c).permute(
                    1, 0, 2, 3).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image(f'train/Predicted_label_{c}',
                                    grid_image, iter_num)

def validation(epoch_num, validloader, deep_supervision, enable_half, model, dice_loss, best_tar_mean_dice,tar_class,best_model_path,writer):
    bm = None
    with autocast(enabled=enable_half):
        with torch.no_grad():
            dices = 0
            num = 0
            logging.info(f"validation num:{len(validloader)}")
            for i_batch, sampled_batch in enumerate(validloader):
                volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

                outputs = model(volume_batch)  
                if deep_supervision:
                    main_ouput,*low_res_outputs = outputs
                else:
                    main_ouput = outputs
                main_ouput_soft = torch.softmax(main_ouput, dim=1)
                #(n_class -1 ) do not caculate background
                dices = dices + dice_loss.dice(main_ouput_soft, label_batch.unsqueeze(1))#,is_dice=True)
                num = num + 1
            mean_dices = dices/num
            # do not caculate background
            mean_dice = mean_dices[1:].mean()

        if tar_class is None:
            tar_dice = mean_dice
        else:
            tar_dice = mean_dices[tar_class]

        if(best_tar_mean_dice <tar_dice):
            best_tar_mean_dice = tar_dice
            torch.save(model.state_dict(), best_model_path)
            bm = model
        writer.add_scalar('info/tar_dice', tar_dice, epoch_num)    
        writer.add_scalar('info/mean_dice', mean_dice, epoch_num)
        return best_tar_mean_dice, bm

def do_split(img_folder,label_folder,cur_fold= 1, kfold=5, name = "split.json", img_suffix="_0000.nii.gz",label_suffix=".nii.gz",seed = 42):    
    import json

    parent_dir = os.path.dirname(img_folder)
    json_path = os.path.join(parent_dir,name)
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
    else:
        img_list = os.listdir(img_folder)
        label_list = [str.replace(f,label_suffix,img_suffix) for f in os.listdir(label_folder) 
                      if str.replace(f,label_suffix,img_suffix) in img_list 
                      and f.endswith(label_suffix)]
        split_list = ma.partition_dataset(data=label_list,num_partitions=kfold,seed=seed)
        data ={}
        for i in range(1,kfold+1):            
            index = i - 1
            train_set = []
            for k in range(0,kfold):
                if k != index:
                    train_set = train_set + split_list[k]
            validation_set = split_list[index]
            data[str(i)] = {'train':train_set, 
            'validation':validation_set}
        with open(json_path, 'w') as f:
            json.dump(data,f,indent=4)

    return data[str(cur_fold)]

def supervised_train(args, snapshot_path):
    base_lr = args.base_lr
    max_iterations = args.max_iterations
    num_classes = args.num_classes
    show_image_per_iterations = args.show_image_per_iterations
    save_model_per_iterations = args.save_model_per_iterations
    deep_supervision = True
    enable_half = True
    tar_class = None
    show_class_list = [1,2,3,4,5,6,7,8,9,10,11,12,13]
    kfold = args.Kfold
    cur_fold = args.cur_fold
    def create_model(ema=False):
        # Network definition
        net = Net(n_channels=1, n_classes=num_classes,normalization="instancenorm",has_dropout=True,deep_supervision=deep_supervision)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)

    cb1 = CurrentLabelBatch(complete_label_batch_size=args.labeled_bs)
    cb2 = CurrentLabelBatch(complete_label_batch_size=args.labeled_bs)
    data = do_split(os.path.join(args.root_path,"images"),os.path.join(args.root_path,"labels"), cur_fold=cur_fold,kfold=kfold,name ="split.json")
    # data['train'] = [f for f in os.listdir() if f.endswith('nii.gz')]
    # data['validation'] = [f for f in os.listdir() if f.endswith('nii.gz')]

    
    
    db_train = mydataset(root_dir=args.root_path,
                       select_label_files = data['train'],
                       transform=transforms.Compose([                            
                        Sample_Normalize(method="min_max"),
                        SampleRandomRotateZ(30),
                        SampleRandomScale(),
                        SampleRandomCrop(args.patch_size,current_label_batch=cb1),                             
                        Sample_Add_Noise(),
                        Sample_Gussian_Blur(),
                        Sample_Brightness_Multiply(),                            
                        Sample_Adjust_contrast(),
                        SampleRandomFlip(),                              
                        SampleLowRes(),
                        SampleGama(invert_image=True,percentage=0.1), 
                        SampleGama(percentage=0.3),                             
                        Sample_Random_Cutout(percentage=0.3),                         
                        SampleToTensor(),
                        ]),is_semi=False)
    db_valid = mydataset(root_dir=args.root_path,
                        select_label_files = data['validation'],
                        transform=transforms.Compose([                            
                        Sample_Normalize(method="min_max"),
                        SampleRandomCrop(args.patch_size,current_label_batch=cb2),                             
                        SampleToTensor()
                        ]),is_semi=False)
    
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
        np.random.seed(args.seed + worker_id)
        torch.manual_seed(args.seed + worker_id)
        
    trainloader = DataLoader(db_train, batch_size = args.labeled_bs,shuffle=True,
                             num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    validloader = DataLoader(db_valid, batch_size = args.labeled_bs,shuffle=True,
                             num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                            momentum=0.99, weight_decay=3e-5,nesterov=True)

    checkpoint_file = os.path.join(snapshot_path, 'model_checkpoint.pth.tar')
    if os.path.isfile(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        iter_num = checkpoint['iter_num']
        epoch_num = checkpoint['epoch_num']
        best_tar_mean_dice = checkpoint['best_tar_mean_dice']
        logging.info("Loaded checkpoint from '{}'".format(checkpoint_file))
        logging.info("continue epoch '{}'".format(epoch_num))

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if enable_half:
            scaler = GradScaler()
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        else:
            scaler = None

    else:
        iter_num = 0
        epoch_num = 0
        best_tar_mean_dice = 0
        logging.info("No checkpoint found at '{}'".format(checkpoint_file))
        if enable_half:
            scaler = GradScaler()
        else:
            scaler = None
    model.train()
    ema_model.train()

    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(n_classes=num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    max_epoch = max_iterations // len(trainloader) + 1
    logging.info("total epoch: {}".format(max_epoch))

    iterator = tqdm(range(epoch_num,max_epoch), ncols=70,initial=epoch_num,total=max_epoch)

    best_model_path = os.path.join(snapshot_path, f'best_model.pth') 
    weights = None    
    best_model = None     
    for epoch_num in iterator:
        if iter_num < max_iterations/2:
            n = 5
        elif iter_num< max_iterations*2.0/3.0:
            n = 3
        else:
            n = 2
        # validation 
        if epoch_num%n==1:
            best_tar_mean_dice,bm = validation(epoch_num, validloader, deep_supervision, enable_half, 
                                               model, dice_loss, best_tar_mean_dice, tar_class, 
                                               best_model_path, writer)
            if best_model is None or bm is not None:
                best_model = bm
                

        for i_batch, sampled_batch in enumerate(trainloader):      
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            with autocast(enabled=enable_half):
                outputs = model(volume_batch[:args.labeled_bs])  
                if deep_supervision:
                    main_ouput,*low_res_outputs = outputs
                    low_res_ouputs_soft = [torch.softmax(output, dim=1) for output in low_res_outputs]
                else:
                    main_ouput = outputs
                main_ouput_soft = torch.softmax(main_ouput, dim=1)

                if deep_supervision:
                    if weights is None:
                        # this gives higher resolution outputs more weight in the loss
                        weights = [1 / (2 ** i) for i in range(len(low_res_outputs) + 1)]
                        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
                        weights = list(weights / np.sum(weights))
                    B,C_1,Z,H,W = label_batch[:args.labeled_bs].unsqueeze(1).size()
                    loss_ce = [ce_loss(low_res_output[:args.labeled_bs], label_batch[:args.labeled_bs])*w 
                                    for low_res_output,w in zip([main_ouput,*low_res_outputs],weights)]
                    loss_dice = [dice_loss(low_res_ouput_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))*w 
                                        for low_res_ouput_soft,w in zip([main_ouput_soft,*low_res_ouputs_soft],weights)]
                    total_loss_ce = sum(loss_ce)
                    total_loss_dice = sum(loss_dice)
                else:
                    loss_ce_main = ce_loss(main_ouput[:args.labeled_bs],label_batch[:args.labeled_bs][:])
                    loss_dice_main = dice_loss(main_ouput_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
                    total_loss_ce = loss_ce_main 
                    total_loss_dice = loss_dice_main       
                supervised_loss = (total_loss_ce + total_loss_dice)#0.5 * 
                loss = supervised_loss                
                optimizer.zero_grad()

            if enable_half:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
                optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            iter_num = iter_num + 1
            # chart
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', total_loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', total_loss_dice, iter_num)
            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
                (iter_num, loss.item(), total_loss_ce.item(), total_loss_dice.item()))
            writer.add_scalar('loss/loss', loss, iter_num)
            
            if iter_num % show_image_per_iterations == 0:            
                show_image(volume_batch, label_batch, main_ouput_soft, writer, iter_num, tar_class, show_class_list)

            if iter_num % save_model_per_iterations == 0:
                save_checkpoint(model, ema_model, optimizer, scaler, iter_num, epoch_num , best_tar_mean_dice, snapshot_path,best_model)

            if iter_num >= max_iterations:
                break

        if iter_num >= max_iterations:
            iterator.close()
            break 
           
    writer.close()
    return "Training Finished!"

def semi_supervised_train(args, snapshot_path):
    base_lr = args.base_lr
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    num_classes = args.num_classes
    show_image_per_iterations = args.show_image_per_iterations
    save_model_per_iterations = args.save_model_per_iterations
    deep_supervision = True
    enable_half = True
    tar_class = None
    show_class_list = [1,2,3,4,7,8,9,11,13]
    kfold = args.Kfold
    cur_fold = args.cur_fold
    unlabel_diff = True
    def create_model(ema=False):
        # Network definition
        if ema:
            net = Net(n_channels=1, n_classes=num_classes,normalization="instancenorm",has_dropout=True,deep_supervision=False)
        else:
            net = Net(n_channels=1, n_classes=num_classes,normalization="instancenorm",has_dropout=True,deep_supervision=deep_supervision)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)

    cb_train_label = CurrentLabelBatch(complete_label_batch_size=args.labeled_bs)
    cb_train_unlabel = CurrentLabelBatch(complete_label_batch_size=args.batch_size - args.labeled_bs)
    cb_valid = CurrentLabelBatch(complete_label_batch_size=args.labeled_bs)

    data = do_split(os.path.join(args.root_path,"images"),os.path.join(args.root_path,"labels"), cur_fold=cur_fold,kfold=kfold,name ="split.json")
    # data={}
    # data['train'] = [f.replace(".nii.gz","_0000.nii.gz") for f in os.listdir(os.path.join(args.root_path,"labels")) if f.endswith('nii.gz')]
    # data['validation'] = [f.replace(".nii.gz","_0000.nii.gz") for f in os.listdir(os.path.join(args.root_path,"labels")) if f.endswith('nii.gz')]
    
    
    unlabel_train_trans = None        
    label_train_trans = transforms.Compose([                            
                            Sample_Normalize(),
                            SampleRandomRotateZ(30),
                            SampleRandomScale(),
                            SampleRandomCrop(args.patch_size,current_label_batch=cb_train_label,
                                             current_unlabel_batch=None if unlabel_diff else cb_train_unlabel,
                                             incomplete_class_num=4),                             
                            Sample_Add_Noise(),
                            Sample_Gussian_Blur(),
                            Sample_Brightness_Multiply(),                            
                            Sample_Adjust_contrast(),
                            SampleRandomFlip(),                              
                            SampleLowRes(),
                            SampleGama(invert_image=True,percentage=0.1), 
                            SampleGama(percentage=0.3),                             
                            Sample_Random_Cutout(),                            
                            SampleToTensor(),
                        ])
    if unlabel_diff:


        unlabel_train_trans = transforms.Compose([    
                            Sample_Normalize(), 
                            SampleRandomRotateZ(30),
                            SampleRandomScale(),
                            SampleRandomCrop(args.patch_size,current_label_batch=cb_train_unlabel,current_unlabel_batch=None,\
                                             foreground_labels=[1]),                             
                            # Sample_Add_Noise(),
                            Sample_Gussian_Blur(),
                            Sample_Brightness_Multiply(),                            
                            Sample_Adjust_contrast(),
                            SampleRandomFlip(),                              
                            SampleLowRes(),
                            SampleGama(invert_image=True,percentage=0.1), 
                            SampleGama(percentage=0.3),                             
                            Sample_Random_Cutout(),                            
                            SampleToTensor(),
                            ])


        
    db_train = mydataset(root_dir=args.root_path,
                    select_label_files = data['train'],
                    transform=label_train_trans,is_semi=True,use_half_label=False,unlabel_transform=unlabel_train_trans) 
    label_train_valid = transforms.Compose([                            
                        Sample_Normalize(),
                        SampleRandomCrop(args.patch_size,current_label_batch=cb_valid),                             
                        SampleToTensor()])

    db_valid = mydataset(root_dir=args.root_path,
                        select_label_files = data['validation'],
                        transform=label_train_valid,is_semi=False)
    


    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
        np.random.seed(args.seed + worker_id)
        torch.manual_seed(args.seed + worker_id)
        

    labeled_idxs = list(range(0, db_train.label_num))
    unlabeled_idxs = list(range(db_train.label_num, len(db_train)))
    batch_sampler = TwoStreamBatchSampler(
    labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train,  batch_sampler=batch_sampler,
                             num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    validloader = DataLoader(db_valid, batch_size = args.labeled_bs, shuffle=False,
                             num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                            momentum=0.99, weight_decay=3e-5,nesterov=True)

    checkpoint_file = os.path.join(snapshot_path, 'model_checkpoint.pth.tar')
    if os.path.isfile(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        iter_num = checkpoint['iter_num']
        epoch_num = checkpoint['epoch_num']
        best_tar_mean_dice = checkpoint['best_tar_mean_dice']
        logging.info("Loaded checkpoint from '{}'".format(checkpoint_file))
        logging.info("continue epoch '{}'".format(epoch_num))
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if enable_half:
            scaler = GradScaler()
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        else:
            scaler = None
    else:
        iter_num = 0
        epoch_num = 0
        best_tar_mean_dice = 0
        logging.info("No checkpoint found at '{}'".format(checkpoint_file))
        if enable_half:
            scaler = GradScaler()
        else:
            scaler = None
    model.train()
    ema_model.train()

    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(n_classes=num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    max_epoch = max_iterations // len(trainloader) + 1
    logging.info("total epoch: {}".format(max_epoch))

    iterator = tqdm(range(epoch_num,max_epoch), ncols=70,initial=epoch_num,total=max_epoch)

    best_model_path = os.path.join(snapshot_path, f'best_model.pth') 
    weights = None    
    best_model = None     
    for epoch_num in iterator:
        if iter_num < max_iterations/2:
            n = 5
        elif iter_num< max_iterations*2.0/3.0:
            n = 3
        else:
            n = 2
        # validation 
        if epoch_num%n==1:
            best_tar_mean_dice,bm = validation(epoch_num, validloader, deep_supervision, enable_half, 
                                               model, dice_loss, best_tar_mean_dice, tar_class, 
                                               best_model_path, writer)
            if best_model is None or bm is not None:
                best_model = bm
            
        for i_batch, sampled_batch in enumerate(trainloader):
            with autocast(enabled=enable_half):           

                volume_batch, label_batch = sampled_batch['image'], sampled_batch['label'] 
                volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
                unlabeled_volume_batch = volume_batch[args.labeled_bs:]     

                noise = torch.clamp(torch.randn_like(
                                volume_batch[args.labeled_bs:]) * 0.1, -0.2, 0.2)
                ema_inputs = unlabeled_volume_batch + noise#unlabeled_volume_batch 
                outputs = model(volume_batch)  

                if deep_supervision:
                    main_ouput,*low_res_outputs = outputs
                    low_res_ouputs_soft = [torch.softmax(output, dim=1) for output in low_res_outputs]
                else:
                    main_ouput = outputs
                main_ouput_soft = torch.softmax(main_ouput, dim=1)
                B, C, Z, H, W = main_ouput_soft.size()
                with torch.no_grad():
                    ema_output = ema_model(ema_inputs)
                    ema_output_soft = torch.softmax(ema_output, dim=1)                
                if deep_supervision:
                    if weights is None:
                        # this gives higher resolution outputs more weight in the loss
                        weights = [1 / (2 ** i) for i in range(len(low_res_outputs) + 1)]
                        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
                        weights = list(weights / np.sum(weights))
                    B,C_1,Z,H,W = label_batch[:args.labeled_bs].unsqueeze(1).size()

                    loss_ce = [ce_loss(low_res_output[:args.labeled_bs], label_batch[:args.labeled_bs])*w 
                                    for low_res_output,w in zip([main_ouput,*low_res_outputs],weights)]
                    loss_dice = [dice_loss(low_res_ouput_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))*w 
                                        for low_res_ouput_soft,w in zip([main_ouput_soft,*low_res_ouputs_soft],weights)]


                    total_loss_ce = sum(loss_ce)
                    total_loss_dice = sum(loss_dice)
                else:
                    loss_ce_main = ce_loss(main_ouput[:args.labeled_bs],label_batch[:args.labeled_bs][:])
                    loss_dice_main = dice_loss(main_ouput_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
                    total_loss_ce = loss_ce_main 
                    total_loss_dice = loss_dice_main       
                
                supervised_loss = 0.5*(total_loss_dice + total_loss_ce)
                consistency_weight = get_current_consistency_weight(
                    iter_num//int(args.max_iterations/args.consistency_rampup))
                consistency_loss = torch.mean(
                    (main_ouput_soft[args.labeled_bs:] - ema_output_soft)**2)
                loss = supervised_loss + consistency_weight * consistency_loss
                loss = supervised_loss
                optimizer.zero_grad()
            if enable_half:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
                optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('info/consistency_loss', consistency_loss, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', total_loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', total_loss_dice, iter_num)
            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
                (iter_num, loss.item(), total_loss_ce.item(), total_loss_dice.item()))
            writer.add_scalar('loss/loss', loss, iter_num)
            
                      
            if iter_num % show_image_per_iterations == 0:            
                show_image(volume_batch, label_batch, main_ouput_soft, writer, iter_num, tar_class, show_class_list)

            if iter_num % save_model_per_iterations == 0:
                save_checkpoint(model, ema_model, optimizer, scaler, iter_num, epoch_num, best_tar_mean_dice, snapshot_path,best_model)

            if iter_num >= max_iterations:
                break

        if iter_num >= max_iterations:
            iterator.close()
            break 


            
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    args = parser.parse_args()
    torch.cuda.empty_cache()

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "./model/{}_f{}/{}".format(
        args.exp, args.cur_fold ,args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/semi'):
        shutil.rmtree(snapshot_path + '/semi')
    shutil.copytree('./semi', snapshot_path + '/semi',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    semi_supervised_train(args, snapshot_path)
