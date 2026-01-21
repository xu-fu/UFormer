import os
import time
import datetime
import random
import torch
import numpy as np
from mydataset3d import My3DDataLoadertrainval, My3DDataLoadertest
from resnet3d import restranscross503
from tensorboardX import SummaryWriter
from loguru import logger
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from monai.utils import first, set_determinism
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    Lambdad,
    RandSpatialCropd,
    SpatialPadd,
    RandFlipd,
    Resized,
    RandRotated,
    RandGaussianNoised,
    RandGaussianSharpend,
    RandGaussianSmoothd,
    RandGibbsNoised,
    RandAdjustContrastd,
    ScaleIntensityRanged,
    HistogramNormalized,
    RandAffine,
    Rand3DElasticd,


    
)

import matplotlib.pyplot as plt
from utils import Calculateindicators, FocalLoss, CosLoss, ContrastiveLoss, AngularPenaltySMLoss, CenterLoss, \
                  CosineAnnealingWarmRestarts, WarmupStepLR, genotsuask

def SegmentationPresetTrain():
    train_transforms = Compose(
            [
                LoadImaged(keys=["T2", "DWI", "ADC"]),
                EnsureChannelFirstd(keys=["T2", "DWI", "ADC"]),
                RandFlipd(keys=["T2", "DWI", "ADC"], spatial_axis=0, prob=0.2),
                RandRotated(keys=["T2", "DWI", "ADC"], mode=['bilinear', 'bilinear', 'bilinear'], range_z=0.1, prob=0.2),
                Resized(keys=["T2", "DWI", "ADC"],mode=['bilinear', 'bilinear', 'bilinear'], spatial_size=[128, 128, 16]),
                
                ScaleIntensityd(keys=["T2", "DWI"], minv=0., maxv=1.),
                ScaleIntensityRanged(keys=["ADC"], a_min=0, a_max=2000, b_min=0., b_max=1., clip=True),

                
            ]
        )
    return train_transforms

def SegmentationPresetEval():
    val_transforms = Compose(
            [
                LoadImaged(keys=["T2", "DWI", "ADC"]),
                EnsureChannelFirstd(keys=["T2", "DWI", "ADC"]),
                
                Resized(keys=["T2", "DWI", "ADC"],mode=['bilinear', 'bilinear', 'bilinear'], spatial_size=[128, 128, 16]),
                
                ScaleIntensityd(keys=["T2", "DWI"], minv=0., maxv=1.),
                ScaleIntensityRanged(keys=["ADC"], a_min=0, a_max=2000, b_min=0., b_max=1., clip=True),
                
            ]
        )
    return val_transforms



def get_transform(train):

    if train:
        return SegmentationPresetTrain()
    else:
        return SegmentationPresetEval()


def create_model():

    model = restranscross503(pretrain=False, num_classes=2)
    
    return model

def trainoneepoch(model, lossdict, optimizer, train_loader, device, epoch, num_classes, lr_scheduler):
    '''
    单次训练
    '''
    model.train()

    if num_classes == 2:
        # 设置cross_entropy中背景和前景的loss权重(根据自己的数据集进行设置)
        # loss_weight = torch.as_tensor([1.0, 2.0], device=device)
        # loss_weight = torch.as_tensor([40.0], device=device)
        loss_weight = None
    else:
        loss_weight = None
    
    # bce_logits_loss = nn.BCEWithLogitsLoss(pos_weight=loss_weight)
    ce_loss = lossdict['ce_loss']
    cos_loss = lossdict['cos_loss']
    con_loss = lossdict['con_loss']
    asoft_loss = lossdict['asoft_loss']
    center_loss = lossdict['center_loss']

   
    e_loss = 0
    flg = 0
    loss_cei = 0
    scorelist = []
    ylist = []
    idlist = []
    for train_data in tqdm(train_loader):
        imaget2, imagedwi, imageadc, label, id= (train_data["T2"],
                                            train_data["DWI"],
                                            train_data["ADC"],
                                            train_data["label"],
                                            train_data['id'])
        # mask = train_data['mask']
        idlist.append(id)

        # imaget2 = torch.Tensor(imaget2)
        # imagedwi = torch.Tensor(imagedwi)
        # imageadc = torch.Tensor(imageadc)
        # mask = genotsuask(imagedwi)
        mask = 1


        label = torch.Tensor(label).long().to(device)
        imaget2 = torch.permute(imaget2*mask, (0, 1, 4, 2, 3)).to(device)
        imagedwi = torch.permute(imagedwi*mask, (0, 1, 4, 2, 3)).to(device)
        imageadc = torch.permute(imageadc*mask, (0, 1, 4, 2, 3)).to(device)
        # labelcos = torch.abs(label[:len(label)//2]-label[len(label)//2:])
        flg = flg + 1


        t2dwiadc, features, outputo = model(imaget2, imagedwi, imageadc)
        output = outputo
        # featurescos1 = features[:len(label)//2]
        # featurescos2 = features[len(label)//2:]

        scorelist.append(torch.softmax(output, dim=1).detach().cpu().numpy())
        ylist.append(label.detach().cpu().numpy())

        # target = torch.squeeze(target, dim=1)###.long()
        # loss_ce = nn.functional.cross_entropy(torch.softmax(output, dim=1), target, weight=loss_weight)
        loss_ce = ce_loss(output, label)
        # loss_cos = con_loss(labelcos, featurescos1, featurescos2)
        # loss_asoft = asoft_loss(features, label)
        # loss_center = center_loss(features, label) /512.
        loss = loss_ce #+ 0.1*loss_center #+ loss_cos 

        # if epoch<0:
        #     loss = loss_cos
        # else:
        #     loss = loss_ce + loss_cos
        # loss = loss_dice


        optimizer.zero_grad()
        loss.backward()

        # for param in center_loss.parameters():
        #     # lr_cent is learning rate for center loss, e.g. lr_cent = 0.5
        #     param.grad.data *= 1#(lr_cent / (alpha * lr))

        optimizer.step()

        # lr_scheduler.step() #按照step更新

        # # 记录梯度
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         writer.add_histogram(f'{name}.grad', param.grad, flg+epoch*len(train_loader))

        lr = optimizer.param_groups[0]["lr"]
        
    
        e_loss = e_loss + loss
        loss_cei = loss_cei + loss_ce


    e_loss = e_loss/len(train_loader)
    loss_cei = loss_cei/len(train_loader)
    yscore = np.concatenate(scorelist, axis=0)
    ytrue = np.concatenate(ylist, axis=0)
    idarray = np.concatenate(idlist, axis=0)
    
    return e_loss, loss_cei, yscore, ytrue, idarray, lr


def evaluateoneepoch(model, lossdict, data_loader, device, epoch, num_classes):
    '''
    模型评估
    '''
    model.eval()
    # confmat = utils.ConfusionMatrix(num_classes)
    if num_classes == 2:
        # 设置cross_entropy中背景和前景的loss权重(根据自己的数据集进行设置)
        # loss_weight = torch.as_tensor([1.0, 2.0], device=device)
        # loss_weight = torch.as_tensor([40.0], device=device)
        loss_weight = None
    else:
        loss_weight = None

    # bce_logits_loss = nn.BCEWithLogitsLoss(pos_weight=loss_weight)
    ce_loss = lossdict['ce_loss']
    cos_loss = lossdict['cos_loss']
    con_loss = lossdict['con_loss']
    asoft_loss = lossdict['asoft_loss']
    center_loss = lossdict['center_loss']
    
    e_loss = 0
    flg = 0
    loss_cei = 0
    scorelist = []
    ylist = []
    idlist = []
    with torch.no_grad():
        for val_data in tqdm(data_loader):
            imaget2, imagedwi, imageadc, label, id= (val_data["T2"],
                                                val_data["DWI"],
                                                val_data["ADC"],
                                                val_data["label"],
                                                val_data["id"])
            # mask = val_data['mask']
            idlist.append(id)
            # imaget2 = torch.Tensor(imaget2)
            # imagedwi = torch.Tensor(imagedwi)
            # imageadc = torch.Tensor(imageadc)
            # mask = genotsuask(imagedwi)
            mask = 1

            label = torch.Tensor(label).long().to(device)
            imaget2 = torch.permute(imaget2*mask, (0, 1, 4, 2, 3)).to(device)
            imagedwi = torch.permute(imagedwi*mask, (0, 1, 4, 2, 3)).to(device)
            imageadc = torch.permute(imageadc*mask, (0, 1, 4, 2, 3)).to(device)
            # labelcos = torch.abs(label[:len(label)//2]-label[len(label)//2:])

            flg = flg + 1
    
            # target = target.float()
            # image = image.repeat(1,3,1,1) #重复3次，变成3通道图
            # image, target = image.to(device), target.to(device)
            # target = torch.unsqueeze(target, dim=1).float()

            t2dwiadc, features, outputo = model(imaget2, imagedwi, imageadc)
            output = outputo

            # featurescos1 = features[:len(label)//2]
            # featurescos2 = features[len(label)//2:]

            scorelist.append(torch.softmax(output, dim=1).detach().cpu().numpy())
            ylist.append(label.detach().cpu().numpy())

            # loss_ce = nn.functional.cross_entropy(torch.softmax(output, dim=1), target, weight=loss_weight)
            loss_ce = ce_loss(output, label)
            # loss_ce = focalloss(output, target, sigmoid=True)
            # loss_cos = con_loss(labelcos, featurescos1, featurescos2)
            # loss_asoft = asoft_loss(features, label)
            # loss_center = center_loss(features, label) /512.
            loss = loss_ce #+ 0.1*loss_center #+ loss_cos


            # if epoch<20:
            #     loss = loss_cos
            # else:
            #     loss = loss_ce + loss_cos

            # loss = loss_dice
            e_loss = e_loss + loss
            loss_cei = loss_cei + loss_ce

        e_loss = e_loss/len(data_loader)
        loss_cei = loss_cei/len(data_loader)
        yscore = np.concatenate(scorelist, axis=0)
        ytrue = np.concatenate(ylist, axis=0)
        idarray = np.concatenate(idlist, axis=0)


    return e_loss, loss_cei, yscore, ytrue, idarray


def main(args):
    # 在 torch 中的代码中避免使用共享内存
    torch.multiprocessing.set_sharing_strategy('file_system')

    # #######设置随机种子固定########
    random.seed(args.myseed)
    np.random.seed(args.myseed)
    torch.manual_seed(args.myseed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.myseed)
        torch.cuda.manual_seed_all(args.myseed)  
    # 创建生成器并手动设定种子
    g = torch.Generator()
    g.manual_seed(args.myseed)
    # set_determinism(seed=args.myseed)


    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    # segmentation nun_classes + background
    num_classes = args.num_classes #+ 1


    # 用来保存训练以及验证过程中信息
    results_log = "/logs/log{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(results_log) 
    
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    train_loader, val_loader = My3DDataLoadertrainval(patientdirlist=args.patientdir_train, clifilelist=args.data_path_train, traintransform=get_transform(train=True), valtransform=get_transform(train=False), batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = My3DDataLoadertest(patientdirlist=args.patientdir_test, clifilelist=args.data_path_test, transform=get_transform(train=False), batch_size=4, num_workers=num_workers, shuffle=False)

    model = create_model()
    # load weights
    weights_path = args.model_path
    if weights_path is not None:
        print('==========load pretrained model==========')
        model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'], strict=False)
    model.to(device)
    print(f'----------using device: {device} ----------')

    # ce_loss = nn.CrossEntropyLoss(weight=torch.as_tensor([1.0, 1.], device=device))
    ce_loss = FocalLoss()
    cos_loss = CosLoss(dim=1)
    con_loss = ContrastiveLoss()
    asoft_loss = AngularPenaltySMLoss(in_features=2048, out_features=2, loss_type='cosface', device=device)
    center_loss = CenterLoss(num_classes=2, feat_dim=1024, device=device)

    params_to_optimize = [p for p in model.parameters() if p.requires_grad] 
    optimizer = torch.optim.AdamW(params_to_optimize, lr=args.lr, weight_decay=args.weight_decay) 

    lossdict = {
            'ce_loss'    : ce_loss,
            'cos_loss'   : cos_loss,
            'con_loss'   : con_loss,
            'asoft_loss' : asoft_loss,
            'center_loss': center_loss
            }

    lr_scheduler = WarmupStepLR(optimizer=optimizer, step_size=15, gamma=1, warmup_steps=0)


    best_loss = 9999999999.
    bestout_loss = 9999999999.
    bestout_auc = 0.
    start_time = time.time()
    for epoch in range(args.epochs):
        print(f'-------------Epoch:[{epoch}]---------------')
        train_loss, train_lossce, train_score, train_true, trainidlist, lr  = trainoneepoch(model, lossdict, optimizer, train_loader, device, epoch, num_classes,lr_scheduler=lr_scheduler)
    
        lr_scheduler.step() 
        valloss, vallossce, val_score, val_true, validlist = evaluateoneepoch(model, lossdict, val_loader, device=device, epoch=epoch, num_classes=num_classes)

        traincal_eval_ind, trainauc, trainacc = Calculateindicators(train_score, train_true, trainidlist, model='train')
        valcal_eval_ind, valauc, valacc = Calculateindicators(val_score, val_true, validlist, model='val')
        # print(traincal_eval_ind)

        logger.info(f"epoch:{epoch}   lr:{lr} \n  train loss:{train_loss.item():.4f}  train ind: {traincal_eval_ind} \n  val loss:{valloss.item():.4f}  val ind: {valcal_eval_ind}")
        
        # 测试集测试模型结果
        if epoch%1==0:
            testloss, testlossce, test_score, test_true, testidlist = evaluateoneepoch(model, lossdict, test_loader, device=device, epoch=epoch, num_classes=num_classes)
            testcal_eval_ind, testauc, testacc = Calculateindicators(test_score, test_true, testidlist, model='test')
            logger.info(f"test epoch:{epoch}   test loss: {testloss.item():.4f} test ind: {testcal_eval_ind}")
            writer.add_scalars('test', {'auc':testauc, 'acc':testacc}, epoch)

            # 保存外部测试集最优权重
            if args.save_best is True and epoch>0:
                
                if testauc > bestout_auc:
                        bestout_auc = testauc
                        print(f'--------------best auc :{bestout_auc}---------------')
                        save_fileloss = {"model": model.state_dict(),
                                        "optimizer": optimizer.state_dict(),
                                        "lr_scheduler": lr_scheduler.state_dict(),
                                        "epoch": epoch,
                                        "args": args}
                        
                        logger.info(f"****best auc epoch:{epoch}   test loss: {testloss.item():.4f}   test ind: {testcal_eval_ind}")
                    

        ##########记录tensorboard##############
        writer.add_scalars('loss/all', {'train':train_loss.detach().cpu().numpy(), 'val':valloss.detach().cpu().numpy()}, epoch)
        writer.add_scalars('loss/ce', {'train':train_lossce.detach().cpu().numpy(), 'val':vallossce.detach().cpu().numpy()}, epoch)
        writer.add_scalars('info/auc', {'train':trainauc, 'val':valauc}, epoch)
        writer.add_scalars('info/acc', {'train':trainacc, 'val':valacc}, epoch)
        

        if args.save_best is True and epoch>=0:

            if best_loss > valloss:
                best_loss = valloss
                print(f'best loss:{best_loss}')
                save_fileloss = {"model": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "lr_scheduler": lr_scheduler.state_dict(),
                                "epoch": epoch,
                                "args": args}
                torch.save(save_fileloss, "/best_model.pth")

                testloss, testlossce, test_score, test_true, testidlist = evaluateoneepoch(model, lossdict, test_loader, device=device, epoch=epoch, num_classes=num_classes)
                testcal_eval_ind, testauc, testacc = Calculateindicators(test_score, test_true, testidlist, model='test')
                logger.info(f"****best loss epoch:{epoch}   test loss: {testloss.item():.4f}   test ind: {testcal_eval_ind}")
            
            

    testloss, testlossce, test_score, test_true, testidlist = evaluateoneepoch(model, lossdict, test_loader, device=device, epoch=epoch, num_classes=num_classes)
    testcal_eval_ind, testauc, testacc = Calculateindicators(test_score, test_true, testidlist, model='test')
    logger.info(f"****end loss epoch:{epoch}   test ind: {testcal_eval_ind}")


    writer.close()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def parse_args(myseed):
    import argparse
    parser = argparse.ArgumentParser(description="pytorch training")

    parser.add_argument("--patientdir-train", default=[
                                                       "/root"  ], help="DRIVE root")
    
    parser.add_argument("--patientdir-test", default=[
                                                    "/root"  ], help="DRIVE root")
    
    parser.add_argument("--data-path-train", default=[
                                                      "/cspca-data.xlsx"
                                                    ], help="DRIVE root")
    
    parser.add_argument("--data-path-test", default=[
                                                    "/cspca-data-test.xlsx"
                                                    ], help="DRIVE root")
    
    # exclude background
    parser.add_argument("--num-classes", default=2, type=int)
    parser.add_argument("--device", default="cuda:1", help="training device")
    parser.add_argument("-b", "--batch-size", default=8, type=int)
    parser.add_argument("--epochs", default=100, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=1e-6, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--model-path', default=None, help='pretrain model path')
    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')
    parser.add_argument('--myseed', type=int, default=myseed, help='input seed') 


    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args(myseed=42)

    if not os.path.exists("/result/save_weights"):
        os.mkdir("/result/save_weights")

    main(args)

    
