import os
import time
import datetime
import random
import torch
import numpy as np
from mydataset3d import My3DDataLoadertest
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


    
)






def SegmentationPresetTrain():
    train_transforms = Compose(
            [
                LoadImaged(keys=["T2", "DWI", "ADC"]),
                EnsureChannelFirstd(keys=["T2", "DWI", "ADC"]),
                RandFlipd(keys=["T2", "DWI", "ADC"], spatial_axis=0, prob=0.5),
                RandRotated(keys=["T2", "DWI", "ADC"], mode=['bilinear', 'bilinear', 'bilinear'], range_z=0.3, prob=0.2),
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




def create_model():
    model = restranscross503(pretrain=False,num_classes=2)

    return model


def evaluateoneepoch(model, lossdict, data_loader, device, num_classes):
    '''
    模型评估
    '''
    model.eval()

    
    e_loss = 0
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

            idlist.append(id)

            mask = 1

            label = torch.Tensor(label).long().to(device)
            imaget2 = torch.permute(imaget2*mask, (0, 1, 4, 2, 3)).to(device)
            imagedwi = torch.permute(imagedwi*mask, (0, 1, 4, 2, 3)).to(device)
            imageadc = torch.permute(imageadc*mask, (0, 1, 4, 2, 3)).to(device)


            t2dwiadc, features, outputo = model(imaget2, imagedwi, imageadc)
            output = outputo


            scorelist.append(torch.softmax(output, dim=1).detach().cpu().numpy())
            ylist.append(label.detach().cpu().numpy())


            
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
        torch.cuda.manual_seed_all(args.myseed)  # 如果有多个GPU
    # 创建生成器并手动设定种子
    g = torch.Generator()
    g.manual_seed(args.myseed)
    # set_determinism(seed=args.myseed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    num_classes = args.num_classes #+ 1


    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
   
    test_loader = My3DDataLoadertest(patientdirlist=args.patientdir_test, clifilelist=args.data_path_test, transform=SegmentationPresetEval(), batch_size=4, num_workers=num_workers, shuffle=False)

    model = create_model()
    # load weights
    weights_path = args.model_path
    if weights_path is not None:
        print('==========load pretrained model==========')
        model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'], strict=True)
        
        model.to(device)
        print(f'----------using device: {device} ----------')

        ce_loss = nn.CrossEntropyLoss()
        cos_loss = CosLoss(dim=1)
        con_loss = ContrastiveLoss()
        asoft_loss = AngularPenaltySMLoss(in_features=2048, out_features=2, loss_type='cosface', device=device)
        center_loss = CenterLoss(num_classes=2, feat_dim=1024, device=device)

        lossdict = {
                'ce_loss'    : ce_loss,
                'cos_loss'   : cos_loss,
                'con_loss'   : con_loss,
                'asoft_loss' : asoft_loss,
                'center_loss': center_loss
                }
            

        start_time = time.time()
        # 测试集测试模型结果
        testloss, testlossce, test_score, test_true, testidlist = evaluateoneepoch(model, lossdict, test_loader, device=device, num_classes=num_classes)
        testcal_eval_ind, testauc, testacc = Calculateindicators(test_score, test_true, testidlist, model='test', save=True)
        logger.info(f"test indicators: {testcal_eval_ind}")


        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("test time {}".format(total_time_str))
    else:
        print('=========== no model ============')



def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch training")

    parser.add_argument("--patientdir-test", default=["/root"], help="DRIVE root")
    
    parser.add_argument("--data-path-test", default=["/root/cspca-data.xlsx"], help="DRIVE root")
    # exclude background
    parser.add_argument("--num-classes", default=2, type=int)
    parser.add_argument("--device", default="cuda:1", help="training device")
    parser.add_argument("-b", "--batch-size", default=8, type=int)
    modelpath = '/model.pth'
    parser.add_argument('--model-path', default=modelpath, help='pretrain model path')
    parser.add_argument('--myseed', type=int, default=25, help='input seed') ###42   34078*64*64


    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()
    main(args)

