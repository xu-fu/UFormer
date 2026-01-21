from monai.utils import first, set_determinism
from monai.transforms import (
    Transform,
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    Lambdad,
    Lambda,
    RandSpatialCropd,
    SpatialPadd,
    RandFlipd,
    Flipd,
    Resized,
    RandRotated,
    Rotated,
    ToTensord,
    HistogramNormalized,
    RandAffine,
    Rand3DElasticd,
    RandGaussianNoised,
    ScaleIntensityRanged,
    RandAdjustContrastd,
    RandBiasFieldd,
    RandHistogramShiftd,
    RandRicianNoised,
)

from monai.data import DataLoader, Dataset
import os
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
from skimage import filters, morphology, measure

def genimgandlabelfromexcel(patientdir, clifile):
    """
    从excel文件中读取图像路径和label
    """
    imgxls = pd.read_excel(clifile, sheet_name=0, dtype=object)
    imgbasepth = os.path.join(patientdir, 'image/')

    idpths = imgbasepth + imgxls['dirname']
    id = list(imgxls['dirname'])
    label = list(imgxls['cspca'])

    t2pthlist = []
    dwipthlist = []
    adcpthlist = []
    for i in idpths:
        t2pth = os.path.join(i, 'T2.nii')
        dwipth = os.path.join(i, 'DWI.nii')
        adcpth = os.path.join(i, 'ADC.nii')
        t2pthlist.append(t2pth)
        dwipthlist.append(dwipth)
        adcpthlist.append(adcpth)


    return t2pthlist, dwipthlist, adcpthlist, label, id


def genimgandlabelfrommulexcel(patientdirlist, clifilelist):
    """
    从多个excel文件中读取图像路径和label
    """
    t2pthlist = []
    dwipthlist = []
    adcpthlist = []
    labellist = []
    idlist = []
    for j in range(len(patientdirlist)):

        imgxls = pd.read_excel(clifilelist[j], sheet_name=0, dtype=object)
        imgbasepth = os.path.join(patientdirlist[j], 'ai', 'image/')

        idpths = imgbasepth + imgxls['dirname']
        idlist1 = list(imgxls['dirname'])
        labellist1 = list(imgxls['cspca'])

        
        for i in range(len(idpths)):
            t2pth = os.path.join(idpths[i], 'T2.nii')
            dwipth = os.path.join(idpths[i], 'DWI.nii')
            adcpth = os.path.join(idpths[i], 'ADC.nii')
            label = labellist1[i]
            id = idlist1[i]
            t2pthlist.append(t2pth)
            dwipthlist.append(dwipth)
            adcpthlist.append(adcpth)
            labellist.append(label)
            idlist.append(id)


    return t2pthlist, dwipthlist, adcpthlist, labellist, idlist

def genimgandmask(indir):
    """
    从输入文件夹中读取图像和mask路径
    """
    imgpth = os.path.join(indir, 'image')
    maskpth = os.path.join(indir, 'mask_l')
    imglist = os.listdir(imgpth)
    train_images = [os.path.join(imgpth, f'{i}', 'T2.nii') for i in imglist]
    train_labels = [os.path.join(maskpth, f'{i}', 'mask.nii') for i in imglist]
    return train_images, train_labels

def My3DDataLoader(patientdirlist, clifilelist, transform, batch_size, num_workers, shuffle):
    """
    根据monai定义Dataloader
    """
    # set_determinism(seed=0)
    t2pthlists, dwipthlists, adcpthlists, labels, ids = genimgandlabelfrommulexcel(patientdirlist, clifilelist)  ##从excel中读取数据
    data_dicts = [{"T2": t2pthlist, 
                   "DWI": dwipthlist,
                   "ADC":adcpthlist,
                   "label": label, 
                   'id': id} for t2pthlist, dwipthlist, adcpthlist, label, id in zip(t2pthlists, dwipthlists, adcpthlists, labels, ids)]

    check_ds = Dataset(data=data_dicts, transform=transform)
    check_loader = DataLoader(check_ds, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

    return check_loader



def My3DDataLoadertrainval(patientdirlist, clifilelist, traintransform, valtransform, batch_size, num_workers, shuffle):
    """
    根据monai定义Dataloader trainval
    """
    # set_determinism(seed=0)
    t2pthlists, dwipthlists, adcpthlists, labels, ids = genimgandlabelfrommulexcel(patientdirlist, clifilelist)  ##从excel中读取数据
    indexlist = [i for i in range(len(t2pthlists))]
    # 打乱列表
    random.shuffle(indexlist)
    trainid = int(len(t2pthlists)*0.8)

    traint2pthlists = np.array(t2pthlists)[indexlist[:trainid]]
    traindwipthlists = np.array(dwipthlists)[indexlist[:trainid]]
    trainadcpthlists = np.array(adcpthlists)[indexlist[:trainid]]
    trainlabels = np.array(labels)[indexlist[:trainid]]
    trainids = np.array(ids)[indexlist[:trainid]]


    valt2pthlists = np.array(t2pthlists)[indexlist[trainid:]]
    valdwipthlists = np.array(dwipthlists)[indexlist[trainid:]]
    valadcpthlists = np.array(adcpthlists)[indexlist[trainid:]] 
    vallabels = np.array(labels)[indexlist[trainid:]]
    valids = np.array(ids)[indexlist[trainid:]]

    traindata_dicts = [{"T2": traint2pthlist, 
                   "DWI": traindwipthlist,
                   "ADC": trainadcpthlist,
                   "label": trainlabel, 
                   'id': trainid} for traint2pthlist, traindwipthlist, trainadcpthlist, trainlabel, trainid in zip(traint2pthlists, traindwipthlists, trainadcpthlists, trainlabels, trainids)]

    traincheck_ds = Dataset(data=traindata_dicts, transform=traintransform)
    traincheck_loader = DataLoader(traincheck_ds, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

    valdata_dicts = [{"T2": valt2pthlist, 
                   "DWI": valdwipthlist,
                   "ADC": valadcpthlist,
                   "label": vallabel, 
                   'id': valid} for valt2pthlist, valdwipthlist, valadcpthlist, vallabel, valid in zip(valt2pthlists, valdwipthlists, valadcpthlists, vallabels, valids)]

    valcheck_ds = Dataset(data=valdata_dicts, transform=valtransform)
    valcheck_loader = DataLoader(valcheck_ds, batch_size=4, num_workers=1, shuffle=False)

    return traincheck_loader, valcheck_loader


def My3DDataLoadertest(patientdirlist, clifilelist, transform, batch_size, num_workers, shuffle):
    """
    根据monai定义Dataloader test
    """
    # set_determinism(seed=0)
    t2pthlists, dwipthlists, adcpthlists, labels, ids = genimgandlabelfrommulexcel(patientdirlist, clifilelist)  ##从excel中读取数据
    data_dicts = [{"T2": t2pthlist, 
                   "DWI": dwipthlist,
                   "ADC":adcpthlist,
                   "label": label, 
                   'id': id} for t2pthlist, dwipthlist, adcpthlist, label, id in zip(t2pthlists, dwipthlists, adcpthlists, labels, ids)]

    check_ds = Dataset(data=data_dicts, transform=transform)
    check_loader = DataLoader(check_ds, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

    return check_loader


def genimgmaskandlabelfrommulexcel(patientdirlist, clifilelist):
    """
    从多个excel文件中读取图像路径和label
    """
    t2pthlist = []
    dwipthlist = []
    adcpthlist = []
    labellist = []
    idlist = []
    masklist = []
    for j in range(len(patientdirlist)):

        imgxls = pd.read_excel(clifilelist[j], sheet_name=0, dtype=object)
        imgbasepth = os.path.join(patientdirlist[j], 'ai', 'image/')
        maskbasepth = os.path.join(patientdirlist[j], 'ai', 'mask_l/')

        idpths = imgbasepth + imgxls['dirname']
        idlist1 = list(imgxls['dirname'])
        labellist1 = list(imgxls['cspca'])
        maskpths = maskbasepth + imgxls['dirname']

        
        for i in range(len(idpths)):
            t2pth = os.path.join(idpths[i], 'T2.nii')
            dwipth = os.path.join(idpths[i], 'DWI.nii')
            adcpth = os.path.join(idpths[i], 'ADC.nii')
            maskpth = os.path.join(maskpths[i], 'mask.nii')
            label = labellist1[i]
            id = idlist1[i]
            t2pthlist.append(t2pth)
            dwipthlist.append(dwipth)
            adcpthlist.append(adcpth)
            labellist.append(label)
            idlist.append(id)
            masklist.append(maskpth)


    return t2pthlist, dwipthlist, adcpthlist, masklist, labellist, idlist

def My3DDataLoadertrainvalmask(patientdirlist, clifilelist, traintransform, valtransform, batch_size, num_workers, shuffle):
    """
    根据monai定义Dataloader trainval
    """
    # set_determinism(seed=0)
    t2pthlists, dwipthlists, adcpthlists, maskpthlists, labels, ids = genimgmaskandlabelfrommulexcel(patientdirlist, clifilelist)  ##从excel中读取数据
    indexlist = [i for i in range(len(t2pthlists))]
    # 打乱列表
    random.shuffle(indexlist)
    trainid = int(len(t2pthlists)*0.8)

    traint2pthlists = np.array(t2pthlists)[indexlist[:trainid]]
    traindwipthlists = np.array(dwipthlists)[indexlist[:trainid]]
    trainadcpthlists = np.array(adcpthlists)[indexlist[:trainid]]
    trainmaskpthlists = np.array(maskpthlists)[indexlist[:trainid]]
    trainlabels = np.array(labels)[indexlist[:trainid]]
    trainids = np.array(ids)[indexlist[:trainid]]


    valt2pthlists = np.array(t2pthlists)[indexlist[trainid:]]
    valdwipthlists = np.array(dwipthlists)[indexlist[trainid:]]
    valadcpthlists = np.array(adcpthlists)[indexlist[trainid:]] 
    valmaskpthlists = np.array(maskpthlists)[indexlist[trainid:]] 
    vallabels = np.array(labels)[indexlist[trainid:]]
    valids = np.array(ids)[indexlist[trainid:]]

    traindata_dicts = [{"T2": traint2pthlist, 
                   "DWI": traindwipthlist,
                   "ADC": trainadcpthlist,
                   "mask": trainmaskpthlist,
                   "label": trainlabel, 
                   'id': trainid} for traint2pthlist, traindwipthlist, trainadcpthlist, trainmaskpthlist, trainlabel, trainid in zip(traint2pthlists, traindwipthlists, trainadcpthlists, trainmaskpthlists, trainlabels, trainids)]

    traincheck_ds = Dataset(data=traindata_dicts, transform=traintransform)
    traincheck_loader = DataLoader(traincheck_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    valdata_dicts = [{"T2": valt2pthlist, 
                   "DWI": valdwipthlist,
                   "ADC": valadcpthlist,
                   "mask": valmaskpthlist,
                   "label": vallabel, 
                   'id': valid} for valt2pthlist, valdwipthlist, valadcpthlist, valmaskpthlist, vallabel, valid in zip(valt2pthlists, valdwipthlists, valadcpthlists, valmaskpthlists, vallabels, valids)]

    valcheck_ds = Dataset(data=valdata_dicts, transform=valtransform)
    valcheck_loader = DataLoader(valcheck_ds, batch_size=4, num_workers=1, shuffle=False)

    return traincheck_loader, valcheck_loader


def My3DDataLoadertestmask(patientdirlist, clifilelist, transform, batch_size, num_workers, shuffle):
    """
    根据monai定义Dataloader test
    """
    # set_determinism(seed=0)
    t2pthlists, dwipthlists, adcpthlists, maskpthlists, labels, ids = genimgmaskandlabelfrommulexcel(patientdirlist, clifilelist)  ##从excel中读取数据
    data_dicts = [{"T2": t2pthlist, 
                   "DWI": dwipthlist,
                   "ADC":adcpthlist,
                   "mask":maskpthlist,
                   "label": label, 
                   'id': id} for t2pthlist, dwipthlist, adcpthlist, maskpthlist, label, id in zip(t2pthlists, dwipthlists, adcpthlists, maskpthlists, labels, ids)]

    check_ds = Dataset(data=data_dicts, transform=transform)
    check_loader = DataLoader(check_ds, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

    return check_loader

