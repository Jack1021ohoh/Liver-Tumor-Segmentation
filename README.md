# Handover Document: Liver Tumor Segmentation

### Purpose
Train deep learning model for liver tumor segmentation prediction.

### Environment
Windows 11 & Python 3.11.0  

GPU相關設置:  CUDA 11.8 & cuDNN  

可以用以下指令重新安裝套件
```
python -m pip install -r requirements.txt
```

### Dataset 
使用LiTS17和HCC-TACE-Seg這兩個公開資料集

LiTS17 Source Link: https://competitions.codalab.org/competitions/17094

HCC-TACE-Seg Source Link: https://www.cancerimagingarchive.net/collection/hcc-tace-seg/

LiTS17 切分方式
- Train-70% (91), Valid-10% (13), Test-20% (27)

HCC-TACE-Seg 切分方式
- Train-70% (52), Valid-10% (7), Test-20% (16)

### Experiments

- TransUNet Model Architecture
    - 利用pytorch手刻的TransUNet
    - 對應TransUNet裡面的三個檔案
- LiTS17 Liver Tumor Segmentation Training
    - 利用LiTS17 dataset訓練TransUNet進行liver tumor segmentation (包含有對loss做weighted以及沒有的這兩種版本)
    - 對應lits_segmentation_transunet.ipynb這個檔案
    - 最佳模型儲存在model_storage中的transunet和transunet_weighted的資料夾裡
- HCC-TACE-Seg Liver Tumor Segmentation Training
    - 利用HCC-TACE-Seg dataset訓練TransUNet進行liver tumor segmentation (包含有對loss做weighted以及沒有的這兩種版本)
    - 對應tcia_segmentation_transunet.ipynb這個檔案
    - 最佳模型儲存在model_storage中的transunet_tcia和transunet_weighted_tcia的資料夾裡

### Others
- other Python files: 針對LiTS17和HCC-TACE-Seg這兩個資料集的preprocessing
- 實驗過程進度報告: https://docs.google.com/presentation/d/1qHifSowSVYeCssV_ELl9qHmkAMDc9H4r6INF6ity1lE/edit?usp=sharing