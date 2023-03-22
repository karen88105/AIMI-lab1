# AIMI-lab1
Training model to detect Pneumonia from chest X-ray images

This project uses Pytorch to train model. You must check you have pytorch and GPU for training.

### Train
1. Download `train.py`
2. Check your environment have python, GPU and Pytorch
3. Adjustment parameters for training
4. Start training.
```
python train.py
```

### Adjustment parameters
```
#儲存圖片的編號
num = 16  #images name number

#繪製accuracy plot
plot_accuracy(train_acc_list, val_acc_list)

#繪製accuracy plot
plot_f1_score(f1_score_list)

#繪製confusion_matrix
plot_confusion_matrix(confusion_matrix)

#Training基本設定
parser.add_argument('--num_epochs', type=int, required=False, default=35)
parser.add_argument('--batch_size', type=int, required=False, default=32)
parser.add_argument('--lr', type=float, default=1e-5)  #1e-5
parser.add_argument('--wd', type=float, default=0.9)

#Data Augementation
train_dataset = ImageFolder(root=os.path.join(args.dataset, 'train'),
                            transform = transforms.Compose([
                                        transforms.Resize((args.resize, args.resize)),        
                                        transforms.RandomRotation(args.degree, resample=False),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        transforms.ToTensor(),
                                        ]))  #transforms.Normalize(mean, std)

#Train model
model = models.resnext50_32x4d(pretrained=True)

#Optimizer
optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)
```

# Environment
### Anaconda env
```
#更新apt
sudo apt update
#安裝curl
sudo apt install curl
# 下載 Anaconda 安裝檔案
curl -O https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
bash Anaconda3-2019.10-Linux-x86_64.sh

# 生效conda 指令
conda init
source ~/.bashrc
export PATH=~/anaconda3/bin:$PATH

#查看環境
conda info --env
#建立新環境
conda create --name myenv python=3.7
#建立環境錯誤
conda clean -i
conda clean -a

#啟動新環境
source activate myenv
```
### Pytorch
```
conda install pytorch==1.8.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.1 -c pytorch
```
