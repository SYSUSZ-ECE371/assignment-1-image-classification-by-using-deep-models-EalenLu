import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
from torch.utils.data import random_split
import os
import time
import copy
import yaml
import datetime
import sys
import torchvision

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Learning Rate: {current_lr:.6f}')

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                save_dir = 'Ex2/work_dir'
                os.makedirs(save_dir, exist_ok=True)
                torch.save(best_model_wts, os.path.join(save_dir, 'best_model.pth'))

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model

def export_config(config, config_path="Ex1/config.yaml"):
    """导出配置文件到指定路径"""
    # 确保Ex1目录存在
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    # 添加元数据
    config['meta'] = {
        'export_time': datetime.datetime.now().isoformat(),
        'author': os.getlogin(),  # 获取当前用户名
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'torch_version': torch.__version__,
        'torchvision_version': torchvision.__version__
    }
    
    # 保存为YAML文件
    with open(config_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False, default_flow_style=False)
    
    print(f"✅ 配置文件已导出到 {os.path.abspath(config_path)}")

def main():
    # 配置字典 - 包含所有重要参数
    config = {
        'data': {
            'dir': 'EX2/flower_dataset',
            'transforms': [
                {'type': 'RandomResizedCrop', 'size': 224},
                {'type': 'RandomRotation', 'degrees': 30},
                {'type': 'RandomHorizontalFlip'},
                {'type': 'RandomVerticalFlip'},
                {'type': 'ColorJitter', 
                 'brightness': 0.2, 
                 'contrast': 0.2, 
                 'saturation': 0.2, 
                 'hue': 0.1},
                {'type': 'ToTensor'},
                {'type': 'Normalize', 
                 'mean': [0.485, 0.456, 0.406], 
                 'std': [0.229, 0.224, 0.225]}
            ],
            'split_ratio': [0.8, 0.2]  # 训练集/验证集比例
        },
        'model': {
            'name': 'resnet18',
            'pretrained': True
        },
        'training': {
            'batch_size': 32,
            'num_epochs': 25,
            'criterion': 'CrossEntropyLoss',
            'optimizer': {
                'type': 'SGD',
                'lr': 0.001,
                'momentum': 0.9
            },
            'scheduler': {
                'type': 'StepLR',
                'step_size': 7,
                'gamma': 0.1
            }
        }
    }
    # Set data directory
    data_dir = 'EX2/flower_dataset'

    # Data augmentation and normalization for training and validation
    data_transforms = transforms.Compose([
            # GRADED FUNCTION: Add five data augmentation methods, Normalizating and Tranform to tensor
            ### START SOLUTION HERE ###

            # 随机裁剪为224x224大小
            transforms.RandomResizedCrop(224),
            # 随机旋转±30度
            transforms.RandomRotation(30),
            # 随机水平翻转
            transforms.RandomHorizontalFlip(),
            # 随机垂直翻转
            transforms.RandomVerticalFlip(),
            # 色彩抖动（亮度/对比度/饱和度/色相）
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            # 转换为张量
            transforms.ToTensor(),
            # 标准化处理（使用ImageNet均值和标准差）
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

            # Add five data augmentation methods, Normalizating and Tranform to tensor
            ### END SOLUTION HERE ###
    ])

    # Load the entire dataset
    full_dataset = datasets.ImageFolder(data_dir, data_transforms)

    # Automatically split into 80% train and 20% validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Use DataLoader for both train and validation datasets
    batch_size = 32
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    dataloaders = {'train': train_loader, 'val': val_loader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

    # Get class names from the dataset
    class_names = full_dataset.classes

    config['model']['num_classes'] = len(class_names)  # 更新类别数
    export_config(config)  # 导出配置文件

    # Load pre-trained model and modify the last layer
    model = models.resnet18(pretrained=True)


    # GRADED FUNCTION: Modify the last fully connected layer of model
    ### START SOLUTION HERE ###
    # Modify the last fully connected layer of model

    # 获取原始全连接层的输入特征数
    num_ftrs = model.fc.in_features
    # 获取数据集中的类别数量
    num_classes = len(class_names)
    # 替换全连接层，输出维度改为实际类别数
    model.fc = nn.Linear(num_ftrs, num_classes)

    ### END SOLUTION HERE ###



    # GRADED FUNCTION: Define the loss function
    ### START SOLUTION HERE ###
    # Define the loss function

    # 使用交叉熵损失函数
    criterion = nn.CrossEntropyLoss()

    ### END SOLUTION HERE ###

    # GRADED FUNCTION: Define the optimizer
    ### START SOLUTION HERE ###
    # Define the optimizer

    # 使用随机梯度下降优化器（SGD）
    # lr=0.001 初始学习率，momentum=0.9 动量参数
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    ### END SOLUTION HERE ###

    # Learning rate scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Training function
    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Print learning rate for current epoch
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Learning Rate: {current_lr:.6f}')

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            
                            # GRADED FUNCTION: Backward pass and optimization
                            ### START SOLUTION HERE ###
                            # Backward pass and optimization

                            # 计算梯度
                            loss.backward()
                            # 更新模型参数
                            optimizer.step()

                            ### END SOLUTION HERE ###

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    scheduler.step()  # Update learning rate based on scheduler

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # Save the model if validation accuracy is the best so far
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    # Save the best model
                    save_dir = 'Ex2/work_dir'
                    os.makedirs(save_dir, exist_ok=True)

                # GRADED FUNCTION: Save the best model
                    ### START SOLUTION HERE ###
                    # Save the best model
                    
                    # 保存最佳模型权重
                    torch.save(best_model_wts, os.path.join(save_dir, 'best_model.pth'))
                    
                    ### END SOLUTION HERE ###

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        model.load_state_dict(best_model_wts)
        return model

    # Train the modelS
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = train_model(model, criterion, optimizer, scheduler, num_epochs=25)

    
    
if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()