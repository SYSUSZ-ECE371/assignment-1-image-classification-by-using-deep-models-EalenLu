import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
from torch.utils.data import random_split
import os
import time
import copy

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

def predict_image(model, image_path, class_names, device):
    """
    预测单张花卉图片的类别和概率
    
    参数:
        model: 训练好的模型
        image_path: 图片文件路径
        class_names: 类别名称列表
        device: 计算设备(CPU/GPU)
    
    返回:
        (预测类别, 预测概率)
    """
    # 预测时的预处理（与验证集相同）
    prediction_transform = transforms.Compose([
        transforms.Resize(256),  # 调整大小
        transforms.CenterCrop(224),  # 中心裁剪
        transforms.ToTensor(),  # 转为张量
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化
    ])
    
    # 加载并预处理图像
    image = Image.open(image_path).convert('RGB')
    image_tensor = prediction_transform(image).unsqueeze(0)  # 增加批次维度
    image_tensor = image_tensor.to(device)
    
    # 模型预测
    model.eval()  # 设置为评估模式
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
    # 获取预测结果
    top_prob, top_idx = torch.topk(probabilities, k=3)  # 获取前三名的概率和索引
    top_prob = top_prob.cpu().numpy().squeeze()  # 转为numpy数组
    top_idx = top_idx.cpu().numpy().squeeze()
    
    # 创建预测结果字典
    predictions = []
    for i, (idx, prob) in enumerate(zip(top_idx, top_prob)):
        predictions.append({
            "rank": i+1,
            "class": class_names[idx],
            "probability": float(prob)
        })
    
    return predictions

def display_prediction(image_path, predictions):
    """
    显示预测结果和图片
    
    参数:
        image_path: 图片路径
        predictions: 预测结果列表
    """
    # 加载图片
    image = Image.open(image_path)
    
    # 创建图表
    plt.figure(figsize=(12, 6))
    
    # 显示图片
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title("Input Image")
    
    # 显示预测结果
    plt.subplot(1, 2, 2)
    classes = [p['class'] for p in predictions]
    probs = [p['probability'] for p in predictions]
    
    # 创建水平条形图
    y_pos = np.arange(len(classes))
    plt.barh(y_pos, probs, align='center', color='skyblue')
    plt.yticks(y_pos, classes)
    plt.xlabel('Probability')
    plt.title('Top Predictions')
    plt.xlim(0, 1)
    
    # 在条形图上添加概率值
    for i, prob in enumerate(probs):
        plt.text(prob + 0.01, i, f"{prob:.4f}", va='center')
    
    plt.tight_layout()
    plt.show()
    
    # 打印预测结果
    print("\n预测结果:")
    for prediction in predictions:
        print(f"{prediction['rank']}. {prediction['class']}: {prediction['probability']:.4f}")


def main():
    # Set data directory
    data_dir = 'EX2/flower_dataset'

    # Data augmentation and normalization for training and validation
    data_transforms = transforms.Compose([
            # GRADED FUNCTION: Add five data augmentation methods, Normalizating and Tranform to tensor
            ### START SOLUTION HERE ###

            # 随机裁剪为224x224大小
            transforms.RandomResizedCrop(224),
            # 随机旋转±30度
            transforms.RandomRotation(20),
            # 随机水平翻转
            transforms.RandomHorizontalFlip(),
            # 随机垂直翻转
            transforms.RandomVerticalFlip(),
            # 色彩抖动（亮度/对比度/饱和度/色相）
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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

    save_dir = 'Ex2/work_dir'
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    
    # 确保模型文件存在
    if not os.path.exists(best_model_path):
        print(f"未找到训练好的模型: {best_model_path}")
        return
    
    # 加载最佳模型
    best_model = models.resnet18(pretrained=False)
    num_ftrs = best_model.fc.in_features
    best_model.fc = nn.Linear(num_ftrs, num_classes)
    best_model.load_state_dict(torch.load(best_model_path))
    best_model = best_model.to(device)
    best_model.eval()
    
    print("\n模型加载完成，准备进行预测...")
    
    # 预测循环
    while True:
        print("\n" + "="*50)
        image_path = input("请输入花卉图片路径(或输入'quit'退出): ")
        
        if image_path.lower() == 'quit':
            break
        
        if not os.path.exists(image_path):
            print(f"文件不存在: {image_path}")
            continue
        
        # 进行预测
        predictions = predict_image(best_model, image_path, class_names, device)
        
        # 显示结果
        display_prediction(image_path, predictions)

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()