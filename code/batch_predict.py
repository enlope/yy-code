import os
import argparse
import json
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_dataset import MyDataSet
from model import convnext_tiny as create_model
from utils import read_split_data, create_lr_scheduler, get_params_groups, train_one_epoch, evaluate, plot_data_loader_image
from PIL import Image

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    num_classes = 51
    img_size = 224
    data_transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.14)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # load image

    imgs_root = r"D:\test"
    assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
    # 读取指定文件夹下所有jpg图像路径
    imgs_labels = [os.path.join(imgs_root, img) for img in os.listdir(imgs_root)]
    print(imgs_labels)
    # for m in imgs_labels:
    #     # t=os.listdir(m)
    #     img_path_list = [os.path.join(m, i) for i in os.listdir(m) if i.endswith(".jpg")]
    img_path_list=[os.path.join(m, i) for m in imgs_labels for i in os.listdir(m)]
    print(img_path_list)
    # read class_indict
    json_path = 'D:\ConvNeXt/class_indices.json'
    assert os.path.exists(json_path), f"file: '{json_path}' dose not exist."
    json_file = open(json_path, "r")
    class_indict = json.load(json_file)#把json文件解码成一个字典，其内容是文件名和所对应的类别

    # create model
    model = create_model(num_classes=num_classes).to(device)

    # load model weights
    model_weight_path = "D:\ConvNeXt/weights/best_model.pth"
    assert os.path.exists(model_weight_path), f"file: '{model_weight_path}' dose not exist."
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()#关掉dropout放法
    # prediction
    batch_size = 8  # 每次预测时将多少张图片打包成一个batch
    with torch.no_grad():#让ptorch不去跟踪网络的损失梯度
        for ids in range(0, len(img_path_list) // batch_size):
            img_list = []
            for img_path in img_path_list[ids * batch_size: (ids + 1) * batch_size]:
                assert os.path.exists(img_path), f"file: '{img_path}' dose not exist."
                img = Image.open(img_path)
                img = data_transform(img)
                img_list.append(img)

            # batch img
            # 将img_list列表中的所有图像打包成一个batch
            batch_img = torch.stack(img_list, dim=0)
            # predict class
            output = model(batch_img.to(device)).cpu()#模型计算时batch_x加入GPU，返回值转回cpu
            predict = torch.softmax(output, dim=1)
            probs, classes = torch.max(predict, dim=1)

            for idx, (pro, cla) in enumerate(zip(probs, classes)):
                print("image: {}  class: {}  prob: {:.3}".format(img_path_list[ids * batch_size + idx],
                                                                 class_indict[str(cla.numpy())],
                                                                 pro.numpy()))


if __name__ == '__main__':
    main()
