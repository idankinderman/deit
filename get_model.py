import torch
import torchvision
from pathlib import Path


if __name__ == '__main__':
    model_path = "../output_dir/ViT-B-16/CIFAR/resnet.pth"
    save_path = "../output_dir/ViT-B-16/CIFAR/resnet50_1"
    model_name = "model.pt"
    data_set = "CIFAR"

    my_dict = torch.load(model_path)
    model_state = my_dict['model']
    best_epoch = my_dict['epoch']
    print("The best epoch: {}".format(best_epoch))

    model = torchvision.models.resnet50(pretrained=True, progress=True)
    if data_set == "CIFAR":
        model.fc = torch.nn.Linear(2048, 100, bias=True)

    model.load_state_dict(model_state)

    Path(save_path).mkdir(parents=True, exist_ok=True)
    torch.save(model, save_path + "/" + model_name)

    print("Done!")