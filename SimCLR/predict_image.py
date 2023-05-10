import torch
from torchvision import transforms
from SimCLR import *
from PIL import Image
from torchvision.datasets import CIFAR10, CIFAR100

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize(32)
                                ])


def image_prediction(image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Clssifier(100, 0)
    model.load_model()
    model.eval()
    image_ = transform(image).unsqueeze(0).to(device)
    output = model(image_)
    output = torch.softmax(output, dim=1)
    prob, obj = output.topk(10)
    if torch.cuda.is_available():
        prob = prob.cpu()
        obj = obj.cpu()
    prob = prob.detach().numpy().reshape(-1)
    obj = obj.detach().numpy().reshape(-1)
    pred = {}
    for p, o in zip(prob, obj):
        pr = str(round(p*100, 2))
        pred[pr] = o
    return pred


def load_image(image_file):
    img = Image.open(image_file)
    return img


if __name__ == '__main__':
    #c_train = CIFAR100(DATA_ROOT_PATH, download=True, train=True)


    img_file_path = r"C:\Users\conta\Downloads\Screenshot 2023-05-10 092429.jpg"
    img = load_image(img_file_path)
    pred = image_prediction(img)
    print(pred)