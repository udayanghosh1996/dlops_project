from torchvision import transforms
from SimCLR import *
from PIL import Image

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize(32)
                                ])


def image_prediction(image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Classifier(100)
    model.eval()
    image_ = transform(image)
    output = model(image_)
    prob, obj = output.topk(10)
    if torch.cuda.is_available():
        prob = prob.cpu()
        obj = obj.cpu()
    prob = prob.detach().numpy().reshape(-1)
    obj = obj.detach().numpy().reshape(-1)
    pred = {}
    for p, o in zip(prob, obj):
        pred[str(p)] = o
    return pred


def load_image(image_file):
    img = Image.open(image_file)
    return img
