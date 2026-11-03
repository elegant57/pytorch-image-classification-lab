from torchvision import transforms
from model_save import *

def test(dataset):
    model.eval()
    with torch.no_grad():
        accuracy = 0
        for data in dataset:
            imgs, labels = data
            imgs = imgs.to(device)
            labels = labels.to(device)
            out_imgs = model(imgs)
            pred = torch.argmax(out_imgs, dim=1)
            accuracy += torch.sum(pred == labels).item()
            print(accuracy)
            if len(pred) < 64:
                print("实际的标签为：",labels)
                print("测试的标签为：",pred)
        print("准确率为{:.2f}%".format(accuracy / 10000 * 100))

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trans = transforms.Compose([transforms.ToTensor(), transforms.Resize((32, 32))])
    Dataset = torchvision.datasets.CIFAR10(root='./data/data_CIFAR', train=False, download=True, transform=trans)
    dataloader = torch.utils.data.DataLoader(Dataset, batch_size=64, shuffle=True)
    Cifar100Net = CIFAR100Net()
    model = torch.load('CIFAR100_net_trained.pth', weights_only=False)
    test(dataloader)