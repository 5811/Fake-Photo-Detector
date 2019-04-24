# variations to try:
    # Naive bayes
    # getting rid of grayscale
    # getting rid of max pooling

import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from Net import Net
from Phase1DataSet import Phase1DataSet

WIDTH = 100
HEIGHT = 100
NUMBER_OF_COLOR_CHANNELS = 1
NUMBER_OF_FIRST_CONVOLUTION_OUTPUT_CHANNELS = 20
NUMBER_OF_SECOND_CONVOLUTION_OUTPUT_CHANNELS = 50
NUMBER_OF_FULLY_CONNECTED_NODES = 500


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        criterion = nn.L1Loss()
        loss = criterion(output, target.float())
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    total_num_tests = 0
    total_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            prediction = [o > 0.5 for o in output.tolist()]
            num_fake_images = len([p == 0 for p in prediction])
            print(f"Predicted num fake images: {num_fake_images}")
            num_correct = sum(v1 == v2 for (v1, v2) in zip(prediction, target.int().tolist()))
            total_correct += num_correct
            total_num_tests += len(target)

    print(f'Test set: Accuracy: {total_correct / total_num_tests * 100.0}% \n')

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Fake Photo Detector Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Build our training set
    training_dataset = Phase1DataSet(transform=transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((WIDTH, HEIGHT)), # all images will be resized
        transforms.Grayscale(), # we only care about one color channel
        transforms.ToTensor(), # conver numpy image to torch image
        transforms.Normalize((0.5,), (0.5,)) # normalize
    ]))

    training_dataset.load_images('training/pristine', 'png', 0)
    training_dataset.load_images('training/fake', 'png', 1)
    training_dataset.shuffle()

    # hold out 200 images for the test set
    testing_images = training_dataset.imagePaths[-200:]
    testing_labels = training_dataset.labels[-200:]

    training_dataset.imagePaths = training_dataset.imagePaths[:1300]
    training_dataset.labels = training_dataset.labels[:1300]

    testing_dataset =  Phase1DataSet(transform=transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((WIDTH, HEIGHT)), # all images will be resized
        transforms.Grayscale(), # we only care about one color channel
        transforms.ToTensor(), # conver numpy image to torch image
        transforms.Normalize((0.5,), (0.5,)) # normalize
    ]))
    testing_dataset.imagePaths = testing_images
    testing_dataset.labels = testing_labels

    train_loader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=args.batch_size, **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        testing_dataset,
        batch_size=args.test_batch_size, **kwargs
    )

    model = Net(WIDTH, HEIGHT,  NUMBER_OF_COLOR_CHANNELS, NUMBER_OF_FIRST_CONVOLUTION_OUTPUT_CHANNELS, NUMBER_OF_SECOND_CONVOLUTION_OUTPUT_CHANNELS, NUMBER_OF_FULLY_CONNECTED_NODES).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(),"fake_photo_detector_cnn.pt")

if __name__ == '__main__':
    main()