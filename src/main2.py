# variations to try:
    # Naive bayes
    # getting rid of max pooling

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from Phase2Net import get_model
from Phase1DataSet import Phase1DataSet

WIDTH = 64
HEIGHT = 64

def print_memory_info(device=None):
    GB = 1_000_000_000
    if torch.cuda.is_available():
        print('')
        print(f'Memory allocated: {torch.cuda.memory_allocated() / GB}')
        print(f'Max memory allocated: {torch.cuda.max_memory_allocated() / GB}')
        print(f'Memory cached: {torch.cuda.memory_cached() / GB}')
        print(f'Max memory cached: {torch.cuda.max_memory_cached() / GB}')
        print('')

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.L1Loss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data).squeeze()
        
        loss = criterion(output, target.float())
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * train_loader.batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            print_memory_info(device)

def test(args, model, device, test_loader):
    model.eval()
    total_num_tests = 0
    total_correct = 0
    total_distance = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data).squeeze()

            # compute sum of squared differences from the correct answers
            diff = sum((t-o)*(t-o) for (t, o) in zip(target.tolist(), output.tolist()))
            print(f"Batch squared diff: {diff}")
            total_distance += diff
            
            # compute number of incorrect images
            predictions = [o > 0.5 for o in output.tolist()]
            num_fake_images = len([p for p in predictions if p == 1])
            print(f"Predicted num fake images: {num_fake_images}")

            num_correct = sum(v1 == v2 for (v1, v2) in zip(predictions, target.int().tolist()))
            total_correct += num_correct
            total_num_tests += len(target)

            print_memory_info(device)

    print('------------------------------------------------------')
    # Best accuracy: 100, worst accuracy: 0
    print(f'Test set: Accuracy: {total_correct / total_num_tests * 100.0}%')
    # Worst distance: 0, best distance: 1
    print(f'Test set: Normalized squared distance: {total_distance / total_num_tests}')
    print('------------------------------------------------------')

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Fake Photo Detector Example')
    parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 5)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    transformParameters = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((WIDTH, HEIGHT)), # all images will be resized
        #transforms.Grayscale(), # we only care about one color channel
        transforms.ToTensor(), # conver numpy image to torch image
        
        #compute means and standard deviations of color channels for normalization 
        transforms.Normalize((127.5, 127.5, 127.5,), (127.5, 127.5, 127.5,)) # normalize
    ])
    
    # Build our training set
    training_dataset = Phase1DataSet(transform=transformParameters)

    training_dataset.load_images('training/pristine_patches', 'png', 0)
    pristineHoldout = training_dataset.imagePaths[-1000:]
    pristineLabels = training_dataset.labels[-1000:]
    training_dataset.imagePaths = training_dataset.imagePaths[:-1000]
    training_dataset.labels = training_dataset.labels[:-1000]

    training_dataset.load_images('training/fake_patches', 'png', 1)
    fakeHoldout = training_dataset.imagePaths[-1000:]
    fakeLabels = training_dataset.labels[-1000:]
    training_dataset.imagePaths = training_dataset.imagePaths[:-1000]
    training_dataset.labels = training_dataset.labels[:-1000]

    training_dataset.shuffle()

    testing_dataset = Phase1DataSet(transform=transformParameters)
    testing_dataset.imagePaths = [*pristineHoldout, *fakeHoldout]
    testing_dataset.labels = [*pristineLabels, *fakeLabels]

    train_loader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=args.batch_size, **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        testing_dataset,
        batch_size=args.test_batch_size, **kwargs
    )

    # https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#
    model = get_model().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # decay learning rate every 7 epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
        exp_lr_scheduler.step()

    if (args.save_model):
        torch.save(model.state_dict(),"fake_photo_detector_cnn.pt")

if __name__ == '__main__':
    main()
