# 양자화 실습
# https://gaussian37.github.io/dl-pytorch-quantization/
import os
import random
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def prepare_dataloader(num_workers=8, train_batch_size=128, eval_batch_size=256):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    train_set = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=test_transform)

    train_sampler = torch.utils.data.RandomSampler(train_set)
    test_sampler = torch.utils.data.SequentialSampler(test_set)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, sampler=train_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=eval_batch_size, sampler=test_sampler, num_workers=num_workers)

    return train_loader, test_loader

def evaluate_model(model, test_loader, device, criterion=None):

    model.eval()
    model.to(device)

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        if criterion is not None:
            loss = criterion(outputs, labels).item()
        else:
            loss = 0

        running_loss += loss * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(test_loader.dataset)
    eval_accuracy = running_corrects / len(test_loader.dataset)

    return epoch_loss, eval_accuracy

def train_model(model ,train_loader, test_loader, device,learning_rate=1e-1, num_epochs=200):

    criterion = nn.CrossEntropyLoss()

    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1, last_epoch=-1)

    model.eval()
    eval_loss, eval_accuracy = evaluate_model(model, test_loader, device, criterion)
    print("Epoch: {:02d} Eval Loss: {:.3f} Eval Acc: {:.3f}".format(-1, eval_loss, eval_accuracy))

    for epoch in range(num_epochs):

        model.train()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = running_corrects / len(train_loader.dataset)

        model.eval()
        eval_loss, eval_accuracy = evaluate_model(model, test_loader, device, criterion)

        scheduler.step()

        print("Epoch: {:02d} Train Loss: {:.3f} Train Acc: {:.3f} Eval Loss: {:.3f} Eval Acc: {:.3f}".format(epoch, train_loss, train_accuracy, eval_loss, eval_accuracy))

    return model

def calibrate_model(model, loader, device=torch.device("cpu:0")):

    model.to(device)
    model.eval()

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        _ = model(inputs)

def measure_inference_latency(model, device, input_size=(1, 3, 32, 32), num_samples=100, num_warmups=10):

    model.to(device)
    model.eval()

    x = torch.rand(size=input_size).to(device)

    with torch.no_grad():
        start_time = time.time()
        for _ in range(num_warmups):
            _ = model(x)
            torch.cuda.synchronize()
        end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_ave = elapsed_time / num_samples

    return elapsed_time_ave

def save_model(model, model_dir, model_filename):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.save(model.state_dict(), model_filepath)

def load_model(model, model_filepath, device):

    model.load_state_dict(torch.load(model_filepath, map_location=device))

def save_torchscript_model(model, model_dir, model_filename):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.jit.save(torch.jit.script(model), model_filepath)

def load_torchscript_model(model_filepath, device):

    model = torch.jit.load(model_filepath, map_location=device)
    return model

def create_model(num_classes=10):
    model = torchvision.models.resnet18(pretrained=True, num_classes=num_classes)

    return model

class QuanizedResNet18(nn.Module):

    def __init__(self, model_fp32):
        super(QuanizedResNet18, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.model_fp32 = model_fp32

    def forward(self, x):
        x = self.quant(x)
        x = self.model_fp32(x)
        x = self.dequant(x)
        return x

def model_equivalence(model_1, model_2, device, rtol=1e-05, atol=1e-08, num_tests=100, input_size=(1, 3, 32, 32)):

    model_1.to(device)
    model_2.to(device)

    model_1.eval()
    model_2.eval()

    for _ in range(num_tests):

        x = torch.rand(size=input_size).to(device)
        y1 = model_1(x).detach().cpu().numpy()
        y2 = model_2(x).detach().cpu().numpy()
        if np.allclose(a=y1, b=y2, rtol=rtol, atol=atol, equal_nan=False) == False:
            print("Model equivalence test sample failed")
            print(y1)
            print(y2)
            return False

    return True

def main():

    random_seed = 0
    num_classes = 10
    cuda_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu:0")

    model_dir = "saved_models"
    model_filename = "resnet18_cifar10.pt"
    quantized_model_filename = "resnet18_quantized_cifar10.pt"
    model_filepath = os.path.join(model_dir, model_filename)
    quantized_model_filepath = os.path.join(model_dir, quantized_model_filename)

    set_random_seeds(random_seed)

    model = create_model(num_classes=num_classes)

    train_loader, test_loader = prepare_dataloader(num_workers=8, train_batch_size=128, eval_batch_size=256)

    print("Training Model...")
    model = train_model(model, train_loader, test_loader, num_epochs=200, learning_rate=1e-1, device=cuda_device)
    save_model(model, model_dir, model_filename)

    model = load_model(model, model_filepath, device=cuda_device)

    model.to(cpu_device)
    fused_model = copy.deepcopy(model)

    model.train()
    fused_model.train()

    fused_model = torch.quantization.fuse_modules(fused_model, [['conv1', 'bn1', 'relu']], inplace=True)
    for module_name, module in fused_model.named_children():
        if 'layer' in module_name:
            for basic_block_name, basic_block in module.named_children():
                torch.quantization.fuse_modules(basic_block, [['conv1', 'bn1', 'relu'], ['conv2', 'bn2']], inplace=True)
                for sub_block_name, sub_block in basic_block.named_children():
                    if sub_block_name == 'downsample':
                        torch.quantization.fuse_modules(sub_block, [['0', '1']], inplace=True)

    print(model)
    print(fused_model)

    model.eval()
    fused_model.eval()
    assert model_equivalence(model, fused_model, device=cpu_device, rtol=1e-03, atol=1e-06, num_test=100, input_size=(1, 3, 32, 32)), "Fused model is not equivalent to the original model!"

    quantized_model = QuanizedResNet18(model)
    quantization_config = torch.quantization.get_default_qconfig('fbgemm')

    quantized_model.qconfig = quantization_config
    print(quantized_model.qconfig)

    torch.quantization.prepare_qat(quantized_model, inplace=True)

    print('Training QAT Model...')
    quantized_model.train()

    train_model(model=quantized_model, train_loader=train_loader, test_loader=test_loader, device=cuda_device, learning_rate=1e-3, num_epochs=10)

    quantized_model.to(cpu_device)

    quantized_model = torch.quantization.convert(quantized_model, inplace=True)

    quantized_model.eval()

    print(quantized_model)

    save_torchscript_model(quantized_model, model_dir, quantized_model_filename)

    quantized_jit_model = load_torchscript_model(model_filepath=quantized_model_filepath, device=cpu_device)

    _, fp32_eval_accuracy = evaluate_model(model=model, test_loader=test_loader, device=cpu_device, criterion=None)
    _, int8_eval_accuracy = evaluate_model(model=quantized_jit_model, test_loader=test_loader, device=cpu_device, criterion=None)

    print('FP32 evaluation accuracy: {:.3f}'.format(fp32_eval_accuracy))
    print('INT8 evaluation accuracy: {:.3f}'.format(int8_eval_accuracy))

    fp32_cpu_inference_latency = measure_inference_latency(model=model, device=cpu_device, input_size=(1,3,32,32), num_samples=100)
    int8_cpu_inference_latency = measure_inference_latency(model=quantized_model, device=cpu_device, input_size=(1,3,32,32), num_samples=100)
    int8_jit_cpu_inference_latency = measure_inference_latency(model=quantized_jit_model, device=cpu_device, input_size=(1,3,32,32), num_samples=100)
    fp32_gpu_inference_latency = measure_inference_latency(model=model, device=cuda_device, input_size=(1,3,32,32), num_samples=100)

    print("FP32 CPU Inference Latency: {:.2f} ms / sample".format(fp32_cpu_inference_latency * 1000))
    print("FP32 CUDA Inference Latency: {:.2f} ms / sample".format(fp32_gpu_inference_latency * 1000))
    print("INT8 CPU Inference Latency: {:.2f} ms / sample".format(int8_cpu_inference_latency * 1000))
    print("INT8 JIT CPU Inference Latency: {:.2f} ms / sample".format(int8_jit_cpu_inference_latency * 1000))

    if __name__ == '__main__':
        main()
