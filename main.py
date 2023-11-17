from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchsummary
import torch.jit


class DCGANGenerator (nn.Module):
    def __init__(self, kNoiseSize: int):
        super(DCGANGenerator, self).__init__();
        # layer 1
        self.conv1 = nn.ConvTranspose2d(kNoiseSize, 256, 4, bias=False);
        self.batch_norm1 = nn.BatchNorm2d(256);
        # layer 2
        self.conv2 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, bias=False);
        self.batch_norm2 = nn.BatchNorm2d(128);
        # layer 3
        self.conv3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False);
        self.batch_norm3 = nn.BatchNorm2d(64);
        # layer 4 (produce output)
        self.conv4 = nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1, bias=False);
        
        # layer 1
        self.register_module("conv1", self.conv1);
        self.register_module("batch_norm1", self.batch_norm1);
        # layer 2
        self.register_module("conv2", self.conv2);
        self.register_module("batch_norm2", self.batch_norm2);
        # layer 3
        self.register_module("conv3", self.conv3);
        self.register_module("batch_norm3", self.batch_norm3);
        # layer 4 (produce output)
        self.register_module("conv4", self.conv4);
    
    def forward(self, x: torch.Tensor):
        # layer 1
        x = torch.relu(self.batch_norm1(self.conv1(x)));
        # layer 2
        x = torch.relu(self.batch_norm2(self.conv2(x)));
        # layer 3
        x = torch.relu(self.batch_norm3(self.conv3(x)));
        # layer 4 (produce output)
        x = torch.tanh(self.conv4(x));
        return x;
    
    
discriminator = nn.Sequential(
    # layer 1
    nn.Conv2d(1, 64, 4, stride=2, padding=1, bias=False),
    nn.LeakyReLU(negative_slope=0.2),
    # layer 2
    nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(negative_slope=0.2),
    # layer 3
    nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(negative_slope=0.2),
    # layer 4
    nn.Conv2d(256, 1, 3, stride=1, padding=0, bias=False),
    nn.Sigmoid()
);


if (__name__ == "__main__"):
    print("Initialize device", end=" :: ");
    device: torch.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu");
    print(device);
    
    print("Initialize DCGAN");
    kNoiseSize: int = 100;
    generator: DCGANGenerator = DCGANGenerator(kNoiseSize);
    generator.eval()
    
    print("Initialize dataset" , end=" :: ");
    kBatchSize: int = 64;
    kDataLoaderWorkers: int = 2;
    dataset = dset.MNIST(root="D:\\usr\\rapee\\projects\\cmake-exercises\\cconsole-libtorch\\x64\\Release", 
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]));
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=kBatchSize, shuffle=True, num_workers=kDataLoaderWorkers);
    print(f"{len(dataset)} images", end=" :: batch target: ");
    for batch in dataloader:
        target = batch[1];
        print(target.size(), end=" :: batch data: ");
        data = batch[0];
        print(data.size())
        break;
    
    model_file = "model-store/jit_trace_generator.pt"
    example = torch.rand((1, 100, 1, 1))
    torch_script = torch.jit.trace(generator, example)
    torch_script.save(model_file)
    print("Save JIT trace model.. generator ::", model_file, "::", type(torch_script))
    torch_script = torch.jit.load(model_file)
    print("Load JIT trace model.. generator ::", model_file, "::", type(torch_script))
    
    model_file = "model-store/jit_script_generator.pt"
    torch_script = torch.jit.script(generator)
    torch_script.save(model_file)
    print("Save JIT script model.. generator ::", model_file, "::", type(torch_script))    
    torch_script = torch.jit.load(model_file)
    print("Load JIT script model.. generator ::", model_file, "::", type(torch_script))
    
    model_file = "model-store/jit_trace_discriminator.pt"
    example = torch.rand((1, 1, 28, 28))
    torch_script = torch.jit.trace(discriminator, example)
    torch_script.save(model_file)
    print("Save JIT trace model.. discriminator ::", model_file, "::", type(torch_script))
    torch_script = torch.jit.load(model_file)
    print("Load JIT trace model.. discriminator ::", model_file, "::", type(torch_script))
    
    model_file = "model-store/jit_script_discriminator.pt"
    torch_script = torch.jit.script(discriminator)
    torch_script.save(model_file)
    print("Save JIT script model.. discriminator ::", model_file, "::", type(torch_script))
    torch_script = torch.jit.load(model_file)
    print("Load JIT script model.. discriminator ::", model_file, "::", type(torch_script))
    
    model_file = "model-store/state_generator.pt"
    print("Save state model.. generator ::", model_file)
    torch.save(generator.state_dict(), model_file)
    state = torch.load(model_file)
    print("Load state model.. generator ::", model_file, "::", type(state))
    generator.load_state_dict(state)
    
    model_file = "model-store/pickled_generator.pt"
    print("Save pickled model.. generator ::", model_file, "::", type(generator))
    torch.save(generator, model_file)
    model = torch.load(model_file)
    print("Load pickled model.. generator (local class defined) ::", model_file, "::", type(model))
    try:
        del DCGANGenerator
        model = torch.load(model_file)
    except AttributeError as e:
        print("Load pickled model.. generator (local class undefined) ::", model_file, "::", e)
    
    model_file = "model-store/state_discriminator.pt"
    print("Save state model.. discriminator ::", model_file)
    torch.save(discriminator.state_dict(), model_file)
    state = torch.load(model_file)
    print("Load state model.. discriminator ::", model_file, "::", type(state))
    discriminator.load_state_dict(state)
    
    model_file = "model-store/pickled_discriminator.pt"
    print("Save pickled model.. discriminator ::", model_file, "::", type(discriminator))
    torch.save(discriminator, model_file)
    model = torch.load(model_file)
    print("Load pickled model.. discriminator (local class defined) ::", model_file, "::", type(model))
    try:
        del torch.nn.modules.container.Sequential
        model = torch.load(model_file)
    except AttributeError as e:
        print("Load pickled model.. discriminator (local class undefined) ::", model_file, "::", e)
    
    # summary model
    print("\nGenerator:")
    torchsummary.summary(model=generator, input_data=[100, 1, 1]);
    print("\nDiscriminator:")
    torchsummary.summary(model=discriminator, input_data=[1, 28, 28]);
