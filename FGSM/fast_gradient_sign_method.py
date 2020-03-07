# This is the code for adversarial attack using FGSM with max norm on images from MNIST dataset
import torch
import torch.nn as nn 
import torch.nn.functional as f
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

conv_kernel = ck = 5
conv_stride = cs = 1
pool_kernel = pk = 2
pool_stride = ps = 2
BATCH_SIZE = 128
epsilons = [0.00, 0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

transform = transforms.Compose([transforms.ToTensor()])
train_set = datasets.MNIST('../data', train = True, download = True, transform = transform)
test_set = datasets.MNIST('../data', train = False, download = True, transform = transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = 1, shuffle = True)

class network(nn.Module):
    def __init__(self, h_in, w_in, ops):
        super(network, self).__init__()
        self.conv0 = nn.Conv2d(1, 16, kernel_size = ck, stride = cs)
        self.conv1 = nn.Conv2d(16, 32, kernel_size = ck, stride = cs)
        self.conv_drop = nn.Dropout2d(p = 0.4)
        def get_ops(size, kernel_size, stride):
            return (size - kernel_size)//stride + 1            
        h_out, w_out = get_ops(get_ops(get_ops(get_ops(h_in, ck, cs), pk, ps), ck, cs), pk, ps), get_ops(get_ops(get_ops(get_ops(w_in, ck, cs), pk, ps), ck, cs), pk, ps)
        self.lin_ips = 32 * h_out * w_out
        self.lin0 = nn.Linear(self.lin_ips, 84)
        self.lin1 = nn.Linear(84, ops)
        
    def forward(self, t):
        t = f.relu(f.max_pool2d(self.conv0(t), pk, ps))
        t = f.relu(f.max_pool2d(self.conv_drop(self.conv1(t)), pk, ps))
        t = t.view(-1, self.lin_ips)
        t = f.relu(self.lin0(t))
        t = f.dropout(t)
        t = self.lin1(t)
        return f.log_softmax(t, dim = 1)

h_in = image.shape[2]
w_in = image.shape[3]
ops = 10
net = network(h_in, w_in, ops)
optimizer = optim.Adam(net.parameters(), lr = 0.01)

def get_num_correct(out_pred, target_batch):
    return (out_pred == target_batch).sum().item()

def train_net():
    for epoch in range(20):
        net_loss = 0
        net_correct = 0
        for batch in train_loader:
            image_batch, target_batch = batch
            out_prob = net(image_batch)
            loss = f.nll_loss(out_prob, target_batch)
            net_loss += loss.item()
            net.zero_grad()
            loss.backward()
            optimizer.step()
            out_pred = out_prob.max(1, keepdim = True)[1].squeeze()
            net_correct += get_num_correct(out_pred, target_batch)
        accuracy = net_correct/(float(len(train_set)))

# call this function to train the model on MNIST
train_net()

# Attack on the input observation (pixels of the image)
def fgsm_attack(original_image, epsilon, inp_grad):
    perturbed_image = original_image + epsilon * inp_grad.sign()
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def generate_adv_samples(epsilon):
    classified_correct = 0
    adversarial_ex = []
    
    for image, label in test_loader:
        image.requires_grad = True
        out_prob = net(image)
        out_pred = out_prob.max(1, keepdim = True)[1]
        
        # if the model misclassifies an image then we just move on to next sample
        if out_pred != label:
            continue
        
        # find the loss the unperturbed image obtains by a backward pass through the network
        loss = f.nll_loss(out_prob, label)
        net.zero_grad()
        loss.backward()
        inp_grad = image.grad.data
        
        # attack the correctly classified image sample
        perturbed_image = fgsm_attack(image, epsilon, inp_grad)
        
        # run a forward pass on the model again using the perturbed image
        prob_perturbed = net(perturbed_image)
        out_perturbed = prob_perturbed.max(1, keepdim = True)[1]
        
        # if the model misclassifies the perturbed image, then we have creatred an adversarial example
        if out_perturbed != out_pred:
            if len(adversarial_ex) < 5:
                perturbed_image = perturbed_image.squeeze().detach().numpy()
                adversarial_ex.append((out_pred, out_perturbed, perturbed_image))
        else:
            classified_correct += 1
            if (epsilon == 0.00) and (len(adversarial_ex) < 5):
                perturbed_image = perturbed_image.squeeze().detach().numpy()
                adversarial_ex.append((out_pred, out_perturbed, perturbed_image))
                
    accuracy = classified_correct/float(len(test_loader))
    return adversarial_ex, accuracy

accuracies = []
adversarial_ex = []

for epsilon in epsilons:
    adv_ex, acc = generate_adv_samples(epsilon)
    accuracies.append(acc)
    adversarial_ex.append(adv_ex)

# Plot for showing decreasing accuracy
plt.figure(figsize = (8, 10))
plt.xlabel('EPSILON')
plt.ylabel('ACCURACY')
plt.plot(accuracies)
plt.grid()
plt.show()

count = 0
plt.figure(figsize = (10, 10))
for i in range(len(epsilons)):
    for j in range(len(adversarial_ex[i])):
        count += 1
        plt.subplot(len(epsilons), len(adversarial_ex[0]),count)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        orig, adv, image = adversarial_ex[i][j]
        plt.title(f'{orig.item()} -> {adv.item()}')
        plt.imshow(image, cmap = 'gray')
plt.tight_layout()
plt.show()