# %%
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
import tqdm
import numpy as np
import utils
import dataloaders
import torchvision
from trainer import Trainer
# %%
torch.random.manual_seed(0)
np.random.seed(0)

# Load the dataset and print some stats
batch_size = 64

image_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

image_transform_normalize = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(0.5, 0.5),
])

dataloader_train, dataloader_test = dataloaders.load_dataset(
    batch_size, image_transform_normalize)

example_images, _ = next(iter(dataloader_train))
print(f"The tensor containing the images has shape: {example_images.shape} (batch size, number of color channels, height, width)",
      f"The maximum value in the image is {example_images.max()}, minimum: {example_images.min()}", sep="\n\t")


def create_model():
    """
        Initializes the mode. Edit the code below if you would like to change the model.
    """
    model = nn.Sequential(
        nn.Flatten(),  # Flattens the image from shape (batch_size, C, Height, width) to (batch_size, C*height*width)
        # nn.Linear(28*28*1, 10),
        nn.Linear(28*28*1, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
        # No need to include softmax, as this is already combined in the loss function
    )
    # Transfer model to GPU memory if a GPU is available
    model = utils.to_cuda(model)
    return model


model = create_model()


# Test if the model is able to do a single forward pass
example_images = utils.to_cuda(example_images)
output = model(example_images)
print("Output shape:", output.shape)
expected_shape = (batch_size, 10)  # 10 since mnist has 10 different classes
assert output.shape == expected_shape,    f"Expected shape: {expected_shape}, but got: {output.shape}"


# Hyperparameters
learning_rate = .0192
# learning_rate = 1.
num_epochs = 5


# Use CrossEntropyLoss for multi-class classification
loss_function = torch.nn.CrossEntropyLoss()

# Define optimizer (Stochastic Gradient Descent)
optimizer = torch.optim.SGD(model.parameters(),
                            lr=learning_rate)


trainer = Trainer(
    model=model,
    dataloader_train=dataloader_train,
    dataloader_test=dataloader_test,
    batch_size=batch_size,
    loss_function=loss_function,
    optimizer=optimizer
)
train_loss_dict, test_loss_dict = trainer.train(num_epochs)

# %%
# Plot loss
# utils.plot_loss(train_loss_dict, label="Train Loss")
# utils.plot_loss(test_loss_dict, label="Test Loss")
utils.plot_loss(train_loss_dict, label="Train Loss, with hidden layer")
utils.plot_loss(test_loss_dict, label="Test Loss, with hidden layer")
# Limit the y-axis of the plot (The range should not be increased!)
plt.ylim([0, 1])
plt.legend()
plt.xlabel("Global Training Step")
plt.ylabel("Cross Entropy Loss")
plt.savefig("image_solutions/task_4a.png")


plt.show()

torch.save(model.state_dict(), "saved_model.torch")
final_loss, final_acc = utils.compute_loss_and_accuracy(
    dataloader_test, model, loss_function)
print(f"Final Test loss: {final_loss}. Final Test accuracy: {final_acc}")


# %%
subplot_shape = (5, 2)
fig_weight, ax_weight = plt.subplots(*subplot_shape)
weight = list(model.children())[1].weight.cpu().data
for i in range(weight.shape[0]):
    img = np.array(weight[i].reshape(28, 28))
    arg = np.unravel_index(i, subplot_shape)
    ax_weight.flat[i].imshow(img, cmap='gray')
    ax_weight.flat[i].set_title(f'Weights for the number {i}')
for i in range(weight.shape[0], np.prod(subplot_shape)):
    ax_weight.flat[i].set_visible(False)
fig_weight.tight_layout()
# We can now plot the training loss with our utility script
