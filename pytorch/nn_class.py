import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class nn_pure:
    
    def __init__(self):
        self.model = None
    
    def build(self, input_size, output_size):
        self.model = nn.Linear(input_size, output_size)

    def train(self, num_epochs, x_train, y_train):
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        # Train the model
        for epoch in range(num_epochs):
            # Convert numpy arrays to torch tensors
            inputs = torch.from_numpy(x_train)
            targets = torch.from_numpy(y_train)

            # Forward pass
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad() #自動計算
            loss.backward()
            optimizer.step()

            if (epoch+1) % 5 == 0:
                print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch +
                        1, num_epochs, loss.item()))

        # Save the model checkpoint
        checkpoint = torch.save(self.model.state_dict(), 'model.ckpt')
        # print(checkpoint)

    def plot(self):
        # Plot the graph
        predicted = self.model(torch.from_numpy(x_train)).detach().numpy()
        plt.plot(x_train, y_train, 'ro', label='Original data')
        plt.plot(x_train, predicted, label='Fitted line')
        plt.legend()
        plt.show()


# Hyper-parameters
input_size = 1
output_size = 1
num_epochs = 60
learning_rate = 0.001

"""

# Linear regression model
model = nn.Linear(input_size, output_size)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    # Convert numpy arrays to torch tensors
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 5 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch +
                                                   1, num_epochs, loss.item()))



# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
"""

if __name__ == "__main__":
    # test nn
    x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                        [9.779], [6.182], [7.59], [2.167], [7.042],
                        [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

    y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                        [3.366], [2.596], [2.53], [1.221], [2.827],
                        [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)
    
    moded1 = nn_pure()

    moded1.build(input_size, output_size)
    moded1.train(num_epochs, x_train, y_train)
    # moded1.plot()
