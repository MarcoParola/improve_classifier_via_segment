import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Specify the path to your TensorBoard logs directory
logs_dir = "./logs/oral/version_147/"

# Load TensorBoard logs
event_acc = EventAccumulator(logs_dir)
event_acc.Reload()

# Extract scalar values (including losses) from TensorBoard logs
train_loss_epoch = event_acc.Scalars('train_loss_epoch')  
train_loss = [scalar.value for scalar in train_loss_epoch]
train_loss_np = np.array(train_loss)

val_loss_epoch = event_acc.Scalars('val_loss_epoch')  
val_loss = [scalar.value for scalar in val_loss_epoch]
val_loss_np = np.array(val_loss)

# plot the losses
import matplotlib.pyplot as plt
plt.plot(train_loss_np, label='train_loss')
plt.plot(val_loss_np, label='val_loss')
plt.title('Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
