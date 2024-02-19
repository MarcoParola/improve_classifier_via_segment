# method to perform average on different length arrays
# the input is an array of the arrays with want to compute the average on
def custom_array_average(arr):
    # save initial lengths
    lengths = []
    for row in arr:
        lengths.append(len(row))
    # compute max_length
    max_length = max(lengths)
    # padding
    for k in range(len(arr)):
        while max_length != len(arr[k]):
            arr[k] = np.append(arr[k], arr[k][-1])

    number_of_arrays = len(arr)
    end_counter = number_of_arrays

    result = np.array([])

    for i in range(max_length):
        for f in range(number_of_arrays):
            if i == lengths[f]:
                end_counter = end_counter-1
        sum = 0
        for m in range(len(arr)):
            sum = sum+arr[m][i]

        result = np.append(result, sum/end_counter)

    return result

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt



# models = {model_id : { experiment_id: {list_of_versions}}
# model_id = 1 -> SqueezeNet
# model_id = 2 -> ConvNext
# model_id = 3 -> Vit
# model_id = 4 -> Swin
# exp_id = 1 -> Experiment 1
# exp_id = 3 -> Experiment 3

models = { 1: {1: {225, 286, 267, 247, 380, 381, 263, 240, 282, 278}, 3: {348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 12, 359}},
          2: {1: {226, 287, 268, 248, 384, 385, 264, 241, 283, 279}, 3: {360, 40, 219, 212, 29, 35, 20, 25, 17, 9, 13, 26}},
          3: {1: {265, 284, 387, 388, 245, 389, 261, 238, 280, 276}, 3: {24, 34, 19, 361, 28, 362, 363, 364, 16, 209, 365, 366, 11}},
          4: {1: {224, 285, 266, 391, 392, 246, 393, 262, 239, 277, 281}, 3: {191, 367, 193, 194, 195, 201, 202, 208, 203, 204, 41, 368, 42, 369, 18, 192, 15, 370, 371, 372, 373, 374, 375, 10, 27, 376, 23, 14, 377}} }

#per ora tralasciamo la loss_train e consideriamo solo la loss_val nei plot
for model_id in models:
    for experiment_id in models[model_id]:
        arr_train = []
        arr_val = []
        for version in models[model_id][experiment_id]:
            # Specify the path to your TensorBoard logs directory
            logs_dir = f"./logs/oral/version_{version}/"
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

            arr_train.append(train_loss_np)
            arr_val.append(val_loss_np)
        #padding
        lengths = []
        for row in arr_val:
            lengths.append(len(row))
        # compute max_length
        max_length = max(lengths)
        for k in range(len(arr_val)):
            while max_length != len(arr_val[k]):
                arr_val[k] = np.append(arr_val[k], arr_val[k][-1])

        # inside result_val there is the average of all the curves of all the versions for
        # a certain experiment_id for a certain model_id
        #result_train = custom_array_average(arr_train)
        #result_val = custom_array_average(arr_val)
        mean_val = np.mean(arr_val, axis=0)
        std_dev_y = np.std(arr_val, axis=0)

        x_values = np.linspace(0, len(mean_val), len(mean_val))

        upper_confidence = mean_val + 1.645 * std_dev_y
        lower_confidence = mean_val - 1.645 * std_dev_y

        if experiment_id == 1:
            plt.plot(x_values, mean_val, linewidth=2, label=f'Std. classification')
        elif experiment_id == 3:
            plt.plot(x_values, mean_val, linewidth=2, label=f'CEntIoU')
        plt.fill_between(x_values, upper_confidence, lower_confidence, alpha=0.2)

    plt.ylim(0.05, 1.35)
    plt.xlabel('Epochs', fontsize=25)
    plt.ylabel('Loss', fontsize=25)
    plt.rcParams.update({'font.size': 25})
    plt.legend(prop={'size': 25})
    #plt.legend()
    plt.savefig(f'loss_val_model_{model_id}.pdf')
    plt.clf()



