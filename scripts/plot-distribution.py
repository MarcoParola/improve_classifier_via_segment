import argparse
import json
import numpy as np
from matplotlib.patches import Ellipse
from matplotlib import pyplot as plt

colors = ['tab:orange', 'tab:green', 'tab:blue']
x_plot_size = 3900
y_plot_size = 2900
percentiles = [1, 25, 50, 75, 90]
font_size_labels = 14
font_size = 15


def plot_distribution(coco_data):
    centroids = {}
    widths = {}
    heights = {}

    # Extract centroids and bounding box dimensions
    for annotation in coco_data["annotations"]:
        category_id = annotation["category_id"]
        bbox = annotation["bbox"]
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]

        if category_id not in centroids:
            centroids[category_id] = {"x": [], "y": []}
            widths[category_id] = []
            heights[category_id] = []

        centroids[category_id]["x"].append(x)
        centroids[category_id]["y"].append(y)
        widths[category_id].append(w)
        heights[category_id].append(h)

    # scatter plot for each class in the first subplot (x-y bbox coordinate distribution)
    fig, ax = plt.subplots()
    for category_id, data in centroids.items():
        x_values = data["x"]
        y_values = data["y"]
        class_name = coco_data["categories"][category_id - 1]["name"]
        ax.scatter(np.mean(x_values), np.mean(y_values), label=class_name, s=40)
        ax.scatter(x_values, y_values, c=colors[category_id-1], alpha=0.15)
    
    # set plot limits
    ax.set_xlim(0, x_plot_size)
    ax.set_ylim(0, y_plot_size)
    plt.xticks(np.arange(0, x_plot_size, 500), fontsize=font_size_labels)
    plt.yticks(np.arange(0, y_plot_size, 500), fontsize=font_size_labels)
    ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    ax.set_xlabel("X Coordinate", fontsize=font_size)
    ax.set_ylabel("Y Coordinate", fontsize=font_size)
    ax.legend(fontsize=16)
    plt.show()  


    # plot ellipses for each class in the second subplot (widths and heights distribution)
    fig, ax = plt.subplots()
    for category_id, data in centroids.items():
        x_values = widths[category_id]
        y_values = heights[category_id]
        class_name = coco_data["categories"][category_id - 1]["name"]

        # Plot ellipses for percentiles
        for i, percentile in enumerate(percentiles):
            width_percentile = np.percentile(x_values, percentiles)[i]
            height_percentile = np.percentile(y_values, percentiles)[i]
            ellipse = Ellipse(
                (np.mean(x_values), np.mean(y_values)),
                width_percentile,
                height_percentile,
                alpha=0.1 + 0.1 / (i+1),
                color = colors[category_id-1])
            ax.add_patch(ellipse)
        
        ax.scatter(np.mean(x_values), np.mean(y_values), label=class_name)

    ax.set_xlabel("W size", fontsize=font_size)
    ax.set_ylabel("H size", fontsize=font_size)
    ax.legend(fontsize=16)
    ax.set_xlim(0, x_plot_size)
    ax.set_ylim(0, y_plot_size)
    plt.xticks(np.arange(0, x_plot_size, 500), fontsize=font_size_labels)
    plt.yticks(np.arange(0, y_plot_size, 500), fontsize=font_size_labels)
    ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    dataset = json.load(open(args.dataset, "r"))
    plot_distribution(dataset)