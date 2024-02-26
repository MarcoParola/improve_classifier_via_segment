import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from src.metrics import calculate_intersection_over_salient_region
from src.utils import get_last_version


class SaliencyAwareLoss(torch.nn.Module):
    def __init__(self, weight_loss):
        super(SaliencyAwareLoss, self).__init__()
        self.weight_loss = weight_loss
        self.requires_grad_(True)

    # current_epoch e stage poi vanno tolti
    def forward(self, actual_lbl, predicted_lbl, salient_area, ground_truth_mask, current_epoch, stage):
        predicted_lbl = predicted_lbl.to(torch.float)
        actual_lbl = actual_lbl.to(torch.long)
        # Calculate standard cross-entropy loss
        cross_entropy_loss = F.cross_entropy(predicted_lbl, actual_lbl)
        # Calculate iou
        iou = calculate_intersection_over_salient_region(salient_area, ground_truth_mask)
        # Combine the losses
        loss = self.weight_loss * cross_entropy_loss - (1 - self.weight_loss) * iou
        loss.requires_grad_(True)

        # this is just to log the two loss components on tensorboard
        log_dir = 'logs/oral/' + get_last_version('logs/oral')
        writer = SummaryWriter(log_dir=log_dir)
        if stage == "val":
            writer.add_scalars('val_loss_components', {'val_Cross_entropy_loss': cross_entropy_loss, 'val_iou': iou}, current_epoch)
        elif stage == "train":
            writer.add_scalars('train_loss_components', {'train_Cross_entropy_loss': cross_entropy_loss, 'train_iou': iou}, current_epoch)
        writer.close()

        return loss


if __name__ == '__main__':

    num_classes = 5
    batch_size = 32
    height, width = 30, 30

    actual_labels = torch.randint(low=0, high=num_classes, size=(batch_size,)).float()
    predicted_labels = torch.randint(low=0, high=num_classes, size=(batch_size,)).float()
    salient_area = torch.rand(batch_size, height, width)
    ground_truth_mask = torch.randn(batch_size, height, width)

    loss_fn = SaliencyAwareLoss(weight_loss=0.8)
    loss = loss_fn(actual_labels, predicted_labels, salient_area, ground_truth_mask)


    print(loss)

