import torch
import torch.nn as nn
import sys
sys.path.append('/home/twsf/work/DSSD')
from utils.box_utils import jaccard


class MultiBoxEval(nn.Module):
    def __init__(self, input_size, iou_thresh):
        super(MultiBoxEval, self).__init__()
        self.width = input_size
        self.height = input_size
        self.iou_thresh = iou_thresh

    def forward(self, output, labels):
        """
        pred is result of after used nms, like:
        ([pred_num, 6],
            ......
         [pred_num, 6]). 6 shape like [box + socre + cls]
        targets is real box, like:
        ([box_nums, 5],
            ......
         [box_nums, 5]). 5 shape like [box + cls]
        """

        num = len(output)
        stats = []
        for id in range(num):
            targets = labels[id]
            preds = output[id, output[id, :, 4].gt(0)]
            num_gt = len(targets)  # number of target
            tcls = targets[:, 4].tolist() if num_gt else []  # target class

            # predict is none
            if preds is None:
                if num_gt:
                    stats.append(([], torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Assign all predictions as incorrect
            correct = [0] * len(preds)
            if num_gt:
                detected = []
                tcls_tensor = targets[:, 4]

                # target boxes
                tboxes = targets[:, :4]
                tboxes[:, [0, 2]] *= self.width
                tboxes[:, [1, 3]] *= self.height
                preds[:, [0, 2]] *= self.width
                preds[:, [1, 3]] *= self.height

                for ii, pred in enumerate(preds):
                    pbox = pred[:4].unsqueeze(0)
                    pcls = pred[5]

                    # Break if all targets already located in image
                    if len(detected) == num_gt:
                        break

                    # Continue if predicted class not among image classes
                    if pcls.item() not in tcls:
                        continue

                    # Best iou, index between pred and targets
                    m = (pcls == tcls_tensor).nonzero().view(-1)
                    iou, bi = jaccard(pbox, tboxes[m]).max(1)

                    # If iou > threshold and class is correct mark as correct
                    if iou > self.iou_thresh and m[bi] not in detected:
                        correct[ii] = 1
                        detected.append(m[bi])

            # (correct, pconf, pcls, tcls)
            stats.append((correct, preds[:, 4].cpu(), preds[:, 5].cpu(), tcls))
        
        return stats


if __name__ == "__main__":
    import numpy as np
    evaluator = MultiBoxEval(1, 0.5)
    temp = torch.tensor([[1, 1, 4, 4, 0.4, 1],
                         [4, 4, 20, 20, 0.9, 1],
                         [1, 1, 4, 4, 0.7, 2],
                         [1, 1, 5, 5, 0.8, 2]]).cuda()

    target = torch.tensor([[1, 1, 20, 20, 1.0],
                           [2, 2, 5, 5, 2.0]]).cuda()

    output = [temp, temp]
    targets = [target, target]
    res = []
    res += evaluator(output, targets, 2)
    res += evaluator(output, targets, 1)
    stats = [np.concatenate(x, 0) for x in list(zip(*res))]
    pass
