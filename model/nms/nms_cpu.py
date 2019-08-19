import numpy as np


def nms_cpu(dets, scores, thresh, top_k=200):
    dets = dets.numpy()
    scores = scores.numpy()
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    order = order[:top_k]
    keep = []
    while order.size > 0:
        i = order.item(0)
        keep.append(i)
        xx1 = np.maximum(x1[i], y1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-16)

        inds = np.where(iou <= thresh)[0]
        order = order[inds + 1]

    return keep


if __name__ == "__main__":
    import torch
    dets = torch.tensor([[1, 1, 4, 4],
            [2, 2, 5, 5],
            [10, 10, 20, 20],
            [11, 11, 21, 21]])
    scores = torch.tensor([0.3, 0.2, 0.7, 0.9])
    keep = nms_cpu(dets, scores, 0.5, 200)

