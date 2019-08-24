import torch


def nms_cpu(dets, scores, thresh, top_k=200):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort(descending=True)
    order = order[:top_k]
    keep = []
    while order.size(0) > 0:
        i = order[0].item()
        keep.append(i)
        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])

        zero = torch.tensor(0.0).to(dets.device)
        w = torch.max(zero, xx2 - xx1)
        h = torch.max(zero, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-16)

        inds = torch.where(iou <= thresh)[0]
        order = order[inds + 1]

    return keep


if __name__ == "__main__":
    import torch
    dets = torch.tensor([[1.0, 1, 4, 4],
            [2, 2, 5, 5],
            [10, 10, 20, 20],
            [11, 11, 21, 21]]).cuda()
    scores = torch.tensor([0.3, 0.2, 0.7, 0.9]).cuda()
    keep = nms_cpu(dets, scores, 0.5, 200)
    pass
