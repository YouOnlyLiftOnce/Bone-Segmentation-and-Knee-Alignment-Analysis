import torch


def pred2mask(pred):
    '''
    :param pred: prediction from model with size [batch=1, channel=3, h, w]
    :return: an rgb masks
    '''
    pred = torch.argmax(pred, 1).squeeze(0)

    h, w = pred.shape

    pred_red = torch.zeros((h, w))
    pred_green = torch.zeros((h, w))
    pred_blue = torch.zeros((h, w))

    pred_red[pred == 0] = 1.0
    pred_green[pred == 1] = 1.0
    pred_blue[pred == 2] = 1.0

    pred_red = pred_red.unsqueeze(2)
    pred_green = pred_green.unsqueeze(2)
    pred_blue = pred_blue.unsqueeze(2)

    pred_mask = torch.cat((pred_red, pred_green), 2)
    pred_mask = torch.cat((pred_mask, pred_blue), 2)

    return pred_mask
