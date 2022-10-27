from tqdm import tqdm
from utils.metrics import dice_loss, dice_coefficient, pixel_accuracy, mIOU
from utils.focal_loss import FocalLoss
import os
import torch
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criteria_ce = torch.nn.CrossEntropyLoss()
criteria_focal = FocalLoss(class_num=3)


def train(args, model, optimizer, dataloaders, scheduler):
    trainloader, valloader = dataloaders

    loss = 0.0
    best_dice = 0.0
    best_dice_epoch = 0
    best_iou = 0.0

    # training
    for epoch in range(args.epochs):
        model.train()
        loop = tqdm(trainloader, leave=True, total=len(trainloader))
        for i, (imgs, masks) in enumerate(loop):
            imgs, masks = imgs.to(device), masks.to(device)
            pred = model(imgs)
            # convert pred to prediction mask to calculate the dice loss
            # the element value is now in range [0, 1]
            pred_mask = torch.softmax(pred, dim=1)
            dc_loss = dice_loss(pred_mask, masks)
            # convert mask to single channel [B,H,W] for calculate cross entropy loss
            # with element value 0:femur,1:tibia,2:background
            masks_cross_entropy = masks.argmax(1)
            ce_loss = criteria_ce(pred, masks_cross_entropy)
            masks_focal = masks.argmax(1).unsqueeze(0)
            f_loss = criteria_focal(pred, masks_focal)
            loss = dc_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update tqdm loop
            loop.set_description(f"Train Epoch[{epoch}/{args.epochs}]")
            loop.set_postfix(total_loss=loss.item(), cross_entropy_loss=ce_loss.item(), dice_loss=dc_loss.item())

        # save model every 50 epochs
        # if (epoch+1) % 50 == 0:
        #     path_model = "./{}_model_epoch_{}.pth".format(args.exp_id,epoch)
        #     torch.save(model, path_model)

        # validation at the end of each epoch
        if epoch % 1 == 0:
            test_loss, dice, iou = evaluate(model, valloader)
            scheduler.step(dice)
            # tensorboard visualization
            writer.add_scalars("Metrics_{}".format(args.exp_id), {"train loss": loss,
                                                                  "validation loss": test_loss,
                                                                  "dice coefficient": dice,
                                                                  "mIoU": iou}, epoch)

            if dice > best_dice:
                # update best accuracy
                best_dice = dice
                best_dice_epoch = epoch
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch
                }
                path_checkpoint = "./{}_checkpoint.pth".format(args.exp_id)
                torch.save(checkpoint, path_checkpoint)
                # save the model
                path_model = "./{}_model.pth".format(args.exp_id)
                torch.save(model, path_model)

            if iou > best_iou:
                best_iou = iou

                print('\n new best model saved at epoch: {}'.format(epoch))
    print('\n -------------------------------------------------')
    print('\n best dice coefficient achieved: {:.3f} at epoch {}'.format(best_dice, best_dice_epoch))
    # print('\n best accuracy achieved: {:.3f}'.format(best_acc))
    print('\n best mIoU achieved: {:.3f}'.format(best_iou))
    writer.flush()
    writer.close()


def evaluate(model, testloader):
    total_count = len(testloader)
    total_loss = 0.0
    total_dice = 0.0
    total_miou = 0.0
    loop = tqdm(testloader, leave=True, total=total_count)
    model.eval()
    for i, data in enumerate(loop):
        imgs, masks = data
        imgs, masks = imgs.to(device), masks.to(device)

        total_count += masks.size(0)
        with torch.no_grad():
            pred = model(imgs)
            # convert pred to prediction mask to calculate the dice loss
            pred_mask = torch.softmax(pred, dim=1)
            dc_loss = dice_loss(pred_mask, masks)
            # convert mask to single channel [B,H,W] for calculate cross entropy loss
            # with value 0:femur,1:tibia,2:background
            masks_cross_entropy = masks.argmax(1)
            ce_loss = criteria_ce(pred, masks_cross_entropy)

            loss = dc_loss

            total_loss += loss.item()
            dice = dice_coefficient(pred_mask, masks)
            total_dice += dice
            iou = mIOU(pred_mask, masks)
            total_miou += iou

            loop.set_description(f"Validation")
            loop.set_postfix(total_loss=total_loss, dice_coefficient=dice.item(), mIOU=iou.item())

    average_loss = total_loss / len(testloader)
    average_dice = total_dice / len(testloader)
    average_miou = total_miou / len(testloader)

    return average_loss, average_dice, average_miou


def resume(args, model, optimizer):
    checkpoint_path = './{}_checkpoint.pth'.format(args.exp_id)
    assert os.path.exists(checkpoint_path), ('checkpoint do not exits for %s' % checkpoint_path)

    ### load the model and the optimizer --------------------------------
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    ### -----------------------------------------------------------------

    print('Resume completed for the model\n')

    return model, optimizer


writer.close()
