import time, random, paddle, os
import numpy as np
from PIL import Image, ImageDraw

def set_seed_paddle(seed=1024):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    paddle.seed(seed)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def cal_miou(bs, h, w, pred, label, mode='mean'):

    #h: 1, 3, 5, 7 , w: 0, 2, 4, 6
    miou = 0
    for i in range(bs):
        mask_pre = Image.new('L', (w, h), 0)
        #print(np.array(mask))
        draw = ImageDraw.Draw(mask_pre, 'L')
        corner_xy = [(pred[i, j]*w, pred[i, j+1]*h) for j in range(0, 8, 2)]
        draw.polygon(corner_xy, fill=1)
        #print(np.array(mask))
        mask_pre = np.array(mask_pre, dtype=np.float32)

        mask_gt = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask_gt, 'L')
        corner_xy = [(label[i, j]*w, label[i, j+1]*h) for j in range(0, 8, 2)]
        draw.polygon(corner_xy, fill=1)
        mask_gt = np.array(mask_gt, dtype=np.float32)

        mul = (mask_gt*mask_pre).sum()
        iou = mul/(mask_gt.sum()+mask_pre.sum()-mul)
        miou = miou+iou

    if mode=="mean":
        return miou/bs
    elif mode=="sum":
        return miou



def evaluate(val_loader, model, criterion, print_interval=100):
    model.eval()
    losses = AverageMeter()
    miou_c = AverageMeter()
    batch_time = AverageMeter()

    for step, data in enumerate(val_loader):

        img, label = data
        end = time.time()
        pre = model(img)
        batch_time.update(time.time() - end)
        loss = criterion(pre, label)
        bs, c, h, w = img.shape
        miou = cal_miou(bs, h, w, pre.numpy(), label.numpy())

        losses.update(loss.item(), img.shape[0])
        miou_c.update(miou, img.shape[0])
        
        if step%print_interval==0:
            print('Test: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'MIOU {miou.val:.3f} ({miou.avg:.3f})'.format(
                step,
                len(val_loader),
                batch_time=batch_time,
                loss=losses,
                miou=miou_c))

    print(' * MIOU {miou.avg:.3f} Time {batch_time.avg:.3f}'
            .format(miou=miou_c, batch_time=batch_time))

    return losses.avg, miou_c.avg


def train_one_epoch(model, train_loader, criterion, opt, writer, now_step, epoch, print_interval=100):
    model.train()
    losses = AverageMeter()
    miou_c = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    for step, data in enumerate(train_loader):

        img, label = data
        data_time.update(time.time() - end)
        pre = model(img)
        loss = criterion(pre, label)

        loss.backward()
        opt.step()
        opt.clear_gradients()

        bs, c, h, w = img.shape
        miou = cal_miou(bs, h, w, pre.numpy(), label.numpy())
        losses.update(loss.item(), img.shape[0])
        miou_c.update(miou, img.shape[0])
        batch_time.update(time.time() - end)
        
        if step%print_interval==0:
            writer.add_scalar('train/loss', losses.val, now_step)
            writer.add_scalar('train/miou', miou_c.val, now_step)
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'MIOU {miou.val:.3f} ({miou.avg:.3f})'.format(
                    epoch,
                    step,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    miou=miou_c))

        now_step += 1
        end = time.time()


    return now_step