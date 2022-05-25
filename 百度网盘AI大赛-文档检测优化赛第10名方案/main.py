import paddle, os, yaml
from visualdl import LogWriter
from utils import train_one_epoch, evaluate, set_seed_paddle
from model import MyNet
from dataset import load_data
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Weight Decay Experiments')

parser.add_argument('--images_root',
                    dest='images_root',
                    help='images root',
                    default="../train_datasets_document_detection_0411/images",
                    type=str)
parser.add_argument('--data_info',
                    dest='data_info',
                    help='data information txt',
                    default="../train_datasets_document_detection_0411/data_info.txt",
                    type=str)
parser.add_argument('--save_dir',
                    dest='save_dir',
                    help='saving data dir',
                    default='output',
                    type=str)
parser.add_argument('--save_interval',
                    dest='save_interval',
                    help='save interval',
                    default=5,
                    type=int)
parser.add_argument('--print_interval',
                    dest='print_interval',
                    help='print interval',
                    default=40,
                    type=int)
parser.add_argument('--batch_size',
                    dest='batch_size',
                    help='training batch size',
                    default=16,
                    type=int)
parser.add_argument('--optimizer',
                    dest='optimizer',
                    help='optimizer',
                    default='adam',
                    type=str)
parser.add_argument('--learning_rate',
                    dest='learning_rate',
                    help='learning rate',
                    default=0.0001,
                    type=float)
parser.add_argument('--momentum',
                    dest='momentum',
                    help='momentum',
                    default=0.9,
                    type=float)
parser.add_argument('--weight_decay',
                    dest='weight_decay',
                    help='weight decay',
                    default=0.,
                    type=float)
parser.add_argument('--epochs',
                    dest='epochs',
                    help='epochs',
                    default=20,
                    type=int)
parser.add_argument('--schedule',
                    dest='schedule',
                    help='Decrease learning rate',
                    default=None,
                    type=int,
                    nargs='+')
parser.add_argument('--gamma',
                    dest='gamma',
                    help='gamma',
                    default=0.1,
                    type=float)
parser.add_argument('--seed',
                    dest='seed',
                    help='seed',
                    default=2021,
                    type=int)
parser.add_argument('--use_schedule',
                    dest='use_schedule',
                    help='whether use lr schedule',
                    default=False,
                    type=str2bool)
parser.add_argument('--train_ratio',
                    dest='train_ratio',
                    help='train data ratio',
                    default=1.0,
                    type=float)

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    writer = LogWriter(args.save_dir)
    with open(os.path.join(args.save_dir, "config.txt"), encoding='utf-8', mode='w') as f:
        yaml.dump(data=vars(args), stream=f, allow_unicode=True)
    if args.seed>0:
        set_seed_paddle(args.seed)

    model = MyNet()
    train_loader, val_loader = load_data(train_imgs_dir=args.images_root, train_txt=args.data_info, train_ratio=args.train_ratio, batch_size=args.batch_size)


    if args.optimizer=="adam":
        if not args.use_schedule:
            scheduler = args.learning_rate
        else:
            #scheduler = paddle.optimizer.lr.LinearWarmup(learning_rate=args.learning_rate, warmup_steps=int(args.epochs/10), start_lr=3.0e-6, end_lr=args.learning_rate)
            scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=args.learning_rate, T_max=args.epochs, eta_min=args.learning_rate//100)
        opt = paddle.optimizer.Adam(learning_rate=scheduler, parameters=model.parameters(), weight_decay=args.weight_decay)
    elif args.optimizer=="sgd":
        if not args.use_schedule:
            scheduler = args.learning_rate
        else:
            scheduler = paddle.optimizer.lr.MultiStepDecay(learning_rate=args.learning_rate, milestones=args.schedule, gamma=args.gamma)
        opt = paddle.optimizer.Momentum(learning_rate=scheduler,
                                    momentum=args.momentum,
                                    parameters=model.parameters(),
                                    weight_decay=args.weight_decay)
        

    criterion = paddle.nn.L1Loss()#MSELoss()


    now_step=0
    best_miou = 0
    for epoch in range(0, args.epochs):
        
        now_step = train_one_epoch(model, train_loader, criterion, opt, writer, now_step, epoch, print_interval=args.print_interval)

        writer.add_scalar('train/lr', opt.get_lr(), epoch)
        if args.use_schedule:
            scheduler.step()

        if args.train_ratio<1.:
            with paddle.no_grad():
                val_loss, val_miou = evaluate(val_loader, model, criterion, print_interval=args.print_interval)
                model.train()
                writer.add_scalar('val/loss', val_loss, epoch)
                writer.add_scalar('val/miou', val_miou, epoch)
            if val_miou>best_miou:
                best_miou = val_miou
                paddle.save(model.state_dict(), os.path.join(args.save_dir, 'model_best.pdparams'))
        if epoch%args.save_interval==0 or (epoch+1)==args.epochs:
            paddle.save(model.state_dict(), os.path.join(args.save_dir, 'model_{}.pdparams'.format(str(epoch))))
            paddle.save(opt.state_dict(), os.path.join(args.save_dir, 'opt_{}.pdopt'.format(str(epoch))))

    print("best miou:", best_miou)