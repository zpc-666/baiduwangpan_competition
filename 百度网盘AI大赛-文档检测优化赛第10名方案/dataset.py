import os, cv2
import numpy as np
import paddle

def get_corner(positions, corner_flag, w, h):
    # corner_flag 1:top_left 2:top_right 3:bottom_right 4:bottom_left
    if corner_flag == 1:
        target_pos = [0,0]
    elif corner_flag == 2 :
        target_pos = [w,0]
    elif corner_flag == 3 :
        target_pos = [w,h]
    elif corner_flag == 4 :
        target_pos = [0,h]

    min_dis = h**2+w**2
    best_x = 0
    best_y = 0
    for pos in positions:
        now_dis = (pos[0]-target_pos[0])**2+(pos[1]-target_pos[1])**2
        if now_dis<min_dis:
            min_dis = now_dis
            corner_x = pos[0]/w
            corner_y = pos[1]/h
    
    return corner_x, corner_y


def create_train_val_data(train_imgs_dir, train_txt, train_ratio=0.8):

    data_info = []
    with open(train_txt,'r') as f:
        for line in f:
            line = line.strip().split(',')
            image_name = line[0]
            img = cv2.imread(os.path.join(train_imgs_dir, image_name+'.jpg'))
            h, w, c = img.shape

            positions = []
            for i in range(1,len(line),2):
                positions.append([float(line[i]), float(line[i+1])])
            label = []
            for i in range(4):
                corner_x, corner_y = get_corner(positions, i+1, w, h)
                label.append(corner_x)
                label.append(corner_y)

            data_info.append((image_name+'.jpg', label, (w, h)))

    train_len = round(len(data_info)*train_ratio)
    train_info = data_info[:train_len]
    if train_ratio<1.:
        val_info = data_info[train_len:]
    else:
        val_info = []
    print("total data num: {}, train num: {}, val_num: {}".format(len(data_info), len(train_info), len(val_info)))
    return train_info, val_info


class MyDateset(paddle.io.Dataset):
    def __init__(self, imgs_root, data_info, data_transform=None):
        super(MyDateset, self).__init__()

        self.imgs_root = imgs_root
        self.data_transform = data_transform
        self.data_info = data_info

    def __getitem__(self, index):

        data_info = self.data_info[index]
        img = cv2.imread(os.path.join(self.imgs_root, data_info[0]))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 对图片进行resize
        img = paddle.vision.transforms.resize(img, (512,512), interpolation='bilinear')


        ######
        if self.data_transform is not None:
            if np.random.rand()<1/3:
                img = paddle.vision.transforms.adjust_brightness(img, np.random.rand()*2)
            else:
                if np.random.rand()<1/2:
                    img = paddle.vision.transforms.adjust_contrast(img, np.random.rand()*2)
                else:
                    img = paddle.vision.transforms.adjust_hue(img, np.random.rand()-0.5)

        ######
        #if self.data_transform is not None:
        #    img = self.data_transform(img)

        img = img.transpose((2,0,1))
        img = img/255

        #c, h, w = img.shape
        #ori_w, ori_h = data_info[2]
        #w_f, h_f = w/ori_w, h/ori_h
        #scale_factor = paddle.to_tensor([w_f, h_f, w_f, h_f, w_f, h_f, w_f, h_f]).astype('float32')
        label = data_info[1]
        img = paddle.to_tensor(img).astype('float32')
        label = paddle.to_tensor(label).astype('float32')

        return img, label#*scale_factor

    def __len__(self):
        return len(self.data_info)

def load_data(train_imgs_dir, train_txt, train_ratio=0.8, batch_size=16):

    train_info, val_info = create_train_val_data(train_imgs_dir, train_txt, train_ratio=train_ratio)
    train_dataset = MyDateset(imgs_root=train_imgs_dir, data_info=train_info, data_transform="aug")

    train_loader = paddle.io.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False)

    if train_ratio<1.:
        val_dataset = MyDateset(imgs_root=train_imgs_dir, data_info=val_info, data_transform=None)
        val_loader = paddle.io.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False)
    else:
        val_loader = None

    return train_loader, val_loader
