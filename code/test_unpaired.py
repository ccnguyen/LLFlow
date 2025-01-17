import glob
import tqdm
from natsort import natsort
import argparse
import options.options as option
from models import create_model
import torch
from utils.util import opt_get
import numpy as np
import os
import cv2


def fiFindByWildcard(wildcard):
    return natsort.natsorted(glob.glob(wildcard, recursive=True))


def load_model(conf_path):
    opt = option.parse(conf_path, is_train=False)
    opt['gpu_ids'] = None
    opt = option.dict_to_nonedict(opt)
    model = create_model(opt)

    model_path = opt_get(opt, ['model_path'], None)
    model.load_network(load_path=model_path, network=model.netG)
    return model, opt


def predict(model, lr):
    model.feed_data({"LQ": t(lr)}, need_GT=False)
    model.test()
    visuals = model.get_current_visuals(need_GT=False)
    return visuals.get('rlt', visuals.get('NORMAL'))


def t(array): return torch.Tensor(np.expand_dims(array.transpose([2, 0, 1]), axis=0).astype(np.float32)) / 255


def rgb(t): return (
        np.clip((t[0] if len(t.shape) == 4 else t).detach().cpu().numpy().transpose([1, 2, 0]), 0, 1) * 255).astype(
    np.uint8)


def imread(path):
    return cv2.imread(path)[:, :, [2, 1, 0]]


def imwrite(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img[:, :, [2, 1, 0]])


def imCropCenter(img, size):
    h, w, c = img.shape

    h_start = max(h // 2 - size // 2, 0)
    h_end = min(h_start + size, h)

    w_start = max(w // 2 - size // 2, 0)
    w_end = min(w_start + size, w)

    return img[h_start:h_end, w_start:w_end]


def impad(img, top=0, bottom=0, left=0, right=0, color=255):
    return np.pad(img, [(top, bottom), (left, right), (0, 0)], 'reflect')


def hiseq_color_cv2_img(img):
    (b, g, r) = cv2.split(img)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    result = cv2.merge((bH, gH, rH))
    return result

def decrease_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    v = v / 1.0
    v *= value
    v = v.astype(np.uint8)
    # v[v < value] = 0
    # v[v >= value] -= value
    final_hsv = cv2.merge((h,s,v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
    return img



def auto_padding(img, times=16):
    # img: numpy image with shape H*W*C

    h, w, _ = img.shape
    h1, w1 = (times - h % times) // 2, (times - w % times) // 2
    h2, w2 = (times - h % times) - h1, (times - w % times) - w1
    img = cv2.copyMakeBorder(img, h1, h2, w1, w2, cv2.BORDER_REFLECT)
    return img, [h1, h2, w1, w2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", default="./confs/LOL_smallNet.yml")
    parser.add_argument("-n", "--name", default="unpaired")
    parser.add_argument('--bright', type=float, default=0.6)
    parser.add_argument('--noise', type=float, default=0.05)
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    conf_path = args.opt
    conf = conf_path.split('/')[-1].replace('.yml', '')
    model, opt = load_model(conf_path)
    model.netG = model.netG.cuda()

    opt['dataroot_unpaired'] = f'/home/cindy/PycharmProjects/data/ocr/test/{args.dataset}'
    
    lr_dir = opt['dataroot_unpaired']
    lr_paths = fiFindByWildcard(os.path.join(lr_dir, '*.*'))
    lr_paths = [f for f in lr_paths if '.png' in f]
    print(lr_paths)

    this_dir = os.path.dirname(os.path.realpath(__file__))
    test_dir = os.path.join(this_dir, '..', 'results', f'{args.dataset}_v{args.bright}_n{args.noise}')
    print(f"Out dir: {test_dir}")

    # lr_paths = random.sample(lr_paths, 10)
    # my_list = ['95.png', '231.png', '282.png', '300.png', '466.png', '632.png', '702.png', '710.png', '739.png', '992.png']

    # lr_paths = [f for f in lr_paths if f.split('/')[-1] in my_list]

    for lr_path, idx_test in tqdm.tqdm(zip(lr_paths, range(len(lr_paths)))):
        print(lr_path)
        lr = imread(lr_path)
        lr = decrease_brightness(lr, value=args.bright) / 255.0
        lr = lr + np.random.randn(*lr.shape) * args.noise
        lr = np.clip(lr, 0.0, 1.0)
        lr = (lr * 255).astype(np.uint8)
        raw_shape = lr.shape
        lr, padding_params = auto_padding(lr)
        his = hiseq_color_cv2_img(lr)
        if opt.get("histeq_as_input", False):
            lr = his

        lr_t = t(lr)
        if opt["datasets"]["train"].get("log_low", False):
            lr_t = torch.log(torch.clamp(lr_t + 1e-3, min=1e-3))
        if opt.get("concat_histeq", False):
            his = t(his)
            lr_t = torch.cat([lr_t, his], dim=1)
        with torch.cuda.amp.autocast():
            sr_t = model.get_sr(lq=lr_t.cuda(), heat=None)

        sr = rgb(torch.clamp(sr_t, 0, 1)[:, :, padding_params[0]:sr_t.shape[2] - padding_params[1],
                 padding_params[2]:sr_t.shape[3] - padding_params[3]])
        assert raw_shape == sr.shape
        path_out_sr = os.path.join(test_dir, os.path.basename(lr_path))
        imwrite(path_out_sr, sr)


def format_measurements(meas):
    s_out = []
    for k, v in meas.items():
        v = f"{v:0.2f}" if isinstance(v, float) else v
        s_out.append(f"{k}: {v}")
    str_out = ", ".join(s_out)
    return str_out


if __name__ == "__main__":
    main()
