from PIL import Image
import os

basedir = '/users/nivl/data/autoencoder/hsp'

paths = [x for x in os.listdir(basedir) if x.startswith('20200403')]

for dir in paths:
    dir_path = '{}/{}'.format(basedir, dir)
    if '{}/{}_res.png'.format(dir_path, dir) in os.listdir(dir_path):
        continue
    blank_image = Image.new(mode="RGB", size=(1265, 1030), color=(255, 255, 255))

    try:
        image_cost = Image.open(dir_path + '/cost.png')
        image_sp = Image.open(dir_path + '/hsp.png')
        image_loss = Image.open(dir_path + '/loss.png')
        image_map = Image.open(dir_path + '/models/epoch_100.png')
    except FileNotFoundError:
        continue
    image_map.thumbnail((625, 550), Image.ANTIALIAS)

    blank_image.paste(image_cost, (0, 0))
    blank_image.paste(image_loss, (625, 0))
    blank_image.paste(image_sp, (0, 550))
    blank_image.paste(image_map, (640, 480))

    param_f = dir_path + '/parameters.txt'
    with open(param_f) as f:
        lines = [line.rstrip() for line in f]
    spline = [item for item in lines if item.startswith('tg_hsp')][0]
    maskingline = [item for item in lines if item.startswith('masking')][0]
    sp = spline[9:]
    sp = sp.replace('.', '')
    mask = maskingline[16:]
    fname = 'hsp{}_pct{}'.format(sp, mask)

    blank_image.save('{}/{}_epoch100.png'.format(dir_path, fname))

