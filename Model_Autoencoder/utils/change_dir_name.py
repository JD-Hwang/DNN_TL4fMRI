import os

basedir = '/users/nivl/data/autoencoder/sparse/0405'

for dir in os.listdir(basedir):

    if not dir.startswith('2020'):
        continue
    dir_path = '{}/{}'.format(basedir, dir)
    param_f = dir_path + '/parameters.txt'
    with open(param_f) as f:
        lines = [line.rstrip() for line in f]
    rholine = [item for item in lines if item.startswith('rho')][0]
    maskingline = [item for item in lines if item.startswith('masking')][0]
    rho = rholine[6:]
    rho = rho.replace('.', '')
    mask = maskingline[16:]
    newdirname = '{}/rho{}_pct{}'.format(basedir, rho, mask)
    dir_cnt = 1
    while True:
        if os.path.isdir(newdirname):
            if dir_cnt == 1:
                newdirname = newdirname + '-' + str(dir_cnt)
            else:
                newdirname = newdirname[:-1] + str(dir_cnt)

            dir_cnt += 1
        else:
            break
    os.rename(dir_path, newdirname)
