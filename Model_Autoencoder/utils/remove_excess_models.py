import os

path = '/users/nivl/data/autoencoder/classifier/fixed/1026'

for run in os.listdir(path):
    run_dir = '{}/{}'.format(path, run)
    print(run)
    for cand in os.listdir(run_dir):
        print(cand)
        if not cand.startswith('l1'):
            continue
        cand_dir = '{}/{}/models'.format(run_dir, cand)
        print(cand_dir)
        for f in os.listdir(cand_dir):
            if f == 'model200.pt':
                continue
            full_f = '{}/{}'.format(cand_dir, f)
            os.remove(full_f)
