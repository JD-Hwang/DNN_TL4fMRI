from data_utils import get_sbj_task_data, is_subject_valid


retest_sbj = ['103818', '105923', '111312', '114823', '115320', '122317', '125525', '135528', '137128',
              '139839', '144226', '146129', '149337', '149741', '151526', '158035', '169343', '172332',
              '175439', '177746', '185442', '187547', '192439', '194140', '195041', '200109', '200614',
              '204521', '250427', '287248', '433839', '562345', '599671', '601127', '627549', '660951',
              '662551', '783462', '859671', '861456', '877168', '917255']
task_lst = ['WM', 'MOTOR', 'EMOTION', 'RELATIONAL', 'SOCIAL', 'LANGUAGE', 'GAMBLING']
num_samples = [[544], [260], [223, 224], [216], [268], [300], [271]]


for task_idx, task in enumerate(task_lst):
    print(task)
    if task_idx < 2:
        continue
    samples_per_sbj = num_samples[task_idx]
    for idx, sbj in enumerate(retest_sbj):
        if not is_subject_valid(sbj, task, True):
            print(idx)
            continue
        sbj_samples, sbj_labels = get_sbj_task_data(sbj, task)
        if len(sbj_labels) not in samples_per_sbj and task_idx != 5:
            print(idx)
