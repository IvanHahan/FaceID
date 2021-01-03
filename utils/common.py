import os


def make_dir_if_needed(dir_):
    if not os.path.exists(dir_):
        os.makedirs(dir_)


def unite_dicts(*dicts):
    union = {}
    for d in dicts:
        for k, v in d.items():
            union.setdefault(k, [])
            union[k].extend(v)
    return union
