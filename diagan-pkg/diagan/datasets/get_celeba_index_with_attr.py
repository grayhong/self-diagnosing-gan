from functools import partial
import os
import pandas

def get_celeba_index_with_attr(root, attr_name):
    base_folder = "celeba"
    fn = partial(os.path.join, root, base_folder)
    attr = pandas.read_csv(fn("list_attr_celeba.txt"), delim_whitespace=True, header=1)

    attribute = (attr.values + 1) // 2  # map from {-1, 1} to {0, 1}
    attr_names = list(attr.columns)
    try:
        attr_num = attr_names.index(attr_name)
    except:
        raise ValueError("Invalid attribute name {}.".format(attr_name))

    attr_index = []
    not_attr_index = []
    for i in range(len(attribute)):
        if attribute[i][attr_num]:
            attr_index.append(i)
        else:
            not_attr_index.append(i)

    return attr_index, not_attr_index
