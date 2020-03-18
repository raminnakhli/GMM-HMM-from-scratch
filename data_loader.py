import xml.etree.ElementTree as ET
import numpy as np


def xml_loader(file_name):
    tree = ET.parse(file_name)
    root = tree.getroot()

    train_samples = list()
    test_samples = list()

    for i in range(len(root) - 1):
        training_example = root[i + 1]
        # sample = np.zeros((len(training_example), 3))
        sample = np.zeros((len(training_example), 2))

        for j in range(len(training_example)):
            coord = training_example[j]
            x = coord.get('x')
            y = coord.get('y')
            t = coord.get('t')
            # sample[j] = [x, y, t]
            sample[j] = [x, y]

        if i % 2 == 0:
            train_samples.append(sample)
        else:
            test_samples.append(sample)

    # return np.array(train_samples), np.array(test_samples)
    return train_samples, test_samples


def dataset_loader():
    a_train, a_test = xml_loader('data/a.xml')
    e_train, e_test = xml_loader('data/e.xml')
    i_train, i_test = xml_loader('data/i.xml')
    o_train, o_test = xml_loader('data/o.xml')
    u_train, u_test = xml_loader('data/u.xml')

    return a_train, e_train, i_train, o_train, u_train, \
           a_test, e_test, i_test, o_test, u_test
