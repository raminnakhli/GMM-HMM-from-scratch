
def CCR(test_labels, classifier_labels):
    CCR_Val = 0
    for label_idx in range(len(test_labels)):
        if test_labels[label_idx] == classifier_labels[label_idx]:
            CCR_Val += 1

    return 100.0 * CCR_Val / len(test_labels)
