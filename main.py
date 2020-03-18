from data_loader import *
from gmm_hmm import *
from plot_conf_mat import plot_confusion_matrix
import matplotlib.pyplot as plt
import argparse

debug = False

parser = argparse.ArgumentParser(description='This is a script used to run the tests')
parser.add_argument('-ht', '--hyperparameter-test', action='store_true', help='test hyper parameters')
parser.add_argument('-stpr', '--stop-rate', default=0.1, help='stop rate for EM')
parser.add_argument('-stc', '--state-count', default=2, help='states count')
parser.add_argument('-mc', '--mixture-count', default=2, help='mixture element count')
parser.add_argument('-blk', '--belkin', action='store_true', help='use belkin')
parser.add_argument('-vtb', '--viterbi', action='store_true', help='viterbi')
parser.add_argument('-vft', '--viterbi-forward-test', action='store_true', help='forward vs viterbi test')
args = parser.parse_args()

print('-------------------------- config ----------------------------------')
print('EM stop rate: {}, state count: {}, mixture count: {}'.
      format(args.stop_rate, args.state_count, args.mixture_count))
print('hyper param test: {}, blkin: {},  viterbi: {}, viterbi vs forward test: {}'.
      format(args.hyperparameter_test, args.belkin, args.viterbi, args.viterbi_forward_test))
print('--------------------------------------------------------------------')
print('\n')

a_train, e_train, i_train, o_train, u_train, a_test, e_test, i_test, o_test, u_test = dataset_loader()
        
def test(dataset_data, dataset_label):

    predicted_label_list = list()

    if viterbi:
        a_out, _ = a_gmm.viterbi(dataset_data)
        e_out, _ = e_gmm.viterbi(dataset_data)
        i_out, _ = i_gmm.viterbi(dataset_data)
        o_out, _ = o_gmm.viterbi(dataset_data)
        u_out, _ = u_gmm.viterbi(dataset_data)
    else:
        a_out = a_gmm.likelihood(dataset_data)
        e_out = e_gmm.likelihood(dataset_data)
        i_out = i_gmm.likelihood(dataset_data)
        o_out = o_gmm.likelihood(dataset_data)
        u_out = u_gmm.likelihood(dataset_data)

    if debug:
        print('******************')
        print('label a : ', a_out)
        print('label e : ', e_out)
        print('label i : ', i_out)
        print('label o : ', o_out)
        print('label u : ', u_out)

    for data_index in range(len(dataset_data)):
        predicted_label = np.argmax([a_out[data_index], e_out[data_index],
                                     i_out[data_index], o_out[data_index],
                                     u_out[data_index]])

        predicted_label_list.append(predicted_label)

    plot_confusion_matrix(dataset_label, predicted_label_list, range(10),
                          title='viterbi={}, belkin={}, N={}, M={}'.
                          format(args.viterbi, args.belkin, N, M)
                          )


try_count = 1 if not args.hyperparameter_test else 5

for i in range(try_count):

    if args.hyperparameter_test:
        N = np.random.randint(2, 5)
        M = np.random.randint(1, 5)
    else:
        N = int(args.state_count)
        M = int(args.mixture_count)

    stop_diff = float(args.stop_rate)

    viterbi = args.viterbi
    belkin = args.belkin

    viterbi_list = [args.viterbi, not args.viterbi] if args.viterbi_forward_test else [args.viterbi]

    a_gmm = GMM_HMM('a_gmm', N, M)
    e_gmm = GMM_HMM('e_gmm', N, M)
    i_gmm = GMM_HMM('i_gmm', N, M)
    o_gmm = GMM_HMM('o_gmm', N, M)
    u_gmm = GMM_HMM('u_gmm', N, M)

    train_data_list = [a_train, e_train, i_train, o_train, u_train]
    train_label_list = [label for label, data in enumerate(train_data_list) for _ in range(len(data))]
    train_data_list = a_train + e_train + i_train + o_train + u_train

    test_data_list = [a_test, e_test, i_test, o_test, u_test]
    test_label_list = [label for label, data in enumerate(test_data_list) for _ in range(len(data))]
    test_data_list = a_test + e_test + i_test + o_test + u_test

    a_gmm.train(a_train, stop_diff, belkin)
    e_gmm.train(e_train, stop_diff, belkin)
    i_gmm.train(i_train, stop_diff, belkin)
    o_gmm.train(o_train, stop_diff, belkin)
    u_gmm.train(u_train, stop_diff, belkin)

    print('\n')

    for viterbi in viterbi_list:

        print('M = {}, N = {}, Viterbi  = {}, Belkin = {}'.format(M, N, viterbi, belkin))

        test(train_data_list, train_label_list)
        test(test_data_list, test_label_list)

plt.show()
