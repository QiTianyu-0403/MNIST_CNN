import pandas as pd
import argparse
import matplotlib.pyplot as plt

def per2float(per):
    """
    Delete the '%' in acc
    """
    return float(per[:-1])

def GetList(train_consult,test_consult):
    """
    Get the prasers of 'x'&'y'
    """
    x = []
    # x_test = []
    train_loss = []
    train_acc = []
    test_acc = []
    for i in range(len(train_consult)-1):
        if train_consult.iloc[i, 0] + 1 == train_consult.iloc[i+1,0] or i == len(train_consult)-2:
            if train_consult.iloc[i, 0] > test_consult.iloc[len(test_consult)-1, 1]:
                break
            x.append(train_consult.iloc[i, 0])
            train_loss.append(train_consult.iloc[i, 4])
            train_acc.append(per2float(train_consult.iloc[i, 7]))
            test_acc.append(per2float(test_consult.iloc[train_consult.iloc[i, 0]-1, 4]))
    return x, train_loss, train_acc, test_acc

def draw(x, train_loss, train_acc, test_acc, train_loss1, train_acc1, test_acc1, train_loss2, train_acc2, test_acc2, train_loss3, train_acc3, test_acc3, args):

    fig = plt.figure(figsize=(6, 6.5))
    
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    
    ax1.title.set_text('Train Loss of MNIST with CNN')
    ax2.title.set_text('Train Accuracy of MNIST with CNN')
    ax3.title.set_text('Test Accuracy of MNIST with CNN')

    plt.subplot(3, 1, 1)
    plt.grid()
    plt.plot(x, train_loss, color='#1f77b4', label='SGD')
    plt.plot(x, train_loss1, color='#ff7f0e', label='Adagrad')
    plt.plot(x, train_loss2, color='#2ca02c', label='Adam')
    plt.plot(x, train_loss3, color='#d62728', label='RMSProp')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.subplot(3, 1, 2)
    plt.grid()
    plt.plot(x, train_acc, label='SGD')
    plt.plot(x, train_acc1, label='Adagrad')
    plt.plot(x, train_acc2, label='Adam')
    plt.plot(x, train_acc3, label='RMSProp')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    
    plt.subplot(3, 1, 3)
    plt.grid()
    plt.plot(x, test_acc, label='SGD')
    plt.plot(x, test_acc1, label='Adagrad')
    plt.plot(x, test_acc2, label='Adam')
    plt.plot(x, test_acc3, label='RMSProp')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Acc')

    plt.tight_layout()

    plt.savefig('fig3.pdf')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot')
    parser.add_argument("-m", "--model", help="resnet18 or cnn or lstm", type=str, default='cnn')
    parser.add_argument("-d", "--data", help="Cifar or  MINIST or FMNIST or Shakespeare", type=str, default='MNIST')
    args = parser.parse_args()

    train_path = "./log_" + args.model + "_" + args.data + ".txt"
    test_path = "./acc_" + args.model + "_" + args.data + ".txt"
    train_path1 = "./log_" + args.model + "_" + args.data + "_Adagrad.txt"
    test_path1 = "./acc_" + args.model + "_" + args.data + "_Adagrad.txt"
    train_path2 = "./log_" + args.model + "_" + args.data + "_Adam.txt"
    test_path2 = "./acc_" + args.model + "_" + args.data + "_Adam.txt"
    train_path3 = "./log_" + args.model + "_" + args.data + "_RMS.txt"
    test_path3 = "./acc_" + args.model + "_" + args.data + "_RMS.txt"

    train_consult = pd.read_csv(train_path, sep="\s", header=None, engine='python',
                                names=['0', '1', '2', '3', '4', '5', '6', '7'])
    test_consult = pd.read_csv(test_path, sep="\s|,|=", header=None, engine='python',
                                names=['0', '1','2','3','4'])
    train_consult1 = pd.read_csv(train_path1, sep="\s", header=None, engine='python',
                                names=['0', '1', '2', '3', '4', '5', '6', '7'])
    test_consult1 = pd.read_csv(test_path1, sep="\s|,|=", header=None, engine='python',
                                names=['0', '1','2','3','4'])
    train_consult2 = pd.read_csv(train_path2, sep="\s", header=None, engine='python',
                                names=['0', '1', '2', '3', '4', '5', '6', '7'])
    test_consult2 = pd.read_csv(test_path2, sep="\s|,|=", header=None, engine='python',
                                names=['0', '1','2','3','4'])
    train_consult3 = pd.read_csv(train_path3, sep="\s", header=None, engine='python',
                                names=['0', '1', '2', '3', '4', '5', '6', '7'])
    test_consult3 = pd.read_csv(test_path3, sep="\s|,|=", header=None, engine='python',
                                names=['0', '1','2','3','4'])


    x, train_loss, train_acc, test_acc = GetList(train_consult, test_consult)
    _, train_loss1, train_acc1, test_acc1 = GetList(train_consult1, test_consult1)
    _, train_loss2, train_acc2, test_acc2 = GetList(train_consult2, test_consult2)
    _, train_loss3, train_acc3, test_acc3 = GetList(train_consult3, test_consult3)
    
    draw(x, train_loss, train_acc, test_acc, train_loss1, train_acc1, test_acc1, train_loss2, train_acc2, test_acc2, train_loss3, train_acc3, test_acc3, args)

    print(train_consult)


if __name__ == "__main__":
    main()