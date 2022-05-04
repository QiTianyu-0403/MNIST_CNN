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
    grad1, grad2, grad3, grad4 = [], [], [], []
    for i in range(len(train_consult)-1):
        if train_consult.iloc[i, 0] + 1 == train_consult.iloc[i+1,0] or i == len(train_consult)-2:
            if train_consult.iloc[i, 0] > test_consult.iloc[len(test_consult)-1, 1]:
                break
            x.append(train_consult.iloc[i, 0])
            train_loss.append(train_consult.iloc[i, 4])
            train_acc.append(per2float(train_consult.iloc[i, 7]))
            test_acc.append(per2float(test_consult.iloc[train_consult.iloc[i, 0]-1, 4]))
            grad1.append(per2float(test_consult.iloc[train_consult.iloc[i, 0]-1, 5])/100)
            grad2.append(per2float(test_consult.iloc[train_consult.iloc[i, 0]-1, 6])/100)
            grad3.append(per2float(test_consult.iloc[train_consult.iloc[i, 0]-1, 7])/100)
            grad4.append(per2float(test_consult.iloc[train_consult.iloc[i, 0]-1, 8])/100)
    return x, train_loss, train_acc, test_acc, grad1, grad2, grad3, grad4

def draw(x, train_loss, train_acc, test_acc, grad1, grad2, grad3, grad4, args):

    fig = plt.figure()
    
    plt.title('Gradient Norm')
    plt.grid()
    plt.plot(x, grad1, color='#1f77b4', linestyle='-.', label='Conv2 bias')
    plt.plot(x, grad2, color='#1f77b4', label='Conv2 weight')
    plt.plot(x, grad3, color='#ff7f0e', linestyle='-.',label='FC2 bias')
    plt.plot(x, grad4, color='#ff7f0e', label='Fc2 weight')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Norm')

    plt.tight_layout()

    plt.savefig('fig4.pdf')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot')
    parser.add_argument("-m", "--model", help="resnet18 or cnn or lstm", type=str, default='cnn')
    parser.add_argument("-d", "--data", help="Cifar or  MINIST or FMNIST or Shakespeare", type=str, default='MNIST')
    args = parser.parse_args()

    train_path = "./log_" + args.model + "_" + args.data + "_grad.txt"
    test_path = "./acc_" + args.model + "_" + args.data + "_grad.txt"

    train_consult = pd.read_csv(train_path, sep="\s", header=None, engine='python',
                                names=['0', '1', '2', '3', '4', '5', '6', '7'])
    test_consult = pd.read_csv(test_path, sep="\s|,|=", header=None, engine='python',
                                names=['0', '1','2','3','4','5','6','7','8'])


    x, train_loss, train_acc, test_acc, grad1, grad2, grad3, grad4 = GetList(train_consult, test_consult)
    draw(x, train_loss, train_acc, test_acc, grad1, grad2, grad3, grad4, args)

    print(test_consult)


if __name__ == "__main__":
    main()