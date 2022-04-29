import argparse
from train_cnn import train_cnn

def main():
    parser = argparse.ArgumentParser(description='MNIST CNN')
    
    parser.add_argument("-bs", "--batchsize", help="the batch size of each epoch", type=int, default=128)
    parser.add_argument("-e", "--EPOCH", help="the number of epochs", type=int, default=50)
    parser.add_argument("-lr", "--learning_rate", help="learning rate", type=float, default=0.01)
    args = parser.parse_args()
    
    train_cnn(args)
    
    
if __name__ == "__main__":
    main()