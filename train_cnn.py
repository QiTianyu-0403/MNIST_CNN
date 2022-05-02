from init_cnn import init
import torch


def train_cnn(args):
    device, trainloader, testloader, net, criterion, optimizer, _, _ = init(args)

    best_acc = 85  # 2 初始化best test accuracy
    pre_epoch = 0

    print("Start Training: "+'CNN'+"--"+"MNIST")
    with open("./acc/"+"acc_"+'CNN'+"_"+"MNIST"+".txt", "w") as f:
        with open("./log/"+"log_"+'CNN'+"_"+"MNIST"+".txt", "w")as f2:
            for epoch in range(pre_epoch, args.EPOCH):
                print('\nEpoch: %d' % (epoch + 1))
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                for i, data in enumerate(trainloader, 0):
                    length = len(trainloader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    # forward + backward
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    # for name, parms in net.named_parameters():	
                    #     print('-->name:', name, '-->grad_requirs:',parms.requires_grad, \
                    #         ' -->grad_value:',parms.grad)
                    grad1 = net.conv2.bias.grad
                    grad2 = net.conv2.weight.grad
                    grad3 = net.fc2.bias.grad
                    grad4 = net.fc2.weight.grad
                    optimizer.step()

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)    #predicted返回的是tensor每行最大的索引值
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('\n')
                    f2.flush()

                # print("conv21.bias.grad = ",net.conv2.bias.grad)
                # print("conv21.weight.grad = ",net.conv2.weight.grad)
                conv2_bias_grad = grad1
                conv2_weight_grad = grad2
                fc2_bias_grad = grad3
                fc2_weight_grad = grad4
                norm_1 = torch.norm(conv2_bias_grad) * 100
                norm_2 = torch.norm(conv2_weight_grad) * 100
                norm_3 = torch.norm(fc2_bias_grad) * 100
                norm_4 = torch.norm(fc2_weight_grad) * 100
                # 每训练完一个epoch测试一下准确率
                print("Waiting Test!")
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for data in testloader:
                        net.eval()
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                    print('测试分类准确率为：%.3f%%' % (100. * correct / total))
                    acc = 100. * correct / total
                    # 将每次测试结果实时写入acc.txt文件中
                    # print('Saving model......')
                    # torch.save(net.state_dict(), '%s/net_%03d.pth' % (args.outf, epoch + 1))
                    f.write("EPOCH=%03d,Accuracy= %.3f%%,%.3f%%,%.3f%%,%.3f%%,%.3f%%" % (epoch + 1, acc, norm_1, norm_2, norm_3, norm_4))
                    f.write('\n')
                    f.flush()

            print("Training Finished, TotalEPOCH=%d" % args.EPOCH)