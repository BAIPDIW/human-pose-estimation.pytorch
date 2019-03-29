from torch import nn
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.conv1 = nn.Conv2d(channel,channel // reduction,kernel_size=1,bias=False)
        self.PReLU1 = nn.PReLU(num_parameters=channel // reduction)
        self.bn1 = nn.BatchNorm2d(channel // reduction)
        self.conv2 = nn.Conv2d(channel,channel // reduction,kernel_size=1,bias=False)
        self.bn2 = nn.BatchNorm2d(channel // reduction)
        self.conv3 = nn.Conv2d(channel,channel // reduction,kernel_size=1,bias=False)
        self.bn3 = nn.BatchNorm2d(channel // reduction)
        self.conv4 = nn.Conv2d(channel // reduction,channel,kernel_size=1,bias=False)
        self.bn4 = nn.BatchNorm2d(channel)
        self.PReLU4 = nn.PReLU(num_parameters=channel)
        self.softmax = nn.Softmax2d()

        # super(SELayer, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Sequential(
        #     nn.Linear(channel, channel // reduction, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(channel // reduction, channel, bias=False),
        #     nn.Sigmoid()
        # )




    def forward(self, x):


        # b, c, _, _ = x.size()
        # y = self.avg_pool(x).view(b, c)
        # y = self.fc(y).view(b, c, 1, 1)
        # return x * y.expand_as(x)
        x_reduction = self.conv1(x)
        x_reduction = self.bn1(x_reduction)
        x_reduction = self.PReLU1(x_reduction)
        b1,c1,h1,w1 = x_reduction.size()
        #print("b1={},c1={},h1={},w1={}".format(b1,c1,h1,w1))
        x_reduction = x_reduction.reshape(b1,c1,h1*w1)

        x_attention_maps = self.conv2(x)
        x_attention_maps = self.bn2(x_attention_maps)
        x_attention_maps = self.softmax(x_attention_maps)
        b2,c2,h2,w2 = x_attention_maps.size()
        #print("b2={},c2={},h2={},x2={}".format(b2,c2,h2,w2))
        x_attention_maps = x_attention_maps.reshape(b2,c2,h2*w2)
        x_attention_maps = x_attention_maps.transpose(1,2)
        #print("x_reduction.shape={}".format(x_reduction.shape))
        #print("x_attention_maps.shape={}".format(x_attention_maps.shape))


        global_descriptors = x_reduction.bmm(x_attention_maps)
        #print('global_descriptors.shape={}'.format(global_descriptors.shape))

        attention_vectors = self.conv3(x)
        attention_vectors = self.bn3(attention_vectors)
        attention_vectors = self.softmax(attention_vectors)
        b3,c3,h3,w3 = attention_vectors.size()
        attention_vectors = attention_vectors.reshape(b3,c3,h3*w3)
        #print("attention_vectors.shape={}".format(attention_vectors.shape))
        z = global_descriptors.matmul(attention_vectors)
        #print("z.shape={}".format(z.shape))
        z = z.reshape(b1,c1,h1,w1)
        z = self.conv4(z)
        z = self.bn4(z)
        z = self.PReLU4(z)
        x = x+z

        return x

        
# b1=32,c1=128,h1=8,w1=6
# b2=32,c2=128,h2=8,x2=6
# x_reduction.shape=torch.Size([32, 128, 48])
# x_attention_maps.shape=torch.Size([32, 48, 128])
# global_descriptors.shape=torch.Size([32, 128, 128])
# attention_vectors.shape=torch.Size([32, 128, 48])
# z.shape=torch.Size([32, 128, 48])
