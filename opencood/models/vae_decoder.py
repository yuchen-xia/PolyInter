from torch import nn

class autoencoder(nn.Module):
    def __init__(self, args):
        super(autoencoder, self).__init__()
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
        #     nn.ReLU(True),
        #     nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
        #     nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
        #     nn.ReLU(True),
        #     nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        # )
        self.conv1 = nn.Conv2d(args['in_channel'], 32, 6)
        self.relu1 = nn.ReLU(True)
        self.conv2 = nn.Conv2d(32, 32, 1)
        self.relu2 = nn.ReLU(True)
        self.conv3 = nn.ConvTranspose2d(32,args['out_channel'], 6)
        self.tanh3 = nn.Tanh()
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
        #     nn.Tanh()
        # )

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.relu1(x)
        x = self.conv2(x)
        # print(x.shape)
        x = self.relu2(x)
        x = self.conv3(x)
        # print(x.shape)
        x = self.tanh3(x)
        return x

# if __name__ == "__main__":
#     def to_img(x):
#         x = 0.5 * (x + 1)
#         x = x.clamp(0, 1)
#         x = x.view(x.size(0), 1, 28, 28)
#         return x
    
#     model = autoencoder().cuda()
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-5)
#     starttime = datetime.datetime.now()

#     for epoch in range(num_epochs):
#         for data in dataloader:
#             img, label = data
#             img = Variable(img).cuda()
#             # ===================forward=====================
#             output = model(img)
#             loss = criterion(output, img)
#             # ===================backward====================
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#         # ===================log========================
#         endtime = datetime.datetime.now()
#         print('epoch [{}/{}], loss:{:.4f}, time:{:.2f}s'.format(epoch+1, num_epochs, loss.item(), (endtime-starttime).seconds))
        
#         # if epoch % 10 == 0:
#         pic = to_img(output.cpu().data)
#         save_image(pic, './dc_img/image_{}.png'.format(epoch))

#     torch.save(model.state_dict(), './conv_autoencoder.pth')