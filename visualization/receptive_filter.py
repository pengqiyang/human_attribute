# -*- coding: utf-8 -*-
#一层表示为一个三元组： [filter size, stride, padding]
import math
def forword(conv, layerIn):
  n_in = layerIn
  k = conv[0]
  s = conv[1]
  p = conv[2]
  return math.floor((n_in - k + 2*p)/s) + 1
def alexnet():
  convnet = [[],
  [7,2,3],
  [3,2,1],
  [3,1,0],[3,1,0],[3,1,0],[3,1,0],
  [3,2,0],[3,1,0],[3,1,0],[3,1,0],
  [3,2,0],[3,1,0],[3,1,0],[3,1,0],
  [3,2,0],[3,1,0],[3,1,0],[3,1,0],
 ]
  layer_names = [['input'],'conv1','pool1','B1_conv_1','B1_conv_2','B1_conv_3','B1_conv_4','B2_conv_1','B2_conv_2','B2_conv_3','B2_conv_4','B3_conv_1','B3_conv_2','B3_conv_3','B3_conv_4','B4_conv_1','B4_conv_2','B4_conv_3','B4_conv_4']
  return [convnet, layer_names]
def testnet():
  convnet = [[],[2,1,0],[3,3,1]]
  layer_names = [['input'],'conv1','conv2']
  return [convnet, layer_names]
# layerid >= 1
def receptivefield(net, layerid):
  if layerid > len(net[0]):
    print ('[error] receptivefield:no such layerid!')
    return 0
  rf = 1
  for i in reversed(range(layerid)):
    filtersize, stride, padding = net[0][i+1]
    rf = (rf - 1)*stride + filtersize
  print ('                感受野大小为:%d.' % (int(rf)))
  return rf
def anylayerout(net, layerin, layerid):
  if layerid > len(net[0]):
    print ('[error] anylayerout:no such layerid!')
    return 0
  for i in range(layerid):
    if i == 0:
      fout = forword(net[0][i+1], layerin)
      continue
    fout = forword(net[0][i+1], fout)
  print ('当前层为:%s, 输出节点维度为:%d.' % (net[1][layerid], int(fout)))
#x,y>=1
def receptivefieldcenter(net, layerid, x, y):
  if layerid > len(net[0]):
    print ('[error] receptivefieldcenter:no such layerid!')
    return 0
  al = 1
  bl = 1
  for i in range(layerid):
    filtersize, stride, padding = net[0][i+1]
    al = al * stride
    ss = 1
    for j in range(i):
      fsize, std, pad = net[0][j+1]
      ss = ss * std
    bl = bl + ss * (float(filtersize-1)/2 - padding)
  xi0 = al * (x - 1) + float(bl)
  yi0 = al * (y - 1) + bl
  print ('                该层上的特征点(%d,%d)在原图的感受野中心坐标为:(%.1f,%.1f).' % (int(x), int(y), float(xi0), float(yi0)))
  return (xi0, yi0)
# net:为某个CNN网络
# insize:为输入层大小
# totallayers：为除了输入层外的所有层个数
# x,y为某层特征点坐标
def printlayer(net, insize, totallayers, x, y):
  for i in range(totallayers):
    # 计算每一层的输出大小
    anylayerout(net, insize, i+1)
    # 计算每层的感受野大小
    receptivefield(net, i+1)
    # 计算feature map上(x,y)点在原图感受野的中心位置坐标
    receptivefieldcenter(net, i+1, x, y)
if __name__ == '__main__':
  #net = testnet() 
  #printlayer(net, insize=6, totallayers=2, x=1, y=1)
  net = alexnet()
  printlayer(net, insize=256, totallayers=18, x=32, y=32)