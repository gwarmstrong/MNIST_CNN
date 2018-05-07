--------------------------Requirements--------------------------------

require 'nn'
--require 'nngraph'
require 'cutorch'
require 'cunn'
--require 'cudnn'


--------------------------Command-------------------------------------
cmd = torch.CmdLine()
cmd:option('-iterations', 100, 'number of iterations to train')
cmd:option('-rate', 0.0001, 'learning rate')
cmd:option('-out', 'final_model', 'model output name')

opt = cmd:parse(arg or {})
---------------------------Data---------------------------------------

print '==> downloading dataset'

-- Here we download dataset files. 

tar = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/mnist.t7.tgz'

if not paths.dirp('data/mnist.t7') then
   os.execute('wget ' .. tar)
   os.execute('tar xvf ' .. paths.basename(tar))
end

train_file = 'data/mnist.t7/train_32x32.t7'
test_file = 'data/mnist.t7/test_32x32.t7'

----------------------------------------------------------------------
print '==> loading dataset'

trainData = torch.load(train_file,'ascii')
testData = torch.load(test_file,'ascii')
classes = {'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'}


print('Training Data:')
print(trainData)
print()

print('Test Data:')
print(testData)
print()

-- set the index operator
setmetatable(trainData, {__index = function(t, i) return {t.data[i], t.labels[i]} end});

function trainData:size()
	return self.data:size(1)
end

-- convert the data from ByteTensor to DoubleTensor
trainData.data = trainData.data:double()
testData.data = testData.data:double()

-- normalize the training data
mean = trainData.data:select(2,1):mean()
trainData.data:select(2,1):add(-mean)

stdv = trainData.data:select(2,1):std()
trainData.data:select(2,1):div(stdv)


net = nn.Sequential()
net:add(nn.SpatialConvolution(1,6,5,5))
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.Threshold())

net:add(nn.SpatialConvolution(6,16,5,5))
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.Threshold())

net:add(nn.View(16*5*5))

net:add(nn.Linear(16*5*5, 120))
net:add(nn.Threshold())
net:add(nn.Linear(120, 84))
net:add(nn.Threshold())
net:add(nn.Linear(84,10))

criterion = nn.CrossEntropyCriterion()

-- Transfer the network, criterion, and data to gpu
net = net:cuda()
criterion = criterion:cuda()
trainData.data = trainData.data:cuda()
testData.data = testData.data:cuda()


-- Train the neural network
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = opt.rate
trainer.maxIteration = opt.iterations

trainer:train(trainData)

-- save the trained model
torch.save('models/'.. opt.out ..'.t7', net)

-- write the weights of the convolution layers to file
file1 = io.open('filters/first_conv.txt','w')
file1:write(tostring(net.modules[1].weight))
file1:close()

file2 = io.open('filters/second_conv.txt','w')
file2:write(tostring(net.modules[4].weight))
file2:close()

-- Find number of correct predictions
correct = 0
class_performance = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
class_total = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
for i=1,10000 do
	local groundtruth = testData.labels[i]
	local prediction = net:forward(testData.data[i])
	local confidences, indices = torch.sort(prediction, true)
	class_total[groundtruth] = class_total[groundtruth] + 1
	if groundtruth == indices[1] then
		correct = correct + 1
		class_performance[groundtruth] = class_performance[groundtruth] + 1
	end
end

print('Correct:' .. correct, 100*correct/10000 .. ' % ')
for i=1,#classes do
	print(classes[i], 100*class_performance[i]/class_total[i] .. '%')
end


