--------------------------Requirements--------------------------------

require 'nn'
require 'cutorch'
require 'cunn'

--------------------------Command-------------------------------------

cmd = torch.CmdLine()
cmd:option('-iterations', 15, 'number of iterations to train')
cmd:option('-rate', 0.0001, 'learning rate')
cmd:option('-out', 'model', 'model output name')
cmd:option('-decay', 0, 'learning rate decay')
cmd:option('-stats', '', 'a = accuracy, e = error func., ae = both')
cmd:option('-batchSize', -1, 'size of training batches')

opt = cmd:parse(arg or {})

----------------------File I/O Shenanigans----------------------------

errorOut = false
accuracyOut = false
write_out_stats = true
if opt.stats == '' then
	write_out_stats = false
elseif opt.stats == 'a' then
	accuracyOut = true
elseif opt.stats == 'e' then
	errorOut = true
elseif opt.stats == 'ae' then
	accuracyOut = true
	errorOut = true
else
	assert(false, 'invalid argument for -stats')
end

function initOutFile()
	if write_out_stats then
		accuracyFile = assert(io.open('accuracy/acc_'..opt.out..'.tsv','w'))	
		accuracyFile:write('time')
		if errorOut then
			accuracyFile:write('\t'..'error')
		end
		if accuracyOut then
			accuracyFile:write('\t'..'train_accuracy'..'\t'..'test_accuracy')
		end
		accuracyFile:write('\n')

		return accuracyFile
	end
end

function updateOutFile(dataset, testset, currentError, iteration, calc_time)
	if write_out_stats then
		accuracyFile:write(calc_time)
		if errorOut then
			currentError = string.format("%02f",currentError)
			accuracyFile:write('\t'..currentError)
		end
		if accuracyOut then
			local currentAccuracy = ModelAccuracy(dataset)
			local currentTestAccuracy = ModelAccuracy(testset)
			local leadingSpaces = string.rep(' ', string.len(tostring(iteration)) + 2)
			print(leadingSpaces..'train accuracy = '..currentAccuracy)
			print(leadingSpaces..'test accuracy = '..currentTestAccuracy)
			-- format outputs to string
			currentAccuracy = string.format("%02f",currentAccuracy)
			currentTestAccuracy = string.format("%02f",currentTestAccuracy)
			-- write accuracy and train/test error to file
			accuracyFile:write('\t'..currentAccuracy..'\t'..currentTestAccuracy)
		end
		accuracyFile:write('\n')
	end
end

function closeOutFile()
	if write_out_stats then
		accuracyFile:close()
	end
end

---------------------------Data---------------------------------------

if not paths.dirp('data/mnist.t7') then
	
	print '==> downloading dataset'

	-- Here we download dataset files. 

	tar = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/mnist.t7.tgz'
	
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

-- provides some operations that make it easier to access the data
setmetatable(trainData, {__index = function(t, i) return {t.data[i], t.labels[i]} end});
setmetatable(testData, {__index = function(t, i) return {t.data[i], t.labels[i]} end});

function trainData:size()
	return self.data:size(1)
end

function testData:size()
	return self.data:size(1)
end

-- convert the data from ByteTensor to DoubleTensor
trainData.data = trainData.data:double()
testData.data = testData.data:double()

-- normalize the training data
mean = trainData.data:select(2,1):mean()
trainData.data:select(2,1):add(-mean)
testData.data:select(2,1):add(-mean)

stdv = trainData.data:select(2,1):std()
trainData.data:select(2,1):div(stdv)
testData.data:select(2,1):div(stdv)

-- build the neural network
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

-- define error function
criterion = nn.CrossEntropyCriterion()

-- Transfer the network, criterion, and data to gpu
net = net:cuda()
criterion = criterion:cuda()
trainData.data = trainData.data:cuda()
testData.data = testData.data:cuda()

-- function for getting the accuracy of net on the data given by dataset
function ModelAccuracy(dataset)
	-- this should evaluate all outputs in parallel on GPU if dataset is on GPU
	local outputs = net:forward(dataset.data)

	num_correct = 0
	for i = 1,dataset:size() do
		-- local prediction = net:forward(dataset[i][1])
		local confidences, indices = torch.sort(outputs[i], true)
		if indices[1] == dataset[i][2] then
			num_correct = num_correct + 1
		end
	end
	return num_correct/dataset:size()
end

-- defines the mini batch gradient descent method
function MiniBatchGD(dataset,testset)
	-- this is my customized version of nn.StochasticGradient
	-- refactor this to update weights only after each batch is completed
	local iteration = 1
	local currentLearningRate = opt.rate
	
	io.write('# MiniBatchGD: training\n')
	
	initOutFile()

	local batchSize
	-- Set the batch size
	assert(opt.batchSize <= dataset:size())
	if opt.batchSize == -1 then
		batchSize = dataset:size()
	else
		batchSize = opt.batchSize
	end

	local calc_time = 0
	local elapsed_time = 0
	local start_time
	local mid_time
	local end_time

	while (opt.iterations > 0 and iteration <= opt.iterations) do
		net:clearState()
		--torch.save('models/'..opt.out..'_' .. iteration .. '.t7', net)

		start_time = os.time()

		local shuffledIndices = torch.randperm(dataset:size(), 'torch.LongTensor') 
		local thisBatch= shuffledIndices[{{1, batchSize}}]	
		
		local inputs = dataset.data:index(1, thisBatch)
		local targets = dataset.labels:index(1, thisBatch)	
		local currentError = criterion:forward(net:forward(inputs),targets)
		
		net:updateGradInput(inputs, criterion:updateGradInput(net.output, targets))
		net:accUpdateGradParameters(inputs, criterion.gradInput, currentLearningRate)

		mid_time = os.time()
		calc_time = calc_time + os.difftime(mid_time, start_time)
		print('calculation time: '..calc_time..' s')
 		print(iteration..': current error = '..currentError)
				
		updateOutFile(dataset, testset, currentError, iteration, calc_time)

		-- torch.save('models/'..opt.out..'_' .. iteration .. '.t7', net)

		iteration = iteration + 1
		currentLearningRate = opt.rate / (1+iteration*opt.decay)

		end_time = os.time()
		elapsed_time = elapsed_time + os.difftime(end_time, start_time)
		print('total time: '..elapsed_time..' s')
		print('')
	end

	closeOutFile()
	
end

-- Train the neural network
MiniBatchGD(trainData,testData)	

-- save the final model
net:clearState()
torch.save('models/'.. opt.out .. '_final.t7', net)
-- to load model, require 'cunn', then load [model].t7 file

-- only necessary if you want to inspect the filters
--[[ write the weights of the convolution layers to file
file1 = io.open('filters/'..opt.out..'_first_conv_final.txt','w')
file1:write(tostring(net.modules[1].weight))
file1:close()

file2 = io.open('filters/'..opt.out..'second_conv_final.txt','w')
file2:write(tostring(net.modules[4].weight))
file2:close()
]]

-- Find number of correct predictions
correct = 0
class_performance = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
class_total = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
for i=1,testData:size() do
	local groundtruth = testData.labels[i]
	local prediction = net:forward(testData.data[i])
	local confidences, indices = torch.sort(prediction, true)
	class_total[groundtruth] = class_total[groundtruth] + 1
	if groundtruth == indices[1] then
		correct = correct + 1
		class_performance[groundtruth] = class_performance[groundtruth] + 1
	end
end

print('Testing Correct:' .. correct, 100*correct/testData:size() .. ' % ')
for i=1,#classes do
	print(classes[i], 100*class_performance[i]/class_total[i] .. '%')
end
