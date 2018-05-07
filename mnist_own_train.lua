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
cmd:option('-accuracyOut', 'false', 'write error and accuracy to file')
cmd:option('-batchSize', -1, 'size of training batches')

opt = cmd:parse(arg or {})

if opt.accuracyOut == 'false' then
	opt.accuracyOut= false
elseif opt.accuracyOut == 'true' then
	opt.accuracyOut = true
else
	assert(false, 'invalid argument for -accuracyOut')
end

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
function ModelAccuracy(dataset)
	num_correct = 0
	for i = 1,dataset:size() do
		local prediction = net:forward(dataset[i][1])
		local confidences, indices = torch.sort(prediction, true)
		if indices[1] == dataset[i][2] then
			num_correct = num_correct + 1
		end
	end
	return num_correct/dataset:size()
end

function MiniBatchSGD(dataset,testset)
	-- this is my customized version of nn.StochasticGradient
	-- refactor this to update weights only after each batch is completed
	local iteration = 1
	local currentLearningRate = opt.rate
	
	io.write('# CustomSGD: training\n')
	
	local accuracyFile 
	if opt.accuracyOut then
		accuracyFile = assert(io.open('accuracy/acc_'..opt.out..'.tsv','w'))	
		accuracyFile:write('time'..'\t'..'error'..'\t'..'train_accuracy'..'\t'..'test_accuracy\n')
	end


	 
	local batchSize
	-- Set the batch size
	assert(opt.batchSize <= dataset:size())
	if opt.batchSize == -1 then
		batchSize = dataset:size()
	else
		batchSize = opt.batchSize
	end

	local elapsed_time = 0
	local start_time
	local end_time

	while (opt.iterations > 0 and iteration <= opt.iterations) do
		net:clearState()
		--torch.save('models/'..opt.out..'_' .. iteration .. '.t7', net)

		start_time = os.time()

		local currentError = 0

		local shuffledIndices = torch.randperm(dataset:size(), 'torch.LongTensor') 
		
		-- Gradient descent step per training example
		for t = 1,batchSize do
			local example = dataset[shuffledIndices[t]]
			local input = example[1]
			local target = example[2]

			currentError = currentError + criterion:forward(net:forward(input),target) 
			
			net:updateGradInput(input, criterion:updateGradInput(net.output, target))
			net:accUpdateGradParameters(input, criterion.gradInput, currentLearningRate)
		end
	
		end_time = os.time()
		currentError = currentError / batchSize
		elapsed_time = elapsed_time + os.difftime(end_time, start_time)
		print('elapsed time:'..elapsed_time..'s')
 		print(iteration..': current error = '..currentError)

		if opt.accuracyOut then
				local currentAccuracy = ModelAccuracy(dataset)
				local currentTestAccuracy = ModelAccuracy(testset)
				local leadingSpaces = string.rep(' ', string.len(tostring(iteration)) + 2)
				print(leadingSpaces..'train accuracy = '..currentAccuracy)
				print(leadingSpaces..'test accuracy = '..currentTestAccuracy)
				-- format outputs to string
				currentError = string.format("%02f",currentError)
				currentAccuracy = string.format("%02f",currentAccuracy)
				currentTestAccuracy = string.format("%02f",currentTestAccuracy)
				-- write accuracy and train/test error to file
				accuracyFile:write(elapsed_time..'\t'..currentError..'\t'..currentAccuracy..'\t'..currentTestAccuracy..'\n')
		end

		iteration = iteration + 1
		currentLearningRate = opt.rate / (1+iteration*opt.decay)
	end
	if opt.accuracyOut then
		accuracyFile:close()
	end
	
end

-- add SGD and Gradient Descent, both slightly updated to output the errors and accuracies as specified
-- update method to specify 'calculateAccuracy' and 'toFile' so that batch/overall errors can be written without computing accuracies.

MiniBatchSGD(trainData,testData)	


-- save the final model
net:clearState()
torch.save('models/'.. opt.out .. '_final.t7', net)
-- to load model, require 'cunn', then load [model].t7 file


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


