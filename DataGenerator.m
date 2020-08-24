function [trainData] = DataGenerator(nOfFme, vidWid, vidHigh, fmeWid, fmeHigh)

totalDataNum = 0;
fileList = dir('*.mp4');
nOfFile = length(fileList);

for i = 1:1:nOfFile
    fileName = strcat(fileList(i).name);
    obj = VideoReader(fileName); 
    totalDataNum = totalDataNum + obj.NumberOfFrames;    
end
fprintf("Total data %d.\n", totalDataNum/nOfFme);
trainData = single(zeros(fix(totalDataNum/nOfFme), fmeHigh, fmeWid, nOfFme));

numOfData = 0;
for i = 1:1:nOfFile
     fprintf("File: %d/%d.\n", i, nOfFile);
     fileName = strcat(fileList(i).name);
     obj = VideoReader(fileName);
     for j=1:nOfFme:(obj.NumberOfFrames-nOfFme)
          numOfData = numOfData + 1;       
          fprintf("num=%d\n", numOfData);
          for k=1:1:nOfFme
                frame = read(obj, j+k-1);
                yCbCr = rgb2ycbcr(frame);
                trainData(numOfData, :, :, k) = imresize(reshape(yCbCr(:, :, 1), vidHigh, vidWid), [fmeHigh, fmeWid]);
                %imshow(reshape(trainData(numOfData, :, :, k), fmeHigh, fmeWid), []);
          end
     end
     clear frame yCbCr;
end

end
