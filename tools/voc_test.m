function score = voc_test(path, dataset_split)
addpath('/media/ajanthan/b7391340-f7ed-49ef-9dab-f3749bde5917/ajanthan/NICTA/Research/ubuntu_codes/data/PascalVOC2010/VOCdevkit/VOCcode');

VOCinit;



VOCopts.datadir = '/media/ajanthan/b7391340-f7ed-49ef-9dab-f3749bde5917/ajanthan/NICTA/Research/ubuntu_codes/data/PascalVOC2010';
testset = dataset_split;


gtimgpath= '/media/ajanthan/b7391340-f7ed-49ef-9dab-f3749bde5917/ajanthan/NICTA/Research/ubuntu_codes/data/PascalVOC2010/SegmentationClass/%s.png';
imgsetpath= '/media/ajanthan/b7391340-f7ed-49ef-9dab-f3749bde5917/ajanthan/NICTA/Research/ubuntu_codes/data/PascalVOC2010/split/%s.txt';
respath_tmpl=[path '/%s.png'];

path_to_test_set = ['/media/ajanthan/b7391340-f7ed-49ef-9dab-f3749bde5917/ajanthan/NICTA/Research/ubuntu_codes/data/PascalVOC2010/split/' dataset_split '.txt'];
[gtids,t]=textread(path_to_test_set,'%s %d');

num = 21; % Number of classes

confcounts = zeros(num);
count=0;

for i=1:length(gtids)
    imname = gtids{i};

    % ground truth label file
    gtfile = sprintf(gtimgpath,imname);
%     break;
    [gtim,map] = imread(gtfile);
    gtim = double(gtim);

    %try
    % results file
    resfile = sprintf(respath_tmpl, imname);
    [resim,map] = imread(resfile);
    %% Added Code
    cmap = VOClabelcolormap(num);
    resim = rgb2ind(resim, cmap);
    resim = double(resim);

    % Check validity of results image
    maxlabel = max(resim(:));
    if (maxlabel>VOCopts.nclasses),
        error(['Results image ''%s'' has out of range value %d (the ' ...
               'value should be <= %d)'],imname,maxlabel,num-1);
    end

    szgtim = size(gtim); szresim = size(resim);
    if any(szgtim~=szresim)
        error('Results image ''%s'' is the wrong size, was %d x %d, should be %d x %d.',imname,szresim(1),szresim(2),szgtim(1),szgtim(2));
    end

    %pixel locations to include in computation
    locs = gtim<255;

    % joint histogram
    sumim = 1+gtim+resim*num;
    hs = histc(sumim(locs),1:num*num);
    count = count + numel(find(locs));
    confcounts(:) = confcounts(:) + hs(:);

    %catch
    %end
end

% confusion matrix - first index is true label, second is inferred label
%conf = zeros(num);
conf = 100*confcounts./repmat(1E-20+sum(confcounts,2),[1 size(confcounts,2)]);
rawcounts = confcounts;

% Percentage correct labels measure is no longer being used.  Uncomment if
% you wish to see it anyway
overall_acc = 100*sum(diag(confcounts)) / sum(confcounts(:));
fprintf('Percentage of pixels correctly labelled overall: %6.3f%%\n',overall_acc);

accuracies = zeros(num-1,1);
for j=1:num

   gtj=sum(confcounts(j,:));
   resj=sum(confcounts(:,j));
   gtjresj=confcounts(j,j);
   % The accuracy is: true positive / (true positive + false positive + false negative)
   % which is equivalent to the following percentage:
   accuracies(j)=100*gtjresj/(gtj+resj-gtjresj);

   clname = 'background';
   if (j>1), clname = VOCopts.classes{j-1};end;
   fprintf('  %14s: %6.3f%%\n',clname,accuracies(j));
end
accuracies = accuracies(1:end);
avacc = mean(accuracies);

score = avacc;
end