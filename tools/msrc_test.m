function score = msrc_test(path, dataset_split)
addpath('/media/ajanthan/b7391340-f7ed-49ef-9dab-f3749bde5917/ajanthan/NICTA/Research/ubuntu_codes/lp_densecrf/densecrf/tools/');

dpath = '/media/ajanthan/b7391340-f7ed-49ef-9dab-f3749bde5917/ajanthan/NICTA/Research/ubuntu_codes/data/';

datadir = [dpath 'MSRC/'];
testset = dataset_split;


%gtimgpath0= [datadir 'MSRC_ObjCategImageDatabase_v2/GroundTruth/%s_GT.bmp'];
%gtimgpath= [datadir 'MSRC_ObjCategImageDatabase_v2/GroundTruth/%s_GT.png'];
gtimgpath= [datadir 'fine_annot/%s_GT.bmp'];
respath_tmpl=[path '/%s.png'];

path_to_test_set = [datadir '/split/' dataset_split '.txt'];
[gtids,t]=textread(path_to_test_set,'%s %d');

num = 22; % Number of classes

confcounts = zeros(num);
count=0;

cmap = MSRClabelcolormap(num);

for i=1:length(gtids)
    imname = gtids{i};
    toks = strsplit(imname, '.');
    imname = toks{1};
     try
    % ground truth label file
    gtfile = sprintf(gtimgpath,imname);
%     break;
    [gtim,map] = imread(gtfile);
    gtim = rgb2ind(gtim, cmap);
    gtim = double(gtim);


    % results file
    resfile = sprintf(respath_tmpl, imname);
    [resim,map] = imread(resfile);
    %% Added Code    
    resim = rgb2ind(resim, cmap);
    resim = double(resim);

    % Check validity of results image
    maxlabel = max(resim(:));
    if (maxlabel>22),
        error(['Results image ''%s'' has out of range value %d (the ' ...
               'value should be <= %d)'],imname,maxlabel,num-1);
    end

    szgtim = size(gtim);
    szresim = size(resim);
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

     catch
     end
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
   if (j>1), clname = int2str(j-1);end;
   fprintf('  %14s: %6.3f%%\n',clname,accuracies(j));
end
accuracies = accuracies(1:end);
avacc = mean(accuracies);

score = avacc;
end

function cmap = MSRClabelcolormap(N)
    assert(N == 22);
    
    cmap = zeros(21,3);
    cmap(1,:) = [128,0,0];
    cmap(2,:) = [0,128,0];
    cmap(3,:) = [128,128,0];
    cmap(4,:) = [0,0,128];
    cmap(5,:) = [0,128,128];
    cmap(6,:) = [128,128,128];
    cmap(7,:) = [192,0,0];
    cmap(8,:) = [64,128,0];
    cmap(9,:) = [192,128,0];
    cmap(10,:) = [64,0,128];
    cmap(11,:) = [192,0,128];
    cmap(12,:) = [64,128,128];
    cmap(13,:) = [192,128,128];
    cmap(14,:) = [0,64,0];
    cmap(15,:) = [128,64,0];
    cmap(16,:) = [0,192,0];
    cmap(17,:) = [128,64,128];
    cmap(18,:) = [0,192,128];
    cmap(19,:) = [128,192,128];
    cmap(20,:) = [64,64,0];
    cmap(21,:) = [192,64,0];
    cmap(22,:) = [0,0,0];
    
    cmap = cmap / 255;
end