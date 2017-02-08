addpath('/home/tomj/Documents/4YP/densecrf/data/PascalVOC2010/VOCdevkit/VOCcode');

VOCinit;

task = 'mf';
results_folder = 'Test'



VOCopts.datadir = '/home/tomj/Documents/4YP/densecrf/data/PascalVOC2010';
VOCopts.testset = 'Test';


VOCopts.seg.clsimgpath= '/home/tomj/Documents/4YP/densecrf/data/PascalVOC2010/SegmentationClass/%s.png';
VOCopts.seg.imgsetpath= '/home/tomj/Documents/4YP/densecrf/data/PascalVOC2010/split/%s.txt';
VOCopts.seg.clsresdir= ['/home/tomj/Documents/4YP/densecrf/data/PascalVOC2010/results/test/lrqp'];
VOCopts.seg.clsrespath=[VOCopts.seg.clsresdir '/%s.png'];



VOCevalseg(VOCopts, task);
