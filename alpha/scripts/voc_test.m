addpath('/data/PascalVOC2010/VOCdevkit/VOCcode');

VOCinit;


task = 'mfiter';


VOCopts.datadir = '/data/PascalVOC2010';
VOCopts.resdir = ['/data/Results/Pascal2010/Test/' task '/'];
VOCopts.testset = 'Test';


VOCopts.seg.clsimgpath='/data/PascalVOC2010/SegmentationClass/%s.png';
VOCopts.seg.imgsetpath='/data/PascalVOC2010/split/%s.txt';
VOCopts.seg.clsresdir='/data/Results/Pascal2010/Test/%s';
VOCopts.seg.clsrespath=[VOCopts.seg.clsresdir '/%s.png'];



VOCevalseg(VOCopts, task);