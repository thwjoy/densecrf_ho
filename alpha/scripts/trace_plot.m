%% plot enegy vs time graphs for different algorithms
clear all;
addpath('/media/ajanthan/b7391340-f7ed-49ef-9dab-f3749bde5917/ajanthan/NICTA/Research/Codes/export_fig/altmany-export_fig-0a0fea6/');

fpath = '/media/ajanthan/b7391340-f7ed-49ef-9dab-f3749bde5917/ajanthan/NICTA/Research/ubuntu_codes/data/densecrf/MSRC/Test_10_5_1000_0.1_1000_1000_1_test_lcf_cv2';
img = '2_14_s';
fname = [fpath '/tracing-%s/' img '.trc'];
epsname = [fpath '/' img '_trc.eps'];
pdfname = [fpath '/' img '_trc.pdf'];
paperfpath = '/media/ajanthan/b7391340-f7ed-49ef-9dab-f3749bde5917/ajanthan/NICTA/Research/ubuntu_docs/lp_densecrf/lp-densecrf-paper/images/msrc';
paperpdfname = [paperfpath '/' img '_trc.pdf'];

algos = {'mf', 'fixedDC-CCV', 'sg_lp', 'prox_lp_0.001', 'prox_lp_0.1', 'prox_lp_rest'};
names = {'MF', 'DC_{neg}', 'SG-LP', 'PROX-LP_{0.001}', 'PROX-LP_{0.1}', 'PROX-LP_{acc}'};
mark = {'-c.', '-y+', '-mp', '-g^', '-r>', '-bo'};
color = {'c', 'y', 'm', 'g', 'r', 'b'};

data = cell(length(algos), 1);
for i = 1 : length(algos)
    try
    data{i} = dlmread(sprintf(fname, algos{i}), '\t');
    time = data{i}(:,2);
    time = cumsum(time);    
    plot(time, data{i}(:,3), mark{i}, 'LineWidth',2,...
            'MarkerEdgeColor',color{i},...
            'MarkerFaceColor',color{i},...
            'MarkerSize',4);
    hold all;
    catch
    end
end

title('Assignment energy as a function of time', 'FontSize', 15);
ylabel('Integral energy', 'FontSize', 15);
xlabel('Time (s)', 'FontSize', 15);
hleg = legend(names{1}, names{2}, names{3}, names{4}, names{5}, names{6});
set(hleg,'Location','NorthEast','FontSize',15);
xlim([0,10]);

print('-depsc2', epsname);
eps2pdf(epsname, pdfname);
eps2pdf(epsname, paperpdfname); 