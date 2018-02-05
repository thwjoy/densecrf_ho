%% plot enegy vs time graphs for different algorithms
clear all;
zoom = 1;
lambda = 0;
addpath('/media/ajanthan/b7391340-f7ed-49ef-9dab-f3749bde5917/ajanthan/NICTA/Research/Codes/export_fig/altmany-export_fig-0a0fea6/');

if (lambda == 1)
    fpath = '/media/ajanthan/b7391340-f7ed-49ef-9dab-f3749bde5917/ajanthan/NICTA/Research/ubuntu_codes/data/densecrf/MSRC/final/lambda_dc';
else
    fpath = '/media/ajanthan/b7391340-f7ed-49ef-9dab-f3749bde5917/ajanthan/NICTA/Research/ubuntu_codes/data/densecrf/MSRC/final/Test_10_5_1000_0.1_1000_1000_1_tracing_dc';
end
paperfpath = '/media/ajanthan/b7391340-f7ed-49ef-9dab-f3749bde5917/ajanthan/NICTA/Research/ubuntu_docs/lp_densecrf/lp-densecrf-paper/images/msrc/dc_params';
img = '2_14_s';
fname = [fpath '/tracing-%s/' img '.trc'];
epsname = [fpath '/' img '_trc.eps'];
if (lambda == 1)
    if (zoom == 1)
        pdfname = [fpath '/' img '_lambda_trc_z.pdf'];
        paperpdfname = [paperfpath '/' img '_lambda_trc_z.pdf'];
    else
        pdfname = [fpath '/' img '_lambda_trc.pdf'];
        paperpdfname = [paperfpath '/' img '_lambda_trc.pdf'];
    end
else if (zoom == 1)
        pdfname = [fpath '/' img '_trc_z.pdf'];
        paperpdfname = [paperfpath '/' img '_trc_z.pdf'];
    else
        pdfname = [fpath '/' img '_trc.pdf'];
        paperpdfname = [paperfpath '/' img '_trc.pdf'];
    end
end

if (lambda == 1)
    algos = {'prox_lp_0.001', 'prox_lp_0.01', 'prox_lp_0.1', 'prox_lp_1', 'prox_lp_10'};
    names = {'PROX-LP_{0.001}', 'PROX-LP_{0.01}', 'PROX-LP_{0.1}', 'PROX-LP_{1}', 'PROX-LP_{10}'};
    mark = {'-co', '-b>', '-rs', '-g^', '-md'};
    color = {'c', 'b', 'r', 'g', 'm'};
else
    algos = {'mf', 'fixedDC-CCV', 'sg_lp_std', 'sg_lp', 'prox_lp', 'prox_lp_acc_l', 'prox_lp_acc'};
    names = {'MF', 'DC_{neg}', 'SG-LP', 'SG-LP_{l}', 'PROX-LP', 'PROX-LP_{l}', 'PROX-LP_{acc}'};
    mark = {'-c.', '-y+', '-kp', '-md', '-g^', '-b>', '-rs'};
    color = {'c', 'y', 'k', 'm', 'g', 'b', 'r'};
end

data = cell(length(algos), 1);
for i = 1 : length(algos)
    try
        data{i} = dlmread(sprintf(fname, algos{i}), '\t');
        time = data{i}(:,2);
        time = cumsum(time);
        plot(time, data{i}(:,3), mark{i}, 'LineWidth',2,...
            'MarkerEdgeColor',color{i},...
            'MarkerFaceColor',color{i},...
            'MarkerSize',6);
        hold all;
    catch
        sprintf(fname, algos{i})
    end
end

if (zoom == 1)
    if (lambda == 1)
        xlim([3,25]);
        ylim([1000000,1800000]);
        %     daspect([1 5000 2]);
    else
            xlim([1,3]);
            ylim([1100000,2500000]);
            daspect([1 400000 2]);
        
%             xlim([2,10]);
%             ylim([200000,1500000]);
%             daspect([1 100000 2]);
    end
else
    %title('Assignment energy as a function of time', 'FontSize', 15);
    ylabel('Integral energy', 'FontSize', 15);
    xlabel('Time (s)', 'FontSize', 15);
    if (lambda == 1)
        hleg = legend(names{1}, names{2}, names{3}, names{4}, names{5});
    else 
        hleg = legend(names{1}, names{2}, names{3}, names{4}, names{5}, names{6}, names{7});    
    end
    set(hleg,'Location','NorthEast','FontSize',15);
end

print('-depsc2', epsname);
eps2pdf(epsname, pdfname);
eps2pdf(epsname, paperpdfname);


