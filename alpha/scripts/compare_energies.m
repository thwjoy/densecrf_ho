%% compare energies computed by brute-force and permutohedral
%% all files in the directory
% fpath = '/media/ajanthan/b7391340-f7ed-49ef-9dab-f3749bde5917/ajanthan/NICTA/Research/ubuntu_codes/data/toy/';
% flist = dir(sprintf('%s%s', fpath, '*.out'));
% for i=1:length(flist)
%     fullname = sprintf('%s%s', fpath, flist(i).name);
%     [fname, ext] = strtok(flist(i).name, '.');
%     fname
%     
%     toks = strsplit(fname, '_');
%     
%     fid = fopen(fullname);
%     data = textscan(fid,'%f,%f,%f');
%     fclose(fid);
%     
%     nb_ones = data{1,1}(2:end-1);   % don't show all zero and all ones for better plot
%     ph = data{1,2}(2:end-1);
%     bf = data{1,3}(2:end-1);
%     
%     semilogy(nb_ones, ph, '-bo', 'LineWidth',2,...
%         'MarkerEdgeColor','b',...
%         'MarkerFaceColor','b',...
%         'MarkerSize',5);
%     
%     hold all
%     semilogy(nb_ones, bf, '-rs', 'LineWidth',2,...
%         'MarkerEdgeColor','r',...
%         'MarkerFaceColor','r',...
%         'MarkerSize',5);
%     
%     bf_over_ph = bf ./ ph;
%     hold all
%     semilogy(nb_ones, bf_over_ph, '-g+', 'LineWidth',2,...
%         'MarkerEdgeColor','g',...
%         'MarkerFaceColor','g',...
%         'MarkerSize',5);
%     
%     
%     strtitle = sprintf('Energy comparison (%sx%s, %s, %s, PH-%s)', toks{4}, toks{5}, toks{6}, toks{7}, toks{8});
%     title(strtitle, 'FontSize', 15);
%     ylabel('Energy', 'FontSize', 15);
%     xlabel('No of ones', 'FontSize', 15);
%     hleg = legend(sprintf('PH-%s', toks{8}), 'Brute-force', 'BF/PH');
%     set(hleg,'Location','Best', 'FontSize', 15)
%     
%     print('-depsc2', sprintf('%s%s.eps', fpath, fname));
%     eps2pdf(sprintf('%s%s.eps', fpath, fname), sprintf('%s%s.pdf', fpath, fname));
%     %break;
%     close all;
% end


%% single file
fpath = '/media/ajanthan/b7391340-f7ed-49ef-9dab-f3749bde5917/ajanthan/NICTA/Research/ubuntu_codes/lp_densecrf/densecrf/build/';
filename = 'toy_bf_ph_50_50_5_lp_new.out';
fullname = sprintf('%s%s', fpath, filename);
[fname, ext] = strtok(filename, '.');
fname

toks = strsplit(fname, '_');

fid = fopen(fullname);
data = textscan(fid,'%f,%f,%f');
fclose(fid);

nb_ones = data{1,1}(2:end-1);   % don't show all zero and all ones for better plot
ph = data{1,2}(2:end-1);
bf = data{1,3}(2:end-1);

semilogy(nb_ones, ph, '-bo', 'LineWidth',2,...
    'MarkerEdgeColor','b',...
    'MarkerFaceColor','b',...
    'MarkerSize',5);

hold all
semilogy(nb_ones, bf, '-rs', 'LineWidth',2,...
    'MarkerEdgeColor','r',...
    'MarkerFaceColor','r',...
    'MarkerSize',5);

bf_over_ph = bf ./ ph;
hold all
semilogy(nb_ones, bf_over_ph, '-g+', 'LineWidth',2,...
    'MarkerEdgeColor','g',...
    'MarkerFaceColor','g',...
    'MarkerSize',5);


strtitle = sprintf('Energy comparison (%sx%s, %s, %s, PH-%s)', toks{4}, toks{5}, toks{6}, toks{7}, toks{8});
title(strtitle, 'FontSize', 15);
ylabel('Energy', 'FontSize', 15);
xlabel('No of ones', 'FontSize', 15);
hleg = legend(sprintf('PH-%s', toks{8}), 'Brute-force', 'BF/PH');
set(hleg,'Location','Best', 'FontSize', 15)

print('-depsc2', sprintf('%s%s.eps', fpath, fname));
eps2pdf(sprintf('%s%s.eps', fpath, fname), sprintf('%s%s.pdf', fpath, fname));
