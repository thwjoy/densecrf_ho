%% generate the energy, accuracy, time table for the given dataset
clear all;
addpath('/media/ajanthan/sheep/Ajanthan/lp_densecrf/densecrf/tools/');
addpath('/media/ajanthan/sheep/Ajanthan/AJ/Codes/matrix2latex/');

dataset = 'Pascal2010';
dpath = '/media/ajanthan/sheep/Ajanthan/data/densecrf/';
folder = 'final2/Test_10_5_1000_0.1_1000_1000_1_final2_dc';
fpath = [dpath, dataset '/' folder '/'];

% algos = {'mf5', 'mf', 'fixedDC-CCV', 'sg_lp', 'prox_lp', 'prox_lp_acc_l', 'prox_lp_acc_p', 'prox_lp_acc'};
% names = {'MF5', 'MF', 'DC$_\text{neg}$', 'SG-LP$_\text{l}$', 'PROX-LP', 'PROX-LP$_\text{l}$', 'PROX-LP$_\text{p}$', 'PROX-LP$_\text{acc}$'};
algos = {'mf5', 'mf', 'fixedDC-CCV', 'sg_lp', 'prox_lp', 'prox_lp_acc_l', 'prox_lp_acc'};
names = {'MF5', 'MF', 'DC$_\text{neg}$', 'SG-LP$_\text{l}$', 'PROX-LP', 'PROX-LP$_\text{l}$', 'PROX-LP$_\text{acc}$'};
nalgos = length(algos);

exepath = '/media/ajanthan/sheep/Ajanthan/lp_densecrf/densecrf/alpha/scripts/';
exe = ['python ' exepath 'energies.py'];
out = [fpath 'energies.out'];

fid = fopen(out, 'w');
fclose(fid);
for i = 1 : nalgos
    for j = i+1 : nalgos
        first = [fpath algos{j} '/'];
        second = [fpath algos{i} '/'];
        fid = fopen(out, 'a');
        fprintf(fid, sprintf('\n### %s vs %s ###\n', algos{j}, algos{i}));
        fclose(fid);
        system(sprintf('%s %s %s >> %s', exe, first, second, out));
    end
end

avgtime = zeros(nalgos, 1);
avgenergy = zeros(nalgos, 1);
avgacc = zeros(nalgos, 1);
iou = zeros(nalgos, 1);
bemat = zeros(nalgos, nalgos);

% fid = fopen(out, 'r');
% timings = fscanf(fid, 'Average timings: %f vs %f');
% fclose(fid);

content = fileread(out) ;
tokens_t  = regexp(content, 'Average timings:');
tokens_e  = regexp(content, 'Average Integer energy:');
tokens_be  = regexp(content, 'First method reach lower integer energy in ');
timings = zeros(length(tokens_t), 2);
energies = zeros(length(tokens_e), 2);
betterenergies = zeros(length(tokens_be), 2);
for i = 1 : length(tokens_t)
    timings(i, :) = sscanf(content(tokens_t(i):end), 'Average timings: %f vs %f');
    energies(i, :) = sscanf(content(tokens_e(i):end), 'Average Integer energy: %f vs %f');
    betterenergies(i, :) = sscanf(content(tokens_be(i):end), 'First method reach lower integer energy in %f%% and same in %f%%');
end

avgtime(1) = timings(1, 2);
avgenergy(1) = energies(1, 2);
for i = 2 : nalgos
    avgtime(i) = timings(i-1, 1);
    avgenergy(i) = energies(i-1, 1);
end

count = 1;
for i = 1 : nalgos
    for j = i+1 : nalgos
        bemat(j, i) = betterenergies(count, 1);
        bemat(i, j) = 100 - (betterenergies(count, 1) + betterenergies(count, 2));
        count = count + 1;
    end
end

for i = 1 : nalgos
    path = [fpath algos{i} '/'];
    if (strcmp(dataset, 'MSRC'))
        [iou(i), avgacc(i)] = msrc_test(path, 'Test');
    else 
        [iou(i), avgacc(i)] = voc_test(path, 'Test');
    end
end

%% matrix to latex table
bemat = bemat - eye(nalgos, nalgos);

r = nalgos;
c = nalgos + 5;
mat = cell(r, c);
for i = 1 : r
    mat{i, 1} = names{i};
    for j = 2 : nalgos + 1
        mat{i, j} = comma_separated(bemat(i, j-1), '%10.0f');
    end
    mat{i, nalgos+2} = comma_separated(avgenergy(i)/1000, '%10.1f');
    mat{i, nalgos+3} = comma_separated(avgtime(i), '%10.1f');
    mat{i, nalgos+4} = comma_separated(avgacc(i), '%10.2f');
    mat{i, nalgos+5} = comma_separated(iou(i), '%10.2f');
end

columnLabels = cell(c);
matrix2latex(mat, sprintf('%s/mat2table.tex', fpath), 'columnLabels', columnLabels, 'alignment', 'c');

