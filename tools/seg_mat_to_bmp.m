function [] =  seg_mat_to_bmp(path_to_seg_mat, path_to_out_bmp)
% Example usage:
% seg_mat_to_bmp('/data/MSRC/newsegmentations_mats', '/data/MSRC/MSRC_ObjCategImageDatabase_v2/GroundTruthv2')

all_files = dir(path_to_seg_mat);

msrc_color_seg = [
    128 0 0 ,
    0 128 0 	 ,
    128 128 0 ,
    0 0 128 ,
    128 0 128 ,
    0 128 128 ,
    128 128 128 ,
    64 0 0 ,
    192 0 0 ,
    64 128 0 ,
    192 128 0 ,
    64 0 128 ,
    192 0 128 ,
    64 128 128 ,
    192 128 128 ,
    0 64 0 ,
    128 64 	0 ,
    0 192 0 ,
    128 64 	128 ,
    0 192 128 ,
    128 192 128 	 ,
    64 64 0 ,
    192 64	0,
    0 0 0];

msrc_color_seg = msrc_color_seg /255;

for image_id = 1:size(all_files, 1)
    if length(all_files(image_id).name) > 4 %&& strcmp(all_files(image_id).name(end-4:end), '.bmp')
        mat_file = [path_to_seg_mat '/' all_files(image_id).name];
        content = load(mat_file);
        content.newlabels = [content.newlabels 24];
        segim = content.newlabels(content.newseg);

        img = ind2rgb(segim, msrc_color_seg);
        out_file = [path_to_out_bmp '/' strrep(all_files(image_id).name, ...
                                           '.mat', '.bmp')]
        imwrite(img, out_file);
    end
end