 fid = fopen('sds_voc2012_trainval.txt');
 all_list = textscan(fid, '%s');
 all_list = all_list{1};
 fclose(fid);
 
 annots = dir('../SegmentationClassAug/*.png');
 annot_list = {annots.name};
 
 missing_list = {};
 
 for i = 1 : numel(all_list)
     %fprintf(1, 'processing %d (%d) ...\n', i, numel(annots));
     if ~ismember( [all_list{i} '.png'], annot_list)
         missing_list{end+1} = all_list{i};
         fprintf(1, '%s \n', all_list{i});
     end
 end
 
 %
fid = fopen('sds_voc2012_train.txt');
all_list = textscan(fid, '%s');
all_list = all_list{1};
fclose(fid);

fid = fopen('sds_voc2011_train.txt', 'w');
for i = 1 : numel(all_list)
    if ~ismember(all_list{i}, missing_list)
        fprintf(fid, '%s\n', all_list{i});
    end
end
fclose(fid);

fid = fopen('sds_voc2012_val.txt');
all_list = textscan(fid, '%s');
all_list = all_list{1};
fclose(fid);

fid = fopen('sds_voc2011_val.txt', 'w');
for i = 1 : numel(all_list)
    if ~ismember(all_list{i}, missing_list)
        fprintf(fid, '%s\n', all_list{i});
    end
end
fclose(fid);

fid = fopen('sds_voc2011_train.txt');
train_list = textscan(fid, '%s');
train_list = train_list{1};
fclose(fid);

fid = fopen('sds_voc2011_val.txt');
val_list = textscan(fid, '%s');
val_list = val_list{1};
fclose(fid);

all_list = [train_list; val_list];

fid = fopen('sds_voc2011_trainval.txt', 'w');
for i = 1 : numel(all_list)
    fprintf(fid, '%s\n', all_list{i});    
end
fclose(fid);
 