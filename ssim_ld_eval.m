path_rec = 'xxx';  % rectified image path
path_scan = './UDIR/gt/';  % scan image path

tarea=598400;
ms=0;
ld=0;

for i=1:195
    path_rec_1 = sprintf("%s%d%s", path_rec, i, '.png');  % rectified image path
    path_scan_new = sprintf("%s%d%s", path_scan, i, '.png');  % corresponding scan image path

    % imread and rgb2gray
    A1 = imread(path_rec_1);
    ref = imread(path_scan_new);
    A1 = rgb2gray(A1);
    ref = rgb2gray(ref);

    % resize
    b = sqrt(tarea/size(ref,1)/size(ref,2));
    ref = imresize(ref,b);
    ref_msk = ref;
    A1 = imresize(A1,[size(ref,1),size(ref,2)]);
    
    # mask the gt image
    m1 = A1 == 0;
    ref_msk(m1) = 0;
    
    % calculate
    [ms_1,ld_1] = evalUnwarp(A1, ref, ref_msk);
    ms = ms + ms_1;
    ld = ld + ld_1;
    
end

ms_m = ms / 195
ld_m = ld / 195
