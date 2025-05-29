All_Type = {'Angle','BendedLine','CShape','DoubleBendedLine','GShape','heee','JShape','JShape_2','Khamesh','Leaf_1','Leaf_2','Line','LShape','NShape','PShape','RShape','Saeghe','Sharpc','Sine','Snake','Spoon','Sshape','Trapezoid','Worm','WShape','Zshape'};

currentPath = pwd; % 获取当前工作目录路径
[parentPath, ~, ~] = fileparts(currentPath); % 使用 fileparts 获取上一级目录路径
[parentPath, ~, ~] = fileparts(parentPath);
disp(parentPath); % 显示上一级目录路径
[numData, textData, raw] = xlsread([parentPath,'\data.xlsx']);
cnt = 3;
for i =1:26
    filename = [All_Type{i},'.mat'];
    disp(filename)
    data = load(filename);
    ori_data = data.odata;
    new_data = data.data;
    sum_all_ase = 0;
    for i = 1:7
       x1 = squeeze(new_data(i,:,:))';
       x2 = squeeze(ori_data(i,:,:))';
       sum_all_ase = sum_all_ase + swept_area_error(x1, x2);
       textData{cnt, 5} = num2str(swept_area_error(x1, x2));
       cnt = cnt + 1;
       disp(sum_all_ase);  
    end
end
xlswrite([parentPath,'\data.xlsx'], textData)

