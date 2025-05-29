All_Type = {'Angle','BendedLine','CShape','DoubleBendedLine','GShape','heee','JShape','JShape_2','Khamesh','Leaf_1','Leaf_2','Line','LShape','NShape','PShape','RShape','Saeghe','Sharpc','Sine','Snake','Spoon','Sshape','Trapezoid','Worm','WShape','Zshape'};
currentPath = pwd; % 获取当前工作目录路径
cnt = 3;
for j =1:26
    figure;
    filename = [All_Type{j},'.mat'];
    disp(filename)
    data = load(filename);
    ori_data = data.odata;
    new_data = data.data;
    x = data.px;
    y = data.py;
    z = data.pz;
    lx = data.lx;
    ly = data.ly;
    lu = data.lu;
    lv = data.lv;
    contourf(x, y, z, 400, 'LineStyle', 'none'); % 填充等高线图，50 个等级
    hold on;
    for i = 1:7
       x1 = squeeze(new_data(i,:,:))';
       x2 = squeeze(ori_data(i,:,:))';  
       h1 = plot(x2(1, :), x2(2, :), 'w-', 'LineWidth', 3.5);
       hold on;
    end
    for i = 1:7
       x1 = squeeze(new_data(i,:,:))';
       x2 = squeeze(ori_data(i,:,:))';  
       h2 = plot(x1(1, :), x1(2, :), 'r-', 'LineWidth', 3.5);
       hold on;
    end  
    plot(0, 0, 'k*', 'MarkerSize', 20, 'LineWidth', 2.5);
    box on;
    saveas(gcf, ['mtlab_plot_e', All_Type{j},'.png']);
    close(gcf);
    
    figure;
    streamslice(lx, ly, lu, lv, 2);
    hold on;
    xlim([min(min(lx)), max(max(lx))]);
    ylim([min(min(ly)), max(max(ly))]);
    plot(0, 0, 'k*', 'MarkerSize', 20, 'LineWidth', 2.5);
    box on;
    hold on;
    saveas(gcf, ['mtlab_plot_v', All_Type{j},'.png']);
    close(gcf);
end

