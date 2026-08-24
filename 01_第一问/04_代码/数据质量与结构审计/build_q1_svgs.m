function build_q1_svgs(plotDataDir, outputDir)
% 第一问数据质量与结构审计图。只读取确定性绘图表，不进行正式模型拟合。

if nargin < 1 || strlength(string(plotDataDir)) == 0
    error('必须提供MATLAB绘图数据目录。');
end
if nargin < 2 || strlength(string(outputDir)) == 0
    error('必须提供SVG输出目录。');
end
plotDataDir = char(plotDataDir);
outputDir = char(outputDir);
if ~isfolder(plotDataDir)
    error('绘图数据目录不存在：%s', plotDataDir);
end
if ~isfolder(outputDir)
    mkdir(outputDir);
end

fonts = listfonts;
if any(strcmpi(fonts, 'Noto Sans SC'))
    q1Font = 'Noto Sans SC';
elseif any(strcmp(fonts, '微软雅黑'))
    q1Font = '微软雅黑';
elseif any(strcmpi(fonts, 'Microsoft YaHei'))
    q1Font = 'Microsoft YaHei';
else
    error('缺少Noto Sans SC或微软雅黑字体，停止以避免中文乱码。');
end
set(groot, 'defaultAxesFontName', q1Font, 'defaultTextFontName', q1Font, ...
    'defaultTextInterpreter', 'none', 'defaultLegendInterpreter', 'none', ...
    'defaultAxesTickLabelInterpreter', 'none');

blue = [37, 99, 235] / 255;
orange = [234, 88, 12] / 255;
green = [22, 163, 74] / 255;
dark = [17, 24, 39] / 255;
gray = [107, 114, 128] / 255;

base = read_utf8_table(fullfile(plotDataDir, '绘图基础数据.csv'));
boxData = read_utf8_table(fullfile(plotDataDir, '图01_箱线摘要.csv'));
weekBins = read_utf8_table(fullfile(plotDataDir, '图03_孕周分箱中位数.csv'));
relations = read_utf8_table(fullfile(plotDataDir, '图04_个体内关系参数.csv'));
repeats = read_utf8_table(fullfile(plotDataDir, '图06_重复检测离散度明细.csv'));
corrData = read_utf8_table(fullfile(plotDataDir, '图07_相关结构摘要.csv'));
histData = read_utf8_table(fullfile(plotDataDir, '图08_日期孕周直方图.csv'));

batch = string(base.('分析批次（683断点口径）'));
isPre = batch == "683前";
isPost = batch == "683后";

%% 图1：两个批次的主要变量分布
fig = new_figure([100, 100, 1260, 430]);
metrics = ["解析检测孕周", "孕妇BMI", "Y染色体浓度"];
ylabels = ["检测孕周（周）", "BMI（kg/m²）", "Y染色体浓度（%）"];
for k = 1:3
    ax = subplot(1, 3, k, 'Parent', fig);
    hold(ax, 'on');
    rows = boxData(string(boxData.('指标')) == metrics(k), :);
    for j = 1:2
        row = rows(j, :);
        color = blue;
        if string(row.('分析批次（683断点口径）')) == "683后"
            color = orange;
        end
        x = j;
        q1 = row.('第一四分位数'); q3 = row.('第三四分位数'); med = row.('中位数');
        low = row.('下须值'); high = row.('上须值');
        patch(ax, [x-0.28 x+0.28 x+0.28 x-0.28], [q1 q1 q3 q3], color, ...
            'FaceAlpha', 0.5, 'EdgeColor', color, 'LineWidth', 1.2);
        plot(ax, [x-0.28 x+0.28], [med med], 'Color', dark, 'LineWidth', 1.5);
        plot(ax, [x x], [low q1], 'Color', color, 'LineWidth', 1.1);
        plot(ax, [x x], [q3 high], 'Color', color, 'LineWidth', 1.1);
        plot(ax, [x-0.12 x+0.12], [low low], 'Color', color, 'LineWidth', 1.1);
        plot(ax, [x-0.12 x+0.12], [high high], 'Color', color, 'LineWidth', 1.1);
        outlierText = string(row.('异常值列表（与指标同单位，分号分隔）'));
        if ~ismissing(outlierText) && strlength(outlierText) > 0
            outlierValues = str2double(split(outlierText, ';'));
            outlierValues = outlierValues(~isnan(outlierValues));
            scatter(ax, repmat(x, numel(outlierValues), 1), outlierValues, 20, ...
                'MarkerEdgeColor', color, 'LineWidth', 0.9, 'HandleVisibility', 'off');
        end
    end
    xlim(ax, [0.5 2.5]); xticks(ax, [1 2]); xticklabels(ax, {'683前','683后'});
    ylabel(ax, ylabels(k)); grid(ax, 'on'); ax.GridAlpha = 0.2;
end
sgtitle(fig, '图1  第683行前后主要变量分布');
export_svg(fig, fullfile(outputDir, '01_683前后数据分布.svg'));

%% 图2：孕周与BMI共同支持域
fig = new_figure([100, 100, 850, 580]); ax = axes(fig); hold(ax, 'on');
scatter(ax, base.('解析检测孕周（周）')(isPre), base.('孕妇BMI（kg/m²）')(isPre), 18, blue, 'filled', ...
    'MarkerFaceAlpha', 0.38, 'DisplayName', sprintf('683前（n=%d）', sum(isPre)));
scatter(ax, base.('解析检测孕周（周）')(isPost), base.('孕妇BMI（kg/m²）')(isPost), 18, orange, 'filled', ...
    'MarkerFaceAlpha', 0.38, 'DisplayName', sprintf('683后（n=%d）', sum(isPost)));
rectangle(ax, 'Position', [11, 28, 9, 8], 'EdgeColor', green, 'LineWidth', 1.5, 'LineStyle', '--');
text(ax, 16.0, 35.5, '密集公共域候选', 'Color', green, 'FontWeight', 'bold');
xlabel(ax, '检测孕周（周）'); ylabel(ax, 'BMI（kg/m²）');
title(ax, '图2  两批次孕周-BMI支持域'); legend(ax, 'Location', 'best'); grid(ax, 'on'); ax.GridAlpha = 0.2;
export_svg(fig, fullfile(outputDir, '02_孕周BMI共同支持域.svg'));

%% 图3：Y浓度与孕周
fig = new_figure([100, 100, 1260, 500]);
for j = 1:2
    ax = subplot(1, 2, j, 'Parent', fig); hold(ax, 'on');
    thisBatch = "683前"; mask = isPre; color = blue;
    if j == 2, thisBatch = "683后"; mask = isPost; color = orange; end
    scatter(ax, base.('解析检测孕周（周）')(mask), base.('Y染色体浓度（比例，0–1）')(mask) * 100, ...
        14, color, 'filled', 'MarkerFaceAlpha', 0.28, 'HandleVisibility', 'off');
    med = weekBins(string(weekBins.('分析批次（683断点口径）')) == thisBatch, :);
    plot(ax, med.('箱内孕周中位数（周）'), med.('箱内Y染色体浓度中位数（%）'), ...
        '-o', 'Color', dark, 'LineWidth', 1.7, 'MarkerSize', 4, 'DisplayName', '分箱中位数');
    xlabel(ax, '检测孕周（周）'); ylabel(ax, 'Y染色体浓度（%）');
    title(ax, sprintf('%s（%d条）', thisBatch, sum(mask))); legend(ax, 'Location', 'best'); grid(ax, 'on'); ax.GridAlpha = 0.2;
end
sgtitle(fig, '图3  Y染色体浓度与孕周（按批次）');
export_svg(fig, fullfile(outputDir, '03_分批次孕周与Y浓度.svg'));

%% 图4：个体内BMI与Y浓度
fig = new_figure([100, 100, 1260, 500]);
for j = 1:2
    ax = subplot(1, 2, j, 'Parent', fig); hold(ax, 'on');
    thisBatch = "683前"; mask = isPre; color = blue;
    if j == 2, thisBatch = "683后"; mask = isPost; color = orange; end
    x = base.('个体内BMI偏差（kg/m²）')(mask);
    y = base.('个体内Y染色体浓度偏差（比例）')(mask) * 100;
    scatter(ax, x, y, 15, color, 'filled', 'MarkerFaceAlpha', 0.32);
    rel = relations(string(relations.('分析批次（683断点口径）')) == thisBatch, :);
    xx = linspace(min(x), max(x), 100);
    plot(ax, xx, rel.('过原点斜率') * xx * 100, 'Color', dark, 'LineWidth', 2);
    xlabel(ax, '个体内BMI偏差（kg/m²）'); ylabel(ax, '个体内Y浓度偏差（百分点）');
    title(ax, sprintf('%s：斜率=%.8f，R²=%.6f', thisBatch, rel.('过原点斜率'), rel.('决定系数R²')));
    grid(ax, 'on'); ax.GridAlpha = 0.2;
end
sgtitle(fig, '图4  个体内BMI变化与Y浓度变化');
export_svg(fig, fullfile(outputDir, '04_分批次个体内BMI与Y浓度.svg'));

%% 图5：读段关系断裂，R2021b使用显式symlog变换
fig = new_figure([100, 100, 1000, 520]); ax = axes(fig); hold(ax, 'on');
rawResidual = base.('唯一比对读段逻辑残差（条）'); transformed = symlog_forward(rawResidual, 10);
hPre = scatter(ax, base.('序号')(isPre), transformed(isPre), 14, blue, 'filled', 'MarkerFaceAlpha', 0.45);
hPost = scatter(ax, base.('序号')(isPost), transformed(isPost), 14, orange, 'filled', 'MarkerFaceAlpha', 0.45);
hBreak = xline(ax, 682.5, '--', '序号683断点', 'Color', dark, 'LineWidth', 1.4);
hZero = yline(ax, 0, '-', 'Color', gray);
hBreak.HandleVisibility = 'off'; hZero.HandleVisibility = 'off';
tickRaw = make_symlog_ticks(rawResidual);
yticks(ax, symlog_forward(tickRaw, 10)); yticklabels(ax, compose('%g', tickRaw));
xlabel(ax, '样本序号'); ylabel(ax, '唯一读段实际值－逻辑计算值（条，symlog）');
title(ax, '图5  读段内部逻辑关系在序号683处断裂'); legend(ax, [hPre hPost], {'683前','683后'}, 'Location', 'best'); grid(ax, 'on'); ax.GridAlpha = 0.2;
export_svg(fig, fullfile(outputDir, '05_测序读段关系断裂.svg'));

%% 图6：同抽血编号与同日检测会话的双口径离散度
fig = new_figure([100, 100, 1260, 520]);
definitions = ["同一抽血编号（B+I）", "同日检测会话（B+I+H）"];
panelTitles = ["同一抽血编号内多次检测", "同日检测会话内重复记录"];
panelColors = [blue; orange];
for j = 1:2
    ax = subplot(1, 2, j, 'Parent', fig); hold(ax, 'on');
    subset = repeats(string(repeats.('复测口径')) == definitions(j), :);
    orders = unique(subset.('复测组绘图顺序'))';
    for order = orders
        values = subset.('Y染色体浓度（%）')(subset.('复测组绘图顺序') == order);
        color = panelColors(j, :);
        plot(ax, repmat(order, size(values)), values, 'o', 'Color', color, 'MarkerFaceColor', color, 'MarkerSize', 4);
        plot(ax, [order order], [min(values) max(values)], '-', 'Color', 0.55 * color + 0.45, 'LineWidth', 1);
    end
    yline(ax, 4, '--', '4%参考线', 'Color', [220 38 38]/255, 'LineWidth', 1.3);
    xlabel(ax, '多记录组（按组均值排序）'); ylabel(ax, 'Y染色体浓度（%）');
    title(ax, sprintf('%s：%d组', panelTitles(j), numel(orders))); grid(ax, 'on'); ax.GridAlpha = 0.2;
end
sgtitle(fig, '图6  复测离散度的两种操作性定义');
export_svg(fig, fullfile(outputDir, '06_重复检测离散度.svg'));

%% 图7：相关结构分解
fig = new_figure([100, 100, 950, 550]); ax = axes(fig);
barColors = [gray; blue; orange; gray; blue; orange];
hold(ax, 'on');
for k = 1:6
    bar(ax, k, corrData.('Pearson相关系数')(k), 0.8, 'FaceColor', barColors(k,:), 'EdgeColor', 'none');
end
yline(ax, 0, '-', 'Color', dark); grid(ax, 'on'); ax.GridAlpha = 0.2;
labels = string(corrData.('变量')) + "-" + string(corrData.('相关结构层级'));
xticks(ax, 1:6); xticklabels(ax, labels); xtickangle(ax, 25);
ylabel(ax, 'Pearson相关系数'); title(ax, '图7  逐行、个体间与个体内相关并不等价');
export_svg(fig, fullfile(outputDir, '07_相关结构分解.svg'));

%% 图8：日期推算孕周一致性
fig = new_figure([100, 100, 1200, 520]);
for j = 1:2
    ax = subplot(1, 2, j, 'Parent', fig); hold(ax, 'on');
    thisBatch = "683前"; color = blue;
    if j == 2, thisBatch = "683后"; color = orange; end
    rows = histData(string(histData.('分析批次（683断点口径）')) == thisBatch, :);
    centers = (rows.('分箱左边界（天）') + rows.('分箱右边界（天）')) / 2;
    widths = rows.('分箱右边界（天）') - rows.('分箱左边界（天）');
    bar(ax, centers, rows.('记录数（条）'), 1, 'FaceColor', color, 'EdgeColor', 'none');
    if numel(widths) == 1, xlim(ax, [-1 1]); end
    xline(ax, 0, '-', 'Color', dark); xline(ax, -14, '--', 'Color', [220 38 38]/255); xline(ax, 14, '--', 'Color', [220 38 38]/255);
    totalCount = rows.('批次总记录数（条）')(1);
    validCount = rows.('有效日期孕周差记录数（条）')(1);
    missingCount = rows.('日期孕周差缺失记录数（条）')(1);
    xlabel(ax, '日期孕周－J列孕周（天）'); ylabel(ax, '记录数');
    title(ax, sprintf('%s（有效n=%d，缺失%d）', thisBatch, validCount, missingCount));
    grid(ax, 'on'); ax.GridAlpha = 0.2;
    if thisBatch == "683后"
        text(ax, 0, max(rows.('记录数（条）')) * 0.92, sprintf('%d/%d条有效差值均为0天', validCount, totalCount), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'top');
    end
end
sgtitle(fig, '图8  末次月经与检测日期推算孕周的一致性');
export_svg(fig, fullfile(outputDir, '08_日期孕周一致性.svg'));

%% 机器可读的SVG校验记录
expectedFiles = {
    '01_683前后数据分布.svg'; '02_孕周BMI共同支持域.svg'; '03_分批次孕周与Y浓度.svg';
    '04_分批次个体内BMI与Y浓度.svg'; '05_测序读段关系断裂.svg'; '06_重复检测离散度.svg';
    '07_相关结构分解.svg'; '08_日期孕周一致性.svg'};
records = struct('file_name', {}, 'bytes', {}, 'contains_raster_image', {});
for k = 1:numel(expectedFiles)
    file = fullfile(outputDir, expectedFiles{k}); info = dir(file); xml = fileread(file);
    assert(~isempty(info) && info.bytes > 1000, 'SVG为空或过小：%s', file);
    records(k).file_name = expectedFiles{k}; %#ok<AGROW>
    records(k).bytes = info.bytes;
    records(k).contains_raster_image = contains(lower(xml), '<image');
    assert(~records(k).contains_raster_image, 'SVG包含非预期栅格图层：%s', file);
end
verification = struct('matlab_version', version, 'font', q1Font, 'svg_count', numel(expectedFiles), ...
    'all_vector', all(~[records.contains_raster_image]), 'files', records);
fid = fopen(fullfile(outputDir, 'SVG制图校验.json'), 'w', 'n', 'UTF-8');
assert(fid >= 0, '无法创建SVG制图校验文件。');
fprintf(fid, '%s', jsonencode(verification)); fclose(fid);
fprintf('MATLAB SVG制图完成：%d/8，全部为纯矢量。\n', numel(expectedFiles));
end


function tableData = read_utf8_table(file)
opts = detectImportOptions(file, 'Delimiter', ',', 'Encoding', 'UTF-8', 'VariableNamingRule', 'preserve');
tableData = readtable(file, opts);
end


function fig = new_figure(position)
fig = figure('Visible', 'off', 'Color', 'white', 'Renderer', 'painters', 'Position', position, 'InvertHardcopy', 'off');
end


function export_svg(fig, outputFile)
drawnow;
print(fig, outputFile, '-dsvg', '-painters');
close(fig);
info = dir(outputFile);
assert(~isempty(info) && info.bytes > 1000, 'SVG导出失败：%s', outputFile);
end


function transformed = symlog_forward(values, linthresh)
values = double(values);
scale = 1 / (1 - 0.1);
transformed = values * scale;
outside = abs(values) > linthresh;
transformed(outside) = sign(values(outside)) .* linthresh .* ...
    (scale + log10(abs(values(outside)) / linthresh));
end


function ticks = make_symlog_ticks(values)
maxAbs = max(abs(values), [], 'omitnan');
power = max(1, ceil(log10(maxAbs)));
positive = 10 .^ (1:power);
ticks = [-fliplr(positive), 0, positive];
end
