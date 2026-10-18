clear; clc; close all;

% Prevent MATLAB from rendering underscores as subscripts in labels.
set(groot, 'defaultTextInterpreter','none');
set(groot, 'defaultAxesTickLabelInterpreter','none');
set(groot, 'defaultLegendInterpreter','none');

%% Paths ---------------------------------------------------------------
resultsDir = fullfile(pwd, 'results');
if ~isfolder(resultsDir), resultsDir = pwd; end

outDir = fullfile(pwd, 'figures');
if ~exist(outDir, 'dir'), mkdir(outDir); end

fprintf('Reading CSVs from: %s\n', resultsDir);
fprintf('Saving figures to: %s\n\n', outDir);

%% Colours -------------------------------------------------------------
COL_RF   = [46  134 171] / 255;   % blue
COL_GB   = [162 59  114] / 255;   % magenta
COL_SVM  = [241 143 1  ] / 255;   % orange (replaces the old kNN colour)
COL_SHAP = [27  158 119] / 255;   % green
COL_LIME = [217 95  2  ] / 255;   % orange-red
COL_CV   = [76  114 176] / 255;
COL_TEST = [221 132 82 ] / 255;
COL_AGREE = [1.0 0.95 0.6];        % light yellow for "both methods agree"

modelColors = {COL_RF, COL_GB, COL_SVM};

%% Load data -----------------------------------------------------------
testTbl  = readtable(fullfile(resultsDir, 'binary_test_metrics.csv'));
cvTbl    = readtable(fullfile(resultsDir, 'binary_cv_metrics.csv'));
stabTbl  = readtable(fullfile(resultsDir, 'lime_stability.csv'));
timeTbl  = readtable(fullfile(resultsDir, 'timing.csv'));
agreementFile = fullfile(resultsDir, 'agreement_summary_main.csv');
assert(isfile(agreementFile), ...
    ['agreement_summary_main.csv is missing. Rerun main.m with the ' ...
     'top-k Jaccard primary-agreement analysis before making figures.']);
agreeTbl = readtable(agreementFile);
requiredAgreementVars = {'Model','TopK_Jaccard_mean', ...
    'TopK_Jaccard_CI_Lower','TopK_Jaccard_CI_Upper', ...
    'SecondarySpearman_rho_mean','LIME_Nonzero_median'};
assert(all(ismember(requiredAgreementVars, agreeTbl.Properties.VariableNames)), ...
    'agreement_summary_main.csv does not use the expected Scheme A schema.');
requiredTimingVars = {'Model','Method','MeanSeconds','CI95Lower','CI95Upper', ...
    'N_CasesAttempted','N_CompleteCases','N_RepeatsPerCase','N_Warmup'};
assert(all(ismember(requiredTimingVars, timeTbl.Properties.VariableNames)), ...
    ['timing.csv uses the old single-run schema. Rerun main.m to generate ' ...
    'the repeated-measurement timing outputs before running make_figures.m.']);
timeTbl.Model = string(timeTbl.Model);
timeTbl.Method = string(timeTbl.Method);

% Per-attack-category agreement (optional - skip fig8 if missing)
attackAgreeFile = fullfile(resultsDir, 'agreement_per_attack_category.csv');
hasAttackAgree = isfile(attackAgreeFile);
if hasAttackAgree
    attackAgreeTbl = readtable(attackAgreeFile);
    if isstring(attackAgreeTbl.Model) || iscell(attackAgreeTbl.Model)
        attackAgreeTbl.Model = string(attackAgreeTbl.Model);
    end
    if isstring(attackAgreeTbl.Family) || iscell(attackAgreeTbl.Family)
        attackAgreeTbl.Family = string(attackAgreeTbl.Family);
    end
end

modelOrder = ["RandomForest","BoostedTrees","SVM"];
modelLabels = {'Random Forest','Boosted Trees','SVM'};

% Per-model feature freq tables
featTbl = struct();
for i = 1:numel(modelOrder)
    fname = fullfile(resultsDir, sprintf('feature_top10_frequency_%s.csv', modelOrder(i)));
    t = readtable(fname);
    if isstring(t.Feature) || iscell(t.Feature)
        t.Feature = string(t.Feature);
    end
    featTbl.(matlab.lang.makeValidName(modelOrder(i))) = t;
end

% Normalise types
testTbl.Model  = string(testTbl.Model);
cvTbl.Model    = string(cvTbl.Model);
stabTbl.Model  = string(stabTbl.Model);
timeTbl.Model  = string(timeTbl.Model);
agreeTbl.Model = string(agreeTbl.Model);

%% =====================================================================
%  FIGURE 1 - Test-set performance (3 models x 5 metrics)
%% =====================================================================
fig1 = figure('Color','w','Position',[100 100 950 520]);
metrics = {'Accuracy','Precision','Recall','F1','AUC'};
[~, ord] = ismember(modelOrder, testTbl.Model);
vals = zeros(numel(metrics), numel(modelOrder));
for j = 1:numel(modelOrder)
    for k = 1:numel(metrics)
        vals(k, j) = testTbl{ord(j), metrics{k}};
    end
end

b = bar(1:numel(metrics), vals, 'grouped', 'EdgeColor','k', 'LineWidth',0.5);
for j = 1:numel(modelOrder), b(j).FaceColor = modelColors{j}; end

for j = 1:numel(modelOrder)
    text(b(j).XEndPoints, b(j).YEndPoints + 0.012, compose('%.3f', b(j).YEndPoints), ...
        'HorizontalAlignment','center', 'FontSize',8);
end

set(gca, 'XTick',1:numel(metrics), 'XTickLabel',metrics);
ylim([0 1.10]);
ylabel('Score');
title('Test-set performance: Random Forest, Boosted Trees, SVM');
legend(modelLabels, 'Location','southwest', 'Box','on');
grid on; set(gca,'GridAlpha',0.3, 'Layer','top');
exportgraphics(fig1, fullfile(outDir, 'fig1_model_comparison.png'), 'Resolution',300);
fprintf('  Saved fig1_model_comparison.png\n');

%% =====================================================================
%  FIGURE 2 - CV vs Test gap (generalisation)
%% =====================================================================
fig2 = figure('Color','w','Position',[100 100 900 540]);
[~, ordCV] = ismember(modelOrder, cvTbl.Model);
cvAcc  = cvTbl.CV_Acc_Mean(ordCV);
cvStd  = cvTbl.CV_Acc_Std(ordCV);
teAcc  = testTbl.Accuracy(ord);

x = 1:numel(modelOrder); w = 0.35;

hold on;
b1 = bar(x - w/2, cvAcc, w, 'FaceColor', COL_CV,   'EdgeColor','k', 'LineWidth',0.5);
b2 = bar(x + w/2, teAcc, w, 'FaceColor', COL_TEST, 'EdgeColor','k', 'LineWidth',0.5);
errorbar(x - w/2, cvAcc, cvStd, '.k', 'CapSize',6, 'LineWidth',1);

% Bar value labels
for i = 1:numel(modelOrder)
    text(x(i)-w/2, cvAcc(i)+0.025, sprintf('%.3f', cvAcc(i)), ...
        'HorizontalAlignment','center', 'FontSize',10, 'FontWeight','bold');
    text(x(i)+w/2, teAcc(i)-0.04, sprintf('%.3f', teAcc(i)), ...
        'HorizontalAlignment','center', 'FontSize',10, 'FontWeight','bold','Color','w');
end

% Gap bracket outside the pair
for i = 1:numel(modelOrder)
    cv = cvAcc(i); te = teAcc(i); gap = cv - te;
    xR = x(i) + w/2 + 0.06;
    plot([xR xR], [te cv], '-', 'Color', [0.85 0.1 0.1], 'LineWidth', 1.6);
    plot([xR-0.04 xR], [cv cv], '-', 'Color', [0.85 0.1 0.1], 'LineWidth', 1.6);
    plot([xR-0.04 xR], [te te], '-', 'Color', [0.85 0.1 0.1], 'LineWidth', 1.6);
    text(xR + 0.04, (cv+te)/2, sprintf('-%.1fpp', gap*100), ...
        'Color',[0.85 0.1 0.1], 'FontWeight','bold', 'FontSize',11, ...
        'VerticalAlignment','middle');
end

set(gca, 'XTick',x, 'XTickLabel',modelLabels);
xlim([0.4, numel(modelOrder)+0.7]);
ylim([0 1.18]);
ylabel('Accuracy');
title({'Generalisation gap: 5-fold CV vs held-out test set', ...
       '(KDDTest+ contains attack types absent from KDDTrain+)'});
legend([b1, b2], {'5-fold CV (training)', 'Held-out Test set'}, ...
    'Location','southwest', 'Box','on');
grid on; set(gca,'GridAlpha',0.3, 'Layer','top');
hold off;
exportgraphics(fig2, fullfile(outDir, 'fig2_cv_vs_test.png'), 'Resolution',300);
fprintf('  Saved fig2_cv_vs_test.png\n');

%% =====================================================================
%  FIGURE 3 - Primary per-sample top-k Jaccard agreement [RQ1, H3]
%% =====================================================================
fig3 = figure('Color','w','Position',[100 100 820 540]);
[~, ordA] = ismember(modelOrder, agreeTbl.Model);
jacMean = agreeTbl.TopK_Jaccard_mean(ordA);
jacLo   = agreeTbl.TopK_Jaccard_CI_Lower(ordA);
jacHi   = agreeTbl.TopK_Jaccard_CI_Upper(ordA);

x = 1:numel(modelOrder);
hold on;
plot([0.5 numel(modelOrder)+0.5], [0 0], '-', ...
    'Color',[0.35 0.35 0.35], 'LineWidth',1.0);

% Point and 95% CI display. No empirical agreement threshold is imposed.
for i = 1:numel(modelOrder)
    plot([x(i) x(i)], [jacLo(i) jacHi(i)], '-', ...
        'Color',[0.20 0.20 0.20], 'LineWidth',2.0);
    plot([x(i)-0.07 x(i)+0.07], [jacLo(i) jacLo(i)], '-', ...
        'Color',[0.20 0.20 0.20], 'LineWidth',1.5);
    plot([x(i)-0.07 x(i)+0.07], [jacHi(i) jacHi(i)], '-', ...
        'Color',[0.20 0.20 0.20], 'LineWidth',1.5);
    scatter(x(i), jacMean(i), 130, modelColors{i}, 'filled', ...
        'MarkerEdgeColor','k', 'LineWidth',0.7);
    text(x(i), min(jacHi(i)+0.05, 0.96), ...
        sprintf('%.3f\n[%.3f, %.3f]', jacMean(i), jacLo(i), jacHi(i)), ...
        'HorizontalAlignment','center', 'VerticalAlignment','bottom', ...
        'FontSize',10, 'FontWeight','bold');
end

set(gca, 'XTick',x, 'XTickLabel',modelLabels);
xlim([0.5, numel(modelOrder)+0.5]);
ylim([-0.05, 1]);
ylabel('Top-k Jaccard similarity');
title({'Attack-class top-attribution agreement by classifier', ...
       'Per-sample Jaccard of up to 10 non-zero |SHAP| and |LIME| features; mean and 95% CI'});
grid on; set(gca,'GridAlpha',0.25, 'Layer','top');
hold off;
ax3 = gca;
ax3.Toolbar = [];
exportgraphics(fig3, fullfile(outDir, 'fig3_top10_jaccard_per_model.png'), 'Resolution',300);
exportgraphics(fig3, fullfile(outDir, 'fig3_spearman_per_model.png'), 'Resolution',300);
fprintf('  Saved fig3_top10_jaccard_per_model.png (and legacy fig3 filename)\n');

%% =====================================================================
%  FIGURE 4 - SHAP vs LIME side-by-side, one row per model  [RQ1]
%% =====================================================================
fig4 = figure('Color','w','Position',[100 100 1100 1500]);
tl4 = tiledlayout(fig4, 3, 1, 'TileSpacing','compact', 'Padding','loose');
ax4 = gobjects(numel(modelOrder), 1);

for m = 1:numel(modelOrder)
    sub = featTbl.(matlab.lang.makeValidName(modelOrder(m)));

    % Build the union of up to ten materially non-zero aggregate features.
    % This avoids padding a sparse LIME list with arbitrary zero-count rows.
    shapTopFeat = positiveTopFeatures(sub, 'SHAP_top10_attack', 10);
    limeTopFeat = positiveTopFeatures(sub, 'LIME_top10_attack', 10);
    unionFeat = unique([shapTopFeat; limeTopFeat], 'stable');

    s2 = sub(ismember(sub.Feature, unionFeat), :);
    s2 = sortrows(s2, {'SHAP_top10_attack','LIME_top10_attack'}, {'descend','descend'});
    s2 = flipud(s2);
    inBoth = ismember(s2.Feature, intersect(shapTopFeat, limeTopFeat));

    ax4(m) = nexttile(tl4, m);
    y = 1:height(s2);  h = 0.4;

    hold on;
    for i = 1:numel(y)
        if inBoth(i)
            patch([0 23 23 0], [y(i)-0.5 y(i)-0.5 y(i)+0.5 y(i)+0.5], ...
                COL_AGREE, 'EdgeColor','none', 'FaceAlpha',0.5);
        end
    end

    b1 = barh(y - h/2, s2.SHAP_top10_attack, h, ...
        'FaceColor', COL_SHAP, 'EdgeColor','k', 'LineWidth',0.5);
    b2 = barh(y + h/2, s2.LIME_top10_attack, h, ...
        'FaceColor', COL_LIME, 'EdgeColor','k', 'LineWidth',0.5);

    for i = 1:height(s2)
        sV = s2.SHAP_top10_attack(i);  lV = s2.LIME_top10_attack(i);
        if sV > 0, text(sV + 0.3, y(i)-h/2, num2str(sV), 'FontSize',7, 'VerticalAlignment','middle'); end
        if lV > 0, text(lV + 0.3, y(i)+h/2, num2str(lV), 'FontSize',7, 'VerticalAlignment','middle'); end
    end

    set(gca, 'YTick',y, 'YTickLabel',s2.Feature, 'FontSize',9);
    xlim([0 23]);
    ylim([0.3, height(s2)+0.7]);
    if m == 3
        xlabel('# of attack samples (out of 20) where feature is in the non-zero top set');
    end

    inter = numel(intersect(shapTopFeat, limeTopFeat));
    uni   = numel(union(shapTopFeat, limeTopFeat));
    jac   = inter / max(uni,1);
    title(sprintf('%s   (aggregate top-set overlap = %d / %d, Jaccard = %.3f)', ...
        modelLabels{m}, inter, uni, jac), 'FontSize',11);

    % Save handles from row 1 for a single figure-level legend below
    if m == 1
        b1_ref = b1;  b2_ref = b2;
        hAgree_ref = patch(NaN, NaN, COL_AGREE, 'EdgeColor','none');
    end
    grid on; set(gca,'GridAlpha',0.3, 'Layer','top');
    hold off;
end

sgtitle(tl4, 'Attack-class SHAP and LIME top-attribution frequencies (up to 10 non-zero features/sample)', ...
    'FontSize',13, 'FontWeight','bold');

% One legend for the whole figure in a layout-managed tile.
legFig = legend(ax4(1), [b1_ref, b2_ref, hAgree_ref], ...
                {'SHAP','LIME','In both aggregate top sets'}, ...
                'Orientation','horizontal', 'NumColumns',3, ...
                'FontSize',10, 'Box','on');
legFig.Layout.Tile = 'south';
tl4.Toolbar = [];

exportgraphics(fig4, fullfile(outDir, 'fig4_shap_vs_lime_per_model.png'), 'Resolution',300);
fprintf('  Saved fig4_shap_vs_lime_per_model.png\n');

%% =====================================================================
%  FIGURE 5 - LIME repeated-run stability                [RQ2, H1]
%% =====================================================================
fig5 = figure('Color','w','Position',[100 100 850 540]);

x = 1:numel(modelOrder);
allCv = stabTbl.CoeffOfVar_top10;
yMax = max(allCv, [], 'omitnan');
if isempty(yMax) || ~isfinite(yMax), yMax = 1; end
yUpper = max(1, yMax * 1.20);
hold on;

% LIME data only: five attack samples per model and their mean.
for i = 1:numel(modelOrder)
    rows = stabTbl(stabTbl.Model == modelOrder(i), :);
    cv_vals = rows.CoeffOfVar_top10;
    if numel(cv_vals) == 1
        jitter = 0;
    else
        jitter = linspace(-0.12, 0.12, numel(cv_vals))';
    end
    scatter(x(i) + jitter, cv_vals, 110, ...
        'MarkerFaceColor', modelColors{i}, 'MarkerEdgeColor','k', ...
        'LineWidth',0.5, 'MarkerFaceAlpha',0.85);
    cvMean = mean(cv_vals, 'omitnan');
    plot([x(i)-0.22, x(i)+0.22], [cvMean, cvMean], ...
        '-', 'Color',[0.2 0.2 0.2], 'LineWidth',2);
    labelY = max(cv_vals) + 0.05*yUpper;
    text(x(i), labelY, sprintf('mean=%.2f', cvMean), ...
        'HorizontalAlignment','center', 'FontSize',10, 'FontWeight','bold');
end

set(gca, 'XTick',x, 'XTickLabel',modelLabels);
xlim([0.5, numel(modelOrder)+0.5]);
ylim([0, yUpper]);
ylabel('Mean CV across each sample''s top-10 LIME attributions');
title({'LIME repeated-run stability by classifier', ...
       'Five attack samples; 20 seeds per sample; lower CV means less relative dispersion'});
grid on; set(gca,'GridAlpha',0.25, 'Layer','top');
hold off;
ax5 = gca;
ax5.Toolbar = [];
exportgraphics(fig5, fullfile(outDir, 'fig5_lime_stability.png'), 'Resolution',300);
fprintf('  Saved fig5_lime_stability.png\n');

%% =====================================================================
%  FIGURE 6 - Timing comparison (SHAP vs LIME, 3 models)  [RQ2, H2]
%% =====================================================================
fig6 = figure('Color','w','Position',[100 100 900 520]);
shapRows = timeTbl(timeTbl.Method == "SHAP", :);
limeRows = timeTbl(timeTbl.Method == "LIME", :);
[~, ordShap] = ismember(modelOrder, shapRows.Model);
[~, ordLime] = ismember(modelOrder, limeRows.Model);
shapRows = shapRows(ordShap, :);
limeRows = limeRows(ordLime, :);
shapT = shapRows.MeanSeconds;
limeT = limeRows.MeanSeconds;
shapLow = shapT - shapRows.CI95Lower;
shapHigh = shapRows.CI95Upper - shapT;
limeLow = limeT - limeRows.CI95Lower;
limeHigh = limeRows.CI95Upper - limeT;
nTiming = min([shapRows.N_CompleteCases; limeRows.N_CompleteCases]);
nTimingAttempted = min([shapRows.N_CasesAttempted; limeRows.N_CasesAttempted]);
nRepeats = min([shapRows.N_RepeatsPerCase; limeRows.N_RepeatsPerCase]);
nWarmup = min([shapRows.N_Warmup; limeRows.N_Warmup]);

x = 1:numel(modelOrder);  w = 0.35;
hold on;
b1 = bar(x - w/2, shapT, w, 'FaceColor', COL_SHAP, 'EdgeColor','k', 'LineWidth',0.5);
b2 = bar(x + w/2, limeT, w, 'FaceColor', COL_LIME, 'EdgeColor','k', 'LineWidth',0.5);
errorbar(x - w/2, shapT, shapLow, shapHigh, 'k.', ...
    'LineWidth',1.2, 'CapSize',10, 'HandleVisibility','off');
errorbar(x + w/2, limeT, limeLow, limeHigh, 'k.', ...
    'LineWidth',1.2, 'CapSize',10, 'HandleVisibility','off');

yMax = max([shapRows.CI95Upper; limeRows.CI95Upper]);
valuePad = 0.025 * yMax;
ratioPad = 0.18 * yMax;

% Value labels
for i = 1:numel(modelOrder)
    text(x(i)-w/2, shapRows.CI95Upper(i)+valuePad, sprintf('%.3fs', shapT(i)), ...
        'HorizontalAlignment','center', 'FontSize',10, 'FontWeight','bold');
    text(x(i)+w/2, limeRows.CI95Upper(i)+valuePad, sprintf('%.3fs', limeT(i)), ...
        'HorizontalAlignment','center', 'FontSize',10, 'FontWeight','bold');
    % Ratio
    r = shapT(i) / max(limeT(i), 1e-9);
    text(x(i), max(shapRows.CI95Upper(i), limeRows.CI95Upper(i)) + ratioPad, ...
        sprintf('SHAP/LIME = %.1fx', r), ...
        'HorizontalAlignment','center', 'FontSize',10, ...
        'FontWeight','bold', 'Color',[0.3 0.3 0.7]);
end

set(gca, 'XTick',x, 'XTickLabel',modelLabels);
xlim([0.4, numel(modelOrder)+0.6]);
ylim([0, yMax*1.55]);
ylabel('Mean wall-clock time per explanation (seconds)');
title({'Attack-score explanation time: SHAP vs LIME', ...
       sprintf(['%d cases/model; %d repeats/case after %d warm-up/method; ' ...
       'mean and 95%% CI (%d complete cases minimum)'], ...
       nTimingAttempted, nRepeats, nWarmup, nTiming)});
legend([b1, b2], {'SHAP','LIME'}, 'Location','northwest', 'Box','on');
grid on; set(gca,'GridAlpha',0.3, 'Layer','top');
hold off;
exportgraphics(fig6, fullfile(outDir, 'fig6_timing.png'), 'Resolution',300);
fprintf('  Saved fig6_timing.png\n');

%% =====================================================================
%  FIGURE 7 - SHAP top-10 features per model (3 panels)  [RQ3]
%% =====================================================================
fig7 = figure('Color','w','Position',[100 100 1300 750]);

for m = 1:numel(modelOrder)
    sub = featTbl.(matlab.lang.makeValidName(modelOrder(m)));
    subplot(1, 3, m);
    draw_top10(sub, 'SHAP_top10_attack', ...
        sprintf('%s', modelLabels{m}), COL_SHAP);
end
sgtitle('SHAP top-10 features by classifier (attack samples)', 'FontSize',13, 'FontWeight','bold');
exportgraphics(fig7, fullfile(outDir, 'fig7_shap_top10_per_model.png'), 'Resolution',300);
fprintf('  Saved fig7_shap_top10_per_model.png\n');

%% =====================================================================
%  FIGURE 8 - Primary top-k Jaccard and secondary full-vector Spearman
%% =====================================================================
if hasAttackAgree
    familyOrder = ["normal","DoS","Probe","R2L","U2R"];
    jacMat = nan(numel(modelOrder), numel(familyOrder));
    jacSdMat = nan(numel(modelOrder), numel(familyOrder));
    rhoMat = nan(numel(modelOrder), numel(familyOrder));
    rhoSdMat = nan(numel(modelOrder), numel(familyOrder));
    nMat = nan(numel(modelOrder), numel(familyOrder));

    for i = 1:numel(modelOrder)
        for j = 1:numel(familyOrder)
            row = attackAgreeTbl(attackAgreeTbl.Model == modelOrder(i) & ...
                                 attackAgreeTbl.Family == familyOrder(j), :);
            if ~isempty(row)
                jacMat(i,j) = row.TopK_Jaccard_mean(1);
                jacSdMat(i,j) = row.TopK_Jaccard_std(1);
                rhoMat(i,j) = row.SecondarySpearman_rho_mean(1);
                rhoSdMat(i,j) = row.SecondarySpearman_rho_std(1);
                nMat(i,j) = row.N_Primary(1);
            end
        end
    end

    fig8 = figure('Color','w','Position',[100 100 1450 560]);
    tl = tiledlayout(fig8, 1, 2, 'TileSpacing','compact', 'Padding','compact');
    nC = 256;

    % Primary measure: overlap among up to ten non-zero attributions.
    ax1 = nexttile(tl, 1);
    imagesc(ax1, jacMat, [0 1]);
    axis(ax1, 'equal'); axis(ax1, 'tight');
    jacLow = [0.97 0.98 0.99];
    jacHigh = [46 134 171] / 255;
    jacMap = [linspace(jacLow(1), jacHigh(1), nC)', ...
              linspace(jacLow(2), jacHigh(2), nC)', ...
              linspace(jacLow(3), jacHigh(3), nC)'];
    colormap(ax1, jacMap);
    cb1 = colorbar(ax1);
    cb1.Label.String = 'Mean top-k Jaccard';
    cb1.Ticks = [0 0.25 0.5 0.75 1];
    set(ax1, 'XTick',1:numel(familyOrder), 'XTickLabel',cellstr(familyOrder), ...
        'YTick',1:numel(modelOrder), 'YTickLabel',modelLabels, ...
        'TickLength',[0 0], 'FontSize',11);
    title(ax1, {'Primary: per-sample non-zero top-k overlap', 'cell = mean Jaccard \pm SD'}, ...
        'Interpreter','tex', 'FontSize',12);
    hold(ax1, 'on');
    draw_heatmap_grid(ax1, numel(modelOrder), numel(familyOrder));
    for i = 1:numel(modelOrder)
        for j = 1:numel(familyOrder)
            if isnan(jacMat(i,j)), continue; end
            if jacMat(i,j) >= 0.55
                txtColor = [1 1 1];
            else
                txtColor = [0.12 0.12 0.12];
            end
            text(ax1, j, i-0.13, sprintf('%.3f', jacMat(i,j)), ...
                'HorizontalAlignment','center', 'FontSize',12, ...
                'FontWeight','bold', 'Color',txtColor);
            text(ax1, j, i+0.19, sprintf('\\pm%.3f', jacSdMat(i,j)), ...
                'HorizontalAlignment','center', 'FontSize',8, ...
                'Color',txtColor, 'Interpreter','tex');
        end
    end
    hold(ax1, 'off');

    % Secondary sensitivity analysis. Sparse LIME vectors can induce tied ranks.
    ax2 = nexttile(tl, 2);
    imagesc(ax2, rhoMat, [-1 1]);
    axis(ax2, 'equal'); axis(ax2, 'tight');
    negCol = [217 95 2] / 255;
    zeroCol = [0.98 0.98 0.98];
    posCol = [46 134 171] / 255;
    rhoMap = [linspace(negCol(1), zeroCol(1), nC/2)', ...
              linspace(negCol(2), zeroCol(2), nC/2)', ...
              linspace(negCol(3), zeroCol(3), nC/2)'; ...
              linspace(zeroCol(1), posCol(1), nC/2)', ...
              linspace(zeroCol(2), posCol(2), nC/2)', ...
              linspace(zeroCol(3), posCol(3), nC/2)'];
    colormap(ax2, rhoMap);
    cb2 = colorbar(ax2);
    cb2.Label.String = 'Secondary Spearman \rho';
    cb2.Label.Interpreter = 'tex';
    cb2.Ticks = [-1 -0.5 0 0.5 1];
    set(ax2, 'XTick',1:numel(familyOrder), 'XTickLabel',cellstr(familyOrder), ...
        'YTick',1:numel(modelOrder), 'YTickLabel',modelLabels, ...
        'TickLength',[0 0], 'FontSize',11);
    title(ax2, {'Secondary: aligned full-vector Spearman \rho', ...
        'sparsity-sensitive; cell = mean \pm SD'}, 'Interpreter','tex', 'FontSize',12);
    hold(ax2, 'on');
    draw_heatmap_grid(ax2, numel(modelOrder), numel(familyOrder));
    for i = 1:numel(modelOrder)
        for j = 1:numel(familyOrder)
            if isnan(rhoMat(i,j)), continue; end
            if abs(rhoMat(i,j)) >= 0.55
                txtColor = [1 1 1];
            else
                txtColor = [0.12 0.12 0.12];
            end
            text(ax2, j, i-0.13, sprintf('%+.2f', rhoMat(i,j)), ...
                'HorizontalAlignment','center', 'FontSize',13, ...
                'FontWeight','bold', 'Color',txtColor);
            text(ax2, j, i+0.19, sprintf('\\pm%.2f', rhoSdMat(i,j)), ...
                'HorizontalAlignment','center', 'FontSize',9, ...
                'Color',txtColor, 'Interpreter','tex');
        end
    end
    hold(ax2, 'off');

    ax1.Toolbar = [];
    ax2.Toolbar = [];
    tl.Toolbar = [];

    sampleNote = sprintf('Normal n=%d; DoS, Probe, R2L and U2R n=%d each', ...
        nMat(1,1), nMat(1,2));
    sgtitle(tl, {'Primary top-set and secondary full-vector agreement by model and NSL-KDD family', ...
                 sampleNote}, 'FontSize',14, 'FontWeight','bold');

    exportgraphics(fig8, fullfile(outDir, 'fig8_agreement_per_attack_heatmap.png'), 'Resolution',300);
    exportgraphics(fig8, fullfile(outDir, 'fig8_spearman_per_attack_heatmap.png'), 'Resolution',300);
    fprintf('  Saved fig8_agreement_per_attack_heatmap.png (and legacy fig8 filename)\n');
else
    fprintf('  (skip fig8: agreement_per_attack_category.csv not found)\n');
end

fprintf('\nAll figures saved to: %s\n', outDir);

%% =====================================================================
%                          HELPER FUNCTION
%% =====================================================================
function draw_top10(tbl, colName, ttl, color)
    [~, ord] = sort(tbl.(colName), 'descend');
    top = tbl(ord(1:min(10, height(tbl))), :);
    top = flipud(top);

    barh(1:height(top), top.(colName), ...
        'FaceColor', color, 'EdgeColor','k', 'LineWidth',0.5);
    set(gca, 'YTick',1:height(top), 'YTickLabel',top.Feature, 'FontSize',9);
    xlim([0 22]);
    xlabel('# of attack samples (out of 20)');
    title(ttl, 'FontSize',11);
    grid on; set(gca,'GridAlpha',0.3, 'Layer','top');
    for i = 1:height(top)
        text(top.(colName)(i) + 0.3, i, num2str(top.(colName)(i)), ...
            'VerticalAlignment','middle', 'FontSize',8);
    end
end

function draw_heatmap_grid(ax, nRows, nCols)
    for i = 0.5:1:(nRows+0.5)
        plot(ax, [0.5 nCols+0.5], [i i], 'w-', 'LineWidth',1.5);
    end
    for j = 0.5:1:(nCols+0.5)
        plot(ax, [j j], [0.5 nRows+0.5], 'w-', 'LineWidth',1.5);
    end
end

function names = positiveTopFeatures(tbl, colName, k)
    values = tbl.(colName);
    eligible = find(isfinite(values) & values > 0);
    if isempty(eligible)
        names = strings(0,1);
        return;
    end
    [~, localOrder] = sort(values(eligible), 'descend');
    eligible = eligible(localOrder(1:min(k, numel(localOrder))));
    names = string(tbl.Feature(eligible));
end
