clear; clc; close all;
tStart = tic;

scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir), scriptDir = pwd; end
addpath(scriptDir);

%% 1) Dataset folder -----------------------------------------------------
dataDir   = nslKddShared("data_dir");
trainFile = fullfile(dataDir, 'KDDTrain+_20Percent.txt');
testFile  = fullfile(dataDir, 'KDDTest+.txt');

outDir = fullfile(scriptDir, 'results');
if ~exist(outDir, 'dir'), mkdir(outDir); end

diaryFile = fullfile(outDir, sprintf('execution_log_main_per_attack_%s.txt', datestr(now,'yyyymmdd_HHMMSS')));
diary(diaryFile);
cleanupDiary = onCleanup(@() diary('off'));

fprintf('Per-attack pipeline started: %s\n', datestr(now));
fprintf('MATLAB version: %s\n', version);
fprintf('Data directory: %s\n', dataDir);
fprintf('Results directory: %s\n', outDir);

%% 2) Load data and map attack families ---------------------------------
fprintf('\n[1/5] Loading data...\n');
[trainTbl, testTbl] = nslKddShared("load_data", trainFile, testFile);
rawAttackTest = string(testTbl.attack);

fprintf('\n[2/5] Mapping attack names to families...\n');
family = nslKddShared("map_attack_families", rawAttackTest);
familiesUse = ["normal","DoS","Probe","R2L","U2R"];
fprintf('Test set family distribution:\n');
for f = 1:numel(familiesUse)
    fprintf('  %-8s : %d\n', familiesUse(f), sum(family == familiesUse(f)));
end
nOther = sum(family == "other");
if nOther > 0
    fprintf('  %-8s : %d  (unmapped attack names - check)\n', 'other', nOther);
end

%% 3) Training-only preprocessing ---------------------------------------
fprintf('\n[3/5] Fitting preprocessing on training data only...\n');
catPredictors = {'protocol_type','service','flag'};
[YTrain, YTest, XTrainRaw, ~, XTrainTbl, XTestTbl, preproc, preprocAudit] = ...
    nslKddShared("preprocess_train_test", trainTbl, testTbl, catPredictors);
writetable(preprocAudit, fullfile(outDir, 'preprocessing_category_audit_per_attack.csv'));

[XTrainZTbl, muX, sdX] = nslKddShared("standardize_table", XTrainTbl);
XTestZTbl = nslKddShared("standardize_table", XTestTbl, muX, sdX);

predictorNames = string(XTrainTbl.Properties.VariableNames);
allFeatures = predictorNames(:);
F = numel(allFeatures); %#ok<NASGU>

fprintf('     Predictors after training-only one-hot: %d\n', numel(allFeatures));
fprintf('     Train rows: %d | Test rows: %d\n', height(XTrainTbl), height(XTestTbl));
unseenTest = preprocAudit(preprocAudit.Split == "test" & ...
    preprocAudit.Count > 0 & ~preprocAudit.InTrainingVocabulary, :);
fprintf('     Test categories outside training vocabulary: %d rows in audit\n', height(unseenTest));

%% 4) Train 3 classifiers ------------------------------------------------
fprintf('\n[4/5] Training 3 classifiers...\n');
rng(1);
rfModel = nslKddShared("train_model", "RandomForest", XTrainTbl, YTrain);
gbModel = nslKddShared("train_model", "BoostedTrees", XTrainTbl, YTrain);
svmModel = nslKddShared("train_model", "SVM", XTrainZTbl, YTrain);

models       = {rfModel, gbModel, svmModel};
modelNames   = {'RandomForest','BoostedTrees','SVM'};
testFeatTbl  = {XTestTbl, XTestTbl, XTestZTbl};
trainFeatTbl = {XTrainTbl, XTrainTbl, XTrainZTbl};
scaleMetas   = { ...
    struct('IsStandardized',false), ...
    struct('IsStandardized',false), ...
    struct('IsStandardized',true, 'Mu',muX, 'Sd',sdX)};
fprintf('     done.\n');

%% 5) Sample selection ---------------------------------------------------
fprintf('\n[5/5] Selecting samples per family...\n');
N_PER_ATTACK = 10;
N_NORMAL = 20;
TOP_K = 10;
MAX_REF = 2000;
N_SYNTHETIC_LIME = 5000;

rng(42);
selectedIdx = [];
selectedFamily = strings(0,1);

for f = 1:numel(familiesUse)
    fam = familiesUse(f);
    idxF = find(family == fam);
    if isempty(idxF)
        fprintf('  %-8s : no samples available, skipping\n', fam);
        continue;
    end
    if fam == "normal"
        nTake = min(N_NORMAL, numel(idxF));
    else
        nTake = min(N_PER_ATTACK, numel(idxF));
    end
    pick = idxF(randperm(numel(idxF), nTake));
    selectedIdx = [selectedIdx; pick]; %#ok<AGROW>
    selectedFamily = [selectedFamily; repmat(fam, nTake, 1)]; %#ok<AGROW>
    fprintf('  %-8s : %d samples picked (out of %d available)\n', ...
        fam, nTake, numel(idxF));
end
fprintf('  Total samples: %d\n', numel(selectedIdx));

writetable(table(selectedIdx, selectedFamily, ...
    'VariableNames', {'SampleIdx','Family'}), ...
    fullfile(outDir, 'selected_samples_per_attack.csv'));

%% 6) Explain each sample with aligned SHAP + LIME -----------------------
fprintf('\nGenerating attack-score explanations and aligned per-sample agreement...\n');
[sampleRows, attributionRows, failureRows, validityRows] = ...
    nslKddShared("run_explanation_set", "per_attack", models, modelNames, trainFeatTbl, testFeatTbl, ...
    selectedIdx, selectedFamily, allFeatures, strings(0,1), TOP_K, MAX_REF, ...
    N_SYNTHETIC_LIME, preproc, scaleMetas);

allRows = sampleRows;
allRows.Properties.VariableNames{'Group'} = 'Family';
writetable(allRows, fullfile(outDir, 'agreement_per_sample_with_family.csv'));
writetable(allRows, fullfile(outDir, 'spearman_per_sample_with_family.csv'));
writetable(attributionRows, fullfile(outDir, 'attribution_vectors_per_attack.csv'));
writetable(failureRows, fullfile(outDir, 'explanation_failure_log_per_attack.csv'));
writetable(validityRows, fullfile(outDir, 'lime_synthetic_validity_per_attack.csv'));
fprintf('\nSaved per-attack sample agreement, raw attribution vectors, validity, and failure logs.\n');

%% 7) Aggregate: per (model, family) -------------------------------------
fprintf('\nAggregating per (model, family)...\n');
summary = table();
for m = 1:numel(modelNames)
    for f = 1:numel(familiesUse)
        sub = allRows(allRows.Model == string(modelNames{m}) & allRows.Family == familiesUse(f), :);
        if isempty(sub), continue; end
        [jacMean, jacStd, jacLo, jacHi, nJac] = meanCI95(sub.TopK_Jaccard);
        jacMed = finiteMedian(sub.TopK_Jaccard);
        [rhoMean, rhoStd, ~, ~, nRho] = meanCI95(sub.Spearman_rho);
        rhoMed = finiteMedian(sub.Spearman_rho);
        limeNonzeroMed = finiteMedian(sub.LIME_NonzeroCount);
        limeTopSizeMed = finiteMedian(sub.LIME_TopK_Size);
        summary = [summary; table(string(modelNames{m}), familiesUse(f), nJac, ...
            jacMean, jacStd, jacMed, jacLo, jacHi, nRho, ...
            rhoMean, rhoStd, rhoMed, limeNonzeroMed, limeTopSizeMed, ...
            'VariableNames', {'Model','Family','N_Primary', ...
            'TopK_Jaccard_mean','TopK_Jaccard_std','TopK_Jaccard_median', ...
            'TopK_Jaccard_CI_Lower','TopK_Jaccard_CI_Upper','N_Secondary', ...
            'SecondarySpearman_rho_mean','SecondarySpearman_rho_std', ...
            'SecondarySpearman_rho_median','LIME_Nonzero_median', ...
            'LIME_TopK_Size_median'})]; %#ok<AGROW>
        fprintf(['  %-13s | %-8s | N=%2d | primary Jaccard=%.3f +/- %.3f' ...
            ' | secondary rho=%+.3f +/- %.3f | LIME nonzero median=%.1f\n'], ...
            modelNames{m}, familiesUse(f), nJac, jacMean, jacStd, ...
            rhoMean, rhoStd, limeNonzeroMed);
    end
end

writetable(summary, fullfile(outDir, 'agreement_per_attack_category.csv'));
writetable(summary, fullfile(outDir, 'spearman_per_attack_category.csv'));
fprintf('\nSaved agreement_per_attack_category.csv and legacy-compatible spearman_per_attack_category.csv\n');

%% 8) Formal model-by-family analysis ------------------------------------
fprintf('\nFitting mixed model: TopK_Jaccard ~ Model*Family + (1|SampleID)...\n');
statsRows = allRows(isfinite(allRows.TopK_Jaccard), :);
mixedFailure = nslKddShared("empty_failure_log");
mixedAnova = table();
mixedCoef = table();

if height(statsRows) >= 2
    try
        statsRows.Model = categorical(statsRows.Model);
        statsRows.Family = categorical(statsRows.Family);
        statsRows.SampleID = categorical(statsRows.SampleIdx);

        lme = fitlme(statsRows, 'TopK_Jaccard ~ Model*Family + (1|SampleID)');
        mixedAnova = statsOutputToTable(anova(lme), "Term");
        mixedAnova.ResponseMetric = repmat(string("TopK_Jaccard"), height(mixedAnova), 1);
        mixedAnova = movevars(mixedAnova, 'ResponseMetric', 'Before', 1);

        mixedCoef = statsOutputToTable(lme.Coefficients, "Coefficient");
        mixedCoef.ResponseMetric = repmat(string("TopK_Jaccard"), height(mixedCoef), 1);
        mixedCoef = movevars(mixedCoef, 'ResponseMetric', 'Before', 1);

        fprintf('  Mixed model fitted successfully; saved ANOVA and coefficient tables.\n');
    catch ME
        mixedAnova = table();
        mixedCoef = table();
        mixedFailure = [mixedFailure; nslKddShared("failure_row", "per_attack", ...
            "mixed_model", NaN, "statistics", NaN, "fitlme_jaccard_model_family", ME)]; %#ok<AGROW>
        warning('Mixed model failed: %s', ME.message);
    end
else
    mixedFailure = [mixedFailure; nslKddShared("failure_row", "per_attack", ...
        "mixed_model", NaN, "statistics", NaN, "fitlme_jaccard_model_family", ...
        "Not enough non-missing top-k Jaccard rows for mixed model.")]; %#ok<AGROW>
end

writetable(mixedAnova, fullfile(outDir, 'mixed_model_family_anova.csv'));
writetable(mixedCoef, fullfile(outDir, 'mixed_model_family_coefficients.csv'));
writetable(mixedFailure, fullfile(outDir, 'mixed_model_family_failure_log.csv'));

fprintf('\n========================================================\n');
fprintf('DONE. Total wall-clock time: %.1f minutes\n', toc(tStart)/60);
fprintf('Execution log: %s\n', diaryFile);
fprintf('========================================================\n');

function T = statsOutputToTable(statsOutput, rowNameVariable)
if istable(statsOutput)
    T = statsOutput;
    names = string(statsOutput.Properties.RowNames);
elseif isa(statsOutput, 'dataset')
    names = string(statsOutput.Properties.ObsNames);
    T = dataset2table(statsOutput);
else
    T = struct2table(statsOutput);
    names = strings(height(T), 1);
end

if strlength(rowNameVariable) > 0 && ~ismember(char(rowNameVariable), T.Properties.VariableNames)
    if isempty(names)
        names = strings(height(T), 1);
    end
    names = names(:);
    if numel(names) ~= height(T)
        names = strings(height(T), 1);
    end
    T.(char(rowNameVariable)) = names;
    T = movevars(T, char(rowNameVariable), 'Before', 1);
end
end

function [mu, sdValue, ciLo, ciHi, n] = meanCI95(values)
values = values(isfinite(values));
n = numel(values);
if n == 0
    mu = NaN;
    sdValue = NaN;
    ciLo = NaN;
    ciHi = NaN;
    return;
end

mu = mean(values);
if n == 1
    sdValue = 0;
    ciLo = NaN;
    ciHi = NaN;
    return;
end

sdValue = std(values, 0);
halfWidth = tinv(0.975, n - 1) * sdValue / sqrt(n);
ciLo = mu - halfWidth;
ciHi = mu + halfWidth;
end

function value = finiteMedian(values)
values = values(isfinite(values));
if isempty(values)
    value = NaN;
else
    value = median(values);
end
end
