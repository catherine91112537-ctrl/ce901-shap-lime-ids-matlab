clear; clc; close all;

scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir), scriptDir = pwd; end
addpath(scriptDir);

%% 1) Dataset folder -----------------------------------------------------
% Default to the repository-local dataset. Override with NSL_KDD_DIR when
% running from a clean checkout with data stored elsewhere.
dataDir   = nslKddShared("data_dir");
trainFile = fullfile(dataDir, 'KDDTrain+_20Percent.txt');
testFile  = fullfile(dataDir, 'KDDTest+.txt');

outDir = fullfile(scriptDir, 'results');
if ~exist(outDir, 'dir'), mkdir(outDir); end

diaryFile = fullfile(outDir, sprintf('execution_log_main_%s.txt', datestr(now,'yyyymmdd_HHMMSS')));
diary(diaryFile);
cleanupDiary = onCleanup(@() diary('off'));

masterTimer = tic;
fprintf('Main pipeline started: %s\n', datestr(now));
fprintf('MATLAB version: %s\n', version);
fprintf('Data directory: %s\n', dataDir);
fprintf('Results directory: %s\n', outDir);

%% 2) Load data ----------------------------------------------------------
fprintf('\n[1/8] Loading data...\n');
[trainTbl, testTbl] = nslKddShared("load_data", trainFile, testFile);

%% 3) Training-only preprocessing ---------------------------------------
fprintf('\n[2/8] Fitting preprocessing on training data only...\n');
catPredictors = {'protocol_type','service','flag'};
[YTrain, YTest, XTrainRaw, XTestRaw, XTrainTbl, XTestTbl, preproc, preprocAudit] = ...
    nslKddShared("preprocess_train_test", trainTbl, testTbl, catPredictors);
writetable(preprocAudit, fullfile(outDir, 'preprocessing_category_audit.csv'));

% z-score normalised version for SVM. The final held-out transform is fitted
% on the full training set only; cross-validation refits it inside each fold.
[XTrainZTbl, muX, sdX] = nslKddShared("standardize_table", XTrainTbl);
XTestZTbl = nslKddShared("standardize_table", XTestTbl, muX, sdX);

predictorNames = string(XTrainTbl.Properties.VariableNames);
allFeatures = predictorNames(:);
F = numel(allFeatures);

fprintf('     Predictors after training-only one-hot: %d\n', F);
fprintf('     Train rows: %d | Test rows: %d\n', height(XTrainTbl), height(XTestTbl));
unseenTest = preprocAudit(preprocAudit.Split == "test" & ...
    preprocAudit.Count > 0 & ~preprocAudit.InTrainingVocabulary, :);
fprintf('     Test categories outside training vocabulary: %d rows in audit\n', height(unseenTest));

%% ======================================================================
%  PART A - Train 3 classifiers (Random Forest, Boosted Trees, SVM)
%% ======================================================================
fprintf('\n[3/8] Training classifiers...\n');
rng(1);

fprintf('     Random Forest (100 bagged trees)... '); tA = tic;
rfModel = nslKddShared("train_model", "RandomForest", XTrainTbl, YTrain);
fprintf('done in %.1f s\n', toc(tA));

fprintf('     Boosted Trees (LogitBoost, 100 cycles)... '); tA = tic;
gbModel = nslKddShared("train_model", "BoostedTrees", XTrainTbl, YTrain);
fprintf('done in %.1f s\n', toc(tA));

fprintf('     SVM (RBF kernel, z-scored features)...\n');
fprintf('     Note: this is the slowest step. Expect 20-60 minutes on 25k samples.\n');
tA = tic;
svmModel = nslKddShared("train_model", "SVM", XTrainZTbl, YTrain);
fprintf('     done in %.1f s (%.1f minutes)\n', toc(tA), toc(tA)/60);

models       = {rfModel, gbModel, svmModel};
modelNames   = {'RandomForest','BoostedTrees','SVM'};
testFeatTbl  = {XTestTbl,  XTestTbl,  XTestZTbl};
trainFeatTbl = {XTrainTbl, XTrainTbl, XTrainZTbl};
scaleMetas   = { ...
    struct('IsStandardized',false), ...
    struct('IsStandardized',false), ...
    struct('IsStandardized',true, 'Mu',muX, 'Sd',sdX)};

%% ======================================================================
%  PART B - 5-fold CV with fold-specific preprocessing
%% ======================================================================
fprintf('\n[4/8] 5-fold cross-validation on training set...\n');
cvResults = table();
cvFoldResults = table();
failureLog = nslKddShared("empty_failure_log");

for i = 1:numel(modelNames)
    tCV = tic;
    [meanAcc, stdAcc, meanF1, stdF1, foldTbl, cvFailures] = ...
        nslKddShared("crossval_binary", modelNames{i}, XTrainRaw, YTrain, 'attack', 5, catPredictors);
    fprintf('     %-13s CV-Acc=%.4f (+/-%.4f)  CV-F1=%.4f (+/-%.4f)  [%.1fs]\n', ...
        modelNames{i}, meanAcc, stdAcc, meanF1, stdF1, toc(tCV));

    cvResults = [cvResults; table(string(modelNames{i}), meanAcc, stdAcc, meanF1, stdF1, ...
        'VariableNames', {'Model','CV_Acc_Mean','CV_Acc_Std','CV_F1_Mean','CV_F1_Std'})]; %#ok<AGROW>
    cvFoldResults = [cvFoldResults; foldTbl]; %#ok<AGROW>
    failureLog = [failureLog; cvFailures]; %#ok<AGROW>
end

writetable(cvResults, fullfile(outDir, 'binary_cv_metrics.csv'));
writetable(cvFoldResults, fullfile(outDir, 'binary_cv_fold_metrics.csv'));
fprintf('     Saved binary_cv_metrics.csv and binary_cv_fold_metrics.csv\n');

%% ======================================================================
%  PART C - Test-set metrics
%% ======================================================================
fprintf('\n[5/8] Test-set metrics:\n');
binResults = table();
for i = 1:numel(models)
    [acc, prec, rec, f1, auc] = nslKddShared("evaluate_binary", models{i}, testFeatTbl{i}, YTest, 'attack');
    fprintf('     %-13s Acc=%.4f  Prec=%.4f  Rec=%.4f  F1=%.4f  AUC=%.4f\n', ...
        modelNames{i}, acc, prec, rec, f1, auc);
    binResults = [binResults; table(string(modelNames{i}), acc, prec, rec, f1, auc, ...
        'VariableNames', {'Model','Accuracy','Precision','Recall','F1','AUC'})]; %#ok<AGROW>
end
writetable(binResults, fullfile(outDir, 'binary_test_metrics.csv'));
fprintf('     Saved binary_test_metrics.csv\n');

classOrder = categorical({'normal','attack'}, {'normal','attack'});
classLabels = {'normal','attack'};
for i = 1:numel(models)
    yp = predict(models{i}, testFeatTbl{i});
    yp = categorical(yp, categories(YTest));
    cm = confusionmat(YTest, yp, 'Order', classOrder);
    figure; confusionchart(cm, classLabels);
    title(sprintf('%s - Confusion Matrix (binary)', modelNames{i}));
end

%% ======================================================================
%  PART D - SHAP and LIME on 40 samples
%% ======================================================================
fprintf('\n[6/8] Explanation generation (SHAP + LIME, 40 samples per model)...\n');

N_PER_CLASS = 20;
TOP_K = 10;
MAX_REF = 2000;
N_SYNTHETIC_LIME = 5000;

idxNormalAll = find(YTest == 'normal');
idxAttackAll = find(YTest == 'attack');
rng(42);
idxNormalSel = idxNormalAll(randperm(numel(idxNormalAll), min(N_PER_CLASS, numel(idxNormalAll))));
idxAttackSel = idxAttackAll(randperm(numel(idxAttackAll), min(N_PER_CLASS, numel(idxAttackAll))));
selectedIdx = [idxNormalSel; idxAttackSel];
selectedClass = [repmat("normal", numel(idxNormalSel), 1); ...
                 repmat("attack", numel(idxAttackSel), 1)];

selectedMainTbl = table(selectedIdx, selectedClass, ...
    'VariableNames', {'SampleIdx','Class'});
writetable(selectedMainTbl, fullfile(outDir, 'selected_samples_main.csv'));

[explainRows, attributionRows, explFailures, validityRows, freqTables, ~] = ...
    nslKddShared("run_explanation_set", "main", models, modelNames, trainFeatTbl, testFeatTbl, ...
    selectedIdx, selectedClass, allFeatures, ["normal","attack"], TOP_K, MAX_REF, ...
    N_SYNTHETIC_LIME, preproc, scaleMetas);

failureLog = [failureLog; explFailures]; %#ok<AGROW>
writetable(explainRows, fullfile(outDir, 'explanation_agreement_main.csv'));
writetable(attributionRows, fullfile(outDir, 'attribution_vectors_main.csv'));
writetable(validityRows, fullfile(outDir, 'lime_synthetic_validity_main.csv'));
fprintf('     Saved explanation_agreement_main.csv, attribution_vectors_main.csv, lime_synthetic_validity_main.csv\n');

for i = 1:numel(modelNames)
    fname = fullfile(outDir, sprintf('feature_top10_frequency_%s.csv', modelNames{i}));
    writetable(freqTables{i}, fname);
    fprintf('     Saved %s\n', fname);
end

%% ======================================================================
%  PART E - LIME stability test
%% ======================================================================
N_STAB = 5;
N_SEEDS_LIME = 20;
fprintf('\n[7/8] LIME attack-score stability (%d attack samples x %d seeds per model)...\n', ...
    N_STAB, N_SEEDS_LIME);
idxStab = idxAttackSel(1:min(N_STAB, numel(idxAttackSel)));

[stabRecords, stabFailures, stabValidity] = nslKddShared("run_lime_stability", ...
    "main_lime_stability", models, modelNames, trainFeatTbl, testFeatTbl, idxStab, ...
    allFeatures, TOP_K, N_SEEDS_LIME, MAX_REF, N_SYNTHETIC_LIME, preproc, scaleMetas);
failureLog = [failureLog; stabFailures]; %#ok<AGROW>
writetable(stabRecords, fullfile(outDir, 'lime_stability.csv'));
writetable(stabValidity, fullfile(outDir, 'lime_synthetic_validity_stability.csv'));
fprintf('     Saved lime_stability.csv and lime_synthetic_validity_stability.csv\n');

%% ======================================================================
%  PART F - Timing measurement
%% ======================================================================
N_TIME = 50;
N_TIME_REPEATS = 5;
N_TIME_WARMUP = 1;
fprintf(['\n[8/8] Timing test (SHAP and LIME, %d separately sampled test cases, ' ...
    '%d repeats/case after %d warm-up/method)...\n'], ...
    N_TIME, N_TIME_REPEATS, N_TIME_WARMUP);
rng(4242);
idxTimeAll = randperm(height(XTestTbl), min(N_TIME, height(XTestTbl)))';
writetable(table(idxTimeAll, 'VariableNames', {'SampleIdx'}), ...
    fullfile(outDir, 'selected_samples_timing.csv'));

[timingTbl, timingRaw, timingCaseTbl, timingFailures, timingValidity] = ...
    nslKddShared("run_timing", ...
    "main_timing", models, modelNames, trainFeatTbl, testFeatTbl, idxTimeAll, ...
    F, MAX_REF, N_SYNTHETIC_LIME, preproc, scaleMetas, ...
    N_TIME_REPEATS, N_TIME_WARMUP);
failureLog = [failureLog; timingFailures]; %#ok<AGROW>
writetable(timingTbl, fullfile(outDir, 'timing.csv'));
writetable(timingRaw, fullfile(outDir, 'timing_raw.csv'));
writetable(timingCaseTbl, fullfile(outDir, 'timing_case_summary.csv'));
writetable(timingFailures, fullfile(outDir, 'timing_failure_log.csv'));
writetable(timingValidity, fullfile(outDir, 'lime_synthetic_validity_timing.csv'));
timingEnvironment = nslKddShared("runtime_environment", ...
    numel(idxTimeAll), N_TIME_REPEATS, N_TIME_WARMUP);
writetable(timingEnvironment, fullfile(outDir, 'timing_environment.csv'));
fprintf(['     Saved timing.csv, timing_raw.csv, timing_case_summary.csv, ' ...
    'timing_failure_log.csv, timing_environment.csv, and ' ...
    'lime_synthetic_validity_timing.csv\n']);
fprintf('     Timing environment and analysis rule:\n');
for iEnv = 1:height(timingEnvironment)
    fprintf('       %-28s %s\n', timingEnvironment.Item(iEnv), timingEnvironment.Value(iEnv));
end

%% ======================================================================
%  PART G - Primary top-k Jaccard agreement and paired model test
%% ======================================================================
fprintf('\n[+] Computing primary per-sample top-k Jaccard agreement...\n');
fprintf('    Full-vector Spearman is retained only as a sparsity-sensitive secondary description.\n');

agreeTbl = table();
limeSparsityTbl = table();
jaccardMatrix = nan(numel(selectedIdx), numel(modelNames));

for m = 1:numel(modelNames)
    mName = modelNames{m};
    modelRows = explainRows(explainRows.Model == string(mName), :);
    [found, loc] = ismember(selectedIdx, modelRows.SampleIdx);

    jaccard_v = nan(numel(selectedIdx), 1);
    rho_v = nan(numel(selectedIdx), 1);
    limeNonzero_v = nan(numel(selectedIdx), 1);
    jaccard_v(found) = modelRows.TopK_Jaccard(loc(found));
    rho_v(found) = modelRows.Spearman_rho(loc(found));
    limeNonzero_v(found) = modelRows.LIME_NonzeroCount(loc(found));
    jaccardMatrix(:, m) = jaccard_v;

    [jacMean, jacSD, jacLo, jacHi, nJac] = meanCI95(jaccard_v);
    [rhoMean, rhoSD, rhoLo, rhoHi, nRho] = meanCI95(rho_v);
    limeClean = limeNonzero_v(isfinite(limeNonzero_v));
    if isempty(limeClean)
        limeMean = NaN;
        limeMedian = NaN;
        limeMin = NaN;
        limeMax = NaN;
    else
        limeMean = mean(limeClean);
        limeMedian = median(limeClean);
        limeMin = min(limeClean);
        limeMax = max(limeClean);
    end

    rawTbl = table(selectedIdx, selectedClass, jaccard_v, rho_v, limeNonzero_v, ...
        'VariableNames', {'SampleIdx','Class','TopK_Jaccard', ...
        'SecondarySpearman_rho','LIME_NonzeroCount'});
    writetable(rawTbl, fullfile(outDir, sprintf('agreement_raw_%s.csv', ...
        matlab.lang.makeValidName(mName))));
    writematrix(rho_v(isfinite(rho_v)), fullfile(outDir, sprintf('rho_raw_%s.csv', ...
        matlab.lang.makeValidName(mName))));

    f = freqTables{m};
    shapTopAttack = aggregateTopKPositive(f.Feature, f.SHAP_top10_attack, TOP_K);
    limeTopAttack = aggregateTopKPositive(f.Feature, f.LIME_top10_attack, TOP_K);
    jacAttack = setJaccard(shapTopAttack, limeTopAttack);
    shapTopNormal = aggregateTopKPositive(f.Feature, f.SHAP_top10_normal, TOP_K);
    limeTopNormal = aggregateTopKPositive(f.Feature, f.LIME_top10_normal, TOP_K);
    jacNormal = setJaccard(shapTopNormal, limeTopNormal);

    fprintf(['  %-13s primary Jaccard mean=%.3f std=%.3f 95%% CI=[%.3f,%.3f] n=%d' ...
        ' | aggregate attack=%.3f normal=%.3f\n'], ...
        mName, jacMean, jacSD, jacLo, jacHi, nJac, jacAttack, jacNormal);
    fprintf(['                secondary rho mean=%.3f std=%.3f n=%d' ...
        ' | LIME nonzero median=%.1f range=[%.0f,%.0f] of %d\n'], ...
        rhoMean, rhoSD, nRho, limeMedian, limeMin, limeMax, F);

    agreeTbl = [agreeTbl; table(string(mName), ...
        string("per_sample_top10_nonzero_attributions"), ...
        jacMean, jacSD, jacLo, jacHi, nJac, jacAttack, jacNormal, ...
        string("secondary_full_vector_abs_sparsity_sensitive"), ...
        rhoMean, rhoSD, rhoLo, rhoHi, nRho, limeMean, limeMedian, limeMin, limeMax, F, ...
        'VariableNames', {'Model','PrimaryAgreementMethod', ...
        'TopK_Jaccard_mean','TopK_Jaccard_std','TopK_Jaccard_CI_Lower', ...
        'TopK_Jaccard_CI_Upper','N_Primary','Jaccard_top10_attack_aggregate', ...
        'Jaccard_top10_normal_aggregate','SecondaryAgreementMethod', ...
        'SecondarySpearman_rho_mean','SecondarySpearman_rho_std', ...
        'SecondarySpearman_CI_Lower','SecondarySpearman_CI_Upper','N_Secondary', ...
        'LIME_Nonzero_mean','LIME_Nonzero_median','LIME_Nonzero_min', ...
        'LIME_Nonzero_max','NumPredictors'})]; %#ok<AGROW>

    limeSparsityTbl = [limeSparsityTbl; table(string(mName), F, numel(limeClean), ...
        limeMean, limeMedian, limeMin, limeMax, limeMedian / F, ...
        'VariableNames', {'Model','NumPredictors','N_Explanations', ...
        'LIME_Nonzero_mean','LIME_Nonzero_median','LIME_Nonzero_min', ...
        'LIME_Nonzero_max','LIME_NonzeroMedianFraction'})]; %#ok<AGROW>
end

writetable(agreeTbl, fullfile(outDir, 'agreement_summary_main.csv'));
writetable(agreeTbl, fullfile(outDir, 'spearman_jaccard.csv'));
writetable(limeSparsityTbl, fullfile(outDir, 'lime_sparsity_audit.csv'));
fprintf('     Saved agreement_summary_main.csv, lime_sparsity_audit.csv, and legacy-compatible spearman_jaccard.csv\n');

completeRows = all(isfinite(jaccardMatrix), 2);
friedmanTbl = table(strings(0,1), nan(0,1), nan(0,1), nan(0,1), zeros(0,1), ...
    'VariableNames', {'ResponseMetric','pValue','ChiSquare','DF','NCompleteCases'});
posthocTbl = table(strings(0,1), strings(0,1), strings(0,1), ...
    nan(0,1), nan(0,1), nan(0,1), nan(0,1), ...
    'VariableNames', {'ResponseMetric','ModelA','ModelB','SignrankP', ...
    'HolmAdjustedP','SignedRankStatistic','MedianPairedDifference'});
if sum(completeRows) >= 2
    [pFriedman, tblFriedman] = friedman(jaccardMatrix(completeRows, :), 1, 'off');
    friedmanTbl = table(string("TopK_Jaccard"), pFriedman, ...
        tblFriedman{2,5}, tblFriedman{2,3}, sum(completeRows), ...
        'VariableNames', {'ResponseMetric','pValue','ChiSquare','DF','NCompleteCases'});
    fprintf('Friedman paired model test on top-k Jaccard: chi2=%.2f, df=%d, p=%.3e, n=%d\n', ...
        friedmanTbl.ChiSquare, friedmanTbl.DF, friedmanTbl.pValue, friedmanTbl.NCompleteCases);

    pairs = [1 2; 1 3; 2 3];
    pVals = nan(size(pairs,1),1);
    statVals = nan(size(pairs,1),1);
    medianDiff = nan(size(pairs,1),1);
    for p = 1:size(pairs,1)
        a = pairs(p,1); b = pairs(p,2);
        medianDiff(p) = median(jaccardMatrix(completeRows,a) - ...
            jaccardMatrix(completeRows,b), 'omitnan');
        try
            [pVals(p), ~, statsSign] = signrank( ...
                jaccardMatrix(completeRows,a), jaccardMatrix(completeRows,b));
            if isfield(statsSign, 'signedrank')
                statVals(p) = statsSign.signedrank;
            end
        catch ME
            pVals(p) = NaN;
            pairName = string(modelNames{a}) + "_vs_" + string(modelNames{b});
            failureLog = [failureLog; nslKddShared("failure_row", "main", pairName, ...
                NaN, "statistics", NaN, "paired_posthoc_jaccard_signrank", ME)]; %#ok<AGROW>
        end
    end
    holmP = holmAdjust(pVals);
    modelA = string(modelNames(pairs(:,1))); modelA = modelA(:);
    modelB = string(modelNames(pairs(:,2))); modelB = modelB(:);
    posthocTbl = table(repmat(string("TopK_Jaccard"), size(pairs,1), 1), ...
        modelA, modelB, pVals, holmP, statVals, medianDiff, ...
        'VariableNames', {'ResponseMetric','ModelA','ModelB','SignrankP', ...
        'HolmAdjustedP','SignedRankStatistic','MedianPairedDifference'});
end
writetable(friedmanTbl, fullfile(outDir, 'friedman_model_test.csv'));
writetable(posthocTbl, fullfile(outDir, 'paired_posthoc_signrank.csv'));

writetable(failureLog, fullfile(outDir, 'explanation_failure_log.csv'));
fprintf('     Saved explanation_failure_log.csv (%d failures)\n', height(failureLog));

fprintf('\n========================================================\n');
fprintf('DONE. Total wall-clock time: %.1f minutes\n', toc(masterTimer)/60);
fprintf('All CSV outputs in: %s\n', outDir);
fprintf('Execution log: %s\n', diaryFile);
fprintf('========================================================\n');

function pAdj = holmAdjust(p)
pAdj = nan(size(p));
valid = isfinite(p);
[sortedP, ord] = sort(p(valid), 'ascend');
m = numel(sortedP);
adjSorted = nan(m,1);
runningMax = 0;
for i = 1:m
    runningMax = max(runningMax, (m - i + 1) * sortedP(i));
    adjSorted(i) = min(runningMax, 1);
end
validIdx = find(valid);
pAdj(validIdx(ord)) = adjSorted;
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

function names = aggregateTopKPositive(features, counts, k)
candidateIdx = find(isfinite(counts) & counts > 0);
if isempty(candidateIdx)
    names = strings(0, 1);
    return;
end

[~, order] = sort(counts(candidateIdx), 'descend');
take = candidateIdx(order(1:min(k, numel(order))));
names = string(features(take));
names = names(:);
end

function jac = setJaccard(a, b)
a = string(a);
b = string(b);
unionNames = union(a, b);
if isempty(unionNames)
    jac = NaN;
else
    jac = numel(intersect(a, b)) / numel(unionNames);
end
end
