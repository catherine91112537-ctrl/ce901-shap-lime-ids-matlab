function varargout = nslKddShared(action, varargin)
% Shared NSL-KDD preprocessing, modelling, and attribution helpers.
%
% The public entry points are dispatched by ACTION so that main.m and
% main_per_attack.m use the same implementation rather than diverging.

switch lower(string(action))
    case "data_dir"
        varargout{1} = dataDir();
    case "load_data"
        [varargout{1:nargout}] = loadData(varargin{:});
    case "map_attack_families"
        varargout{1} = mapAttackFamilies(varargin{:});
    case "preprocess_train_test"
        [varargout{1:nargout}] = preprocessTrainTest(varargin{:});
    case "standardize_table"
        [varargout{1:nargout}] = standardizeTable(varargin{:});
    case "train_model"
        varargout{1} = trainModel(varargin{:});
    case "evaluate_binary"
        [varargout{1:nargout}] = evaluateBinary(varargin{:});
    case "crossval_binary"
        [varargout{1:nargout}] = crossValBinary(varargin{:});
    case "run_explanation_set"
        [varargout{1:nargout}] = runExplanationSet(varargin{:});
    case "run_lime_stability"
        [varargout{1:nargout}] = runLimeStability(varargin{:});
    case "run_timing"
        [varargout{1:nargout}] = runTiming(varargin{:});
    case "runtime_environment"
        varargout{1} = runtimeEnvironment(varargin{:});
    case "empty_failure_log"
        varargout{1} = emptyFailureLog();
    case "failure_row"
        varargout{1} = failureRow(varargin{:});
    otherwise
        error('Unknown nslKddShared action: %s', action);
end
end

function d = dataDir()
envDir = getenv('NSL_KDD_DIR');
if ~isempty(envDir)
    d = envDir;
else
    d = fullfile(fileparts(mfilename('fullpath')), 'KDD Dataset');
end
end

function [trainTbl, testTbl, featureNames] = loadData(trainFile, testFile)
featureNames = { ...
    'duration','protocol_type','service','flag', ...
    'src_bytes','dst_bytes','land','wrong_fragment','urgent', ...
    'hot','num_failed_logins','logged_in','num_compromised', ...
    'root_shell','su_attempted','num_root','num_file_creations', ...
    'num_shells','num_access_files','num_outbound_cmds', ...
    'is_host_login','is_guest_login', ...
    'count','srv_count', ...
    'serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate', ...
    'same_srv_rate','diff_srv_rate','srv_diff_host_rate', ...
    'dst_host_count','dst_host_srv_count', ...
    'dst_host_same_srv_rate','dst_host_diff_srv_rate', ...
    'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate', ...
    'dst_host_serror_rate','dst_host_srv_serror_rate', ...
    'dst_host_rerror_rate','dst_host_srv_rerror_rate', ...
    'attack','level'};

numVars = numel(featureNames);
opts = delimitedTextImportOptions('NumVariables', numVars, ...
    'Delimiter', ',', 'DataLines', [1 Inf]);
opts.VariableNames = featureNames;
opts.VariableTypes = repmat({'double'}, 1, numVars);
stringVars = {'protocol_type','service','flag','attack'};
for k = 1:numel(stringVars)
    idx = strcmp(opts.VariableNames, stringVars{k});
    opts.VariableTypes{idx} = 'string';
end

if ~isfile(trainFile), error('Training file not found: %s', trainFile); end
if ~isfile(testFile),  error('Test file not found: %s',  testFile);  end

trainTbl = readtable(trainFile, opts);
testTbl  = readtable(testFile,  opts);
end

function [YTrain, YTest, XTrainRaw, XTestRaw, XTrainTbl, XTestTbl, preproc, auditTbl] = ...
    preprocessTrainTest(trainTbl, testTbl, catPredictors)

YTrain = categorical(trainTbl.attack == "normal", [true false], {'normal','attack'});
YTest  = categorical(testTbl.attack  == "normal", [true false], {'normal','attack'});

XTrainRaw = removevars(trainTbl, {'attack','level'});
XTestRaw  = removevars(testTbl,  {'attack','level'});

preproc = fitPreprocessor(XTrainRaw, catPredictors);
XTrainTbl = transformWithPreprocessor(XTrainRaw, preproc);
XTestTbl  = transformWithPreprocessor(XTestRaw,  preproc);
auditTbl  = categoryAudit(XTrainRaw, XTestRaw, catPredictors, preproc);
end

function preproc = fitPreprocessor(XTrainRaw, catPredictors)
varNames = string(XTrainRaw.Properties.VariableNames);
catPredictors = string(catPredictors(:));
numNames = varNames(~ismember(varNames, catPredictors));

preproc = struct();
preproc.CategoricalPredictors = catPredictors;
preproc.NumericPredictors = numNames(:);
preproc.CategoryLevels = struct();
preproc.PredictorNames = numNames(:);
preproc.OneHotGroups = table(strings(0,1), zeros(0,1), zeros(0,1), ...
    'VariableNames', {'Variable','StartIndex','EndIndex'});

nextCol = numel(preproc.PredictorNames) + 1;
for i = 1:numel(catPredictors)
    v = catPredictors(i);
    cats = string(categories(categorical(XTrainRaw.(char(v)))));
    cats = cats(:);
    preproc.CategoryLevels.(char(v)) = cats;

    startIdx = nextCol;
    newNames = v + "_" + cats;
    preproc.PredictorNames = [preproc.PredictorNames; newNames(:)]; %#ok<AGROW>
    nextCol = nextCol + numel(cats);
    endIdx = nextCol - 1;
    preproc.OneHotGroups = [preproc.OneHotGroups; ...
        table(v, startIdx, endIdx, ...
        'VariableNames', {'Variable','StartIndex','EndIndex'})]; %#ok<AGROW>
end
end

function XTbl = transformWithPreprocessor(XRaw, preproc)
Xnum = table2array(XRaw(:, cellstr(preproc.NumericPredictors)));
X = Xnum;

for i = 1:numel(preproc.CategoricalPredictors)
    v = preproc.CategoricalPredictors(i);
    values = string(XRaw.(char(v)));
    cats = preproc.CategoryLevels.(char(v));
    M = zeros(height(XRaw), numel(cats));
    for c = 1:numel(cats)
        M(:, c) = values == cats(c);
    end
    X = [X, M]; %#ok<AGROW>
end

XTbl = array2table(X, 'VariableNames', cellstr(preproc.PredictorNames));
end

function auditTbl = categoryAudit(XTrainRaw, XTestRaw, catPredictors, preproc)
catPredictors = string(catPredictors(:));
auditTbl = table(strings(0,1), strings(0,1), strings(0,1), zeros(0,1), logical(zeros(0,1)), ...
    'VariableNames', {'Split','Variable','Category','Count','InTrainingVocabulary'});

for i = 1:numel(catPredictors)
    v = catPredictors(i);
    trainValues = string(XTrainRaw.(char(v)));
    testValues = string(XTestRaw.(char(v)));
    trainCats = preproc.CategoryLevels.(char(v));
    allCats = unique([trainValues; testValues]);

    for c = 1:numel(allCats)
        cat = allCats(c);
        auditTbl = [auditTbl; table("train", v, cat, sum(trainValues == cat), ...
            any(trainCats == cat), 'VariableNames', auditTbl.Properties.VariableNames)]; %#ok<AGROW>
        auditTbl = [auditTbl; table("test", v, cat, sum(testValues == cat), ...
            any(trainCats == cat), 'VariableNames', auditTbl.Properties.VariableNames)]; %#ok<AGROW>
    end
end
end

function [stabRecords, failureRows, validityRows] = runLimeStability( ...
    pipelineName, models, modelNames, trainFeatTbl, testFeatTbl, idxStab, ...
    allFeatures, topK, nSeeds, maxRef, nSynthetic, preproc, scaleMetas)

stabRecords = table(strings(0,1), zeros(0,1), zeros(0,1), ...
    'VariableNames', {'Model','SampleIdx','CoeffOfVar_top10'});
failureRows = emptyFailureLog();
validityRows = emptyValidityRows();
F = numel(allFeatures);

for m = 1:numel(models)
    mName = string(modelNames{m});
    fprintf('  Model: %s\n', mName);
    XtrainM = trainFeatTbl{m};
    XtestM = testFeatTbl{m};
    XRef = XtrainM(1:min(maxRef, height(XtrainM)), :);
    attackScoreFcn = @(X) attackScore(models{m}, X);

    for s = 1:numel(idxStab)
        rowIdx = idxStab(s);
        xq = XtestM(rowIdx, :);
        impMat = nan(F, nSeeds);

        for seed = 1:nSeeds
            rng(seed);
            try
                synthetic = makeValidLimeSyntheticData(XRef, nSynthetic);
                validity = oneHotValidity(synthetic, preproc.OneHotGroups, scaleMetas{m});
                validityRows = [validityRows; table(string(pipelineName), mName, ...
                    nSynthetic, validity.InvalidRows, validity.InvalidRate, ...
                    'VariableNames', validityRows.Properties.VariableNames)]; %#ok<AGROW>
                limeExp = lime(attackScoreFcn, 'CustomSyntheticData', synthetic, ...
                    'Type','regression', 'SimpleModelType','linear');
                limeFit = fit(limeExp, xq, F);
                fullLime = alignLimeVector(limeFit, F);
                impMat(:, seed) = abs(fullLime);
            catch ME
                failureRows = [failureRows; failureRow(pipelineName, mName, rowIdx, ...
                    "attack", seed, "lime_stability_attack_score", ME)]; %#ok<AGROW>
            end
        end

        featStd = std(impMat, 0, 2, 'omitnan');
        featMean = mean(impMat, 2, 'omitnan');
        topIdx = topKIndices(featMean, topK);
        if isempty(topIdx)
            cvMean = NaN;
        else
            cv = featStd(topIdx) ./ max(featMean(topIdx), 1e-9);
            cvMean = mean(cv, 'omitnan');
        end
        stabRecords = [stabRecords; table(mName, rowIdx, cvMean, ...
            'VariableNames', stabRecords.Properties.VariableNames)]; %#ok<AGROW>
        fprintf('    sample %d (idx=%d): mean CV across top-10 = %.3f\n', s, rowIdx, cvMean);
    end
end
end

function [timingTbl, timingRaw, timingCaseTbl, failureRows, validityRows] = runTiming( ...
    pipelineName, models, modelNames, trainFeatTbl, testFeatTbl, idxTime, ...
    topKFit, maxRef, nSynthetic, preproc, scaleMetas, nRepeats, nWarmup)

timingTbl = emptyTimingSummary();
timingRaw = emptyTimingRaw();
timingCaseTbl = emptyTimingCaseSummary();
failureRows = emptyFailureLog();
validityRows = emptyValidityRows();
methodNames = ["SHAP","LIME"];
idxTime = idxTime(:);

for m = 1:numel(models)
    mName = string(modelNames{m});
    fprintf('  Model: %s\n', mName);
    XtrainM = trainFeatTbl{m};
    XtestM = testFeatTbl{m};
    XRef = XtrainM(1:min(maxRef, height(XtrainM)), :);
    attackScoreFcn = @(X) attackScore(models{m}, X);

    shapExp = [];
    limeExp = [];
    try
        shapExp = shapley(models{m}, XRef, ...
            'NumObservationsToSample', min(100, height(XRef)));
    catch ME
        failureRows = [failureRows; failureRow(pipelineName, mName, NaN, "", NaN, ...
            "timing_build_shapley_attack_class", ME)]; %#ok<AGROW>
    end

    try
        rng(2000 + m);
        synthetic = makeValidLimeSyntheticData(XRef, nSynthetic);
        validity = oneHotValidity(synthetic, preproc.OneHotGroups, scaleMetas{m});
        validityRows = [validityRows; table(string(pipelineName), mName, ...
            nSynthetic, validity.InvalidRows, validity.InvalidRate, ...
            'VariableNames', validityRows.Properties.VariableNames)]; %#ok<AGROW>
        limeExp = lime(attackScoreFcn, 'CustomSyntheticData', synthetic, ...
            'Type','regression', 'SimpleModelType','linear');
    catch ME
        failureRows = [failureRows; failureRow(pipelineName, mName, NaN, "", NaN, ...
            "timing_build_lime_attack_score", ME)]; %#ok<AGROW>
    end

    fprintf('    Warm-up: %d call(s) per available method...\n', nWarmup);
    for methodCode = 1:2
        methodName = methodNames(methodCode);
        for w = 1:nWarmup
            warmIdx = idxTime(mod(w - 1, numel(idxTime)) + 1);
            try
                if methodName == "SHAP"
                    if isempty(shapExp), break; end
                    fit(shapExp, XtestM(warmIdx, :));
                else
                    if isempty(limeExp), break; end
                    fit(limeExp, XtestM(warmIdx, :), topKFit);
                end
            catch ME
                failureRows = [failureRows; failureRow(pipelineName, mName, warmIdx, ...
                    "timing_warmup", w, "timing_warmup_" + lower(methodName), ME)]; %#ok<AGROW>
            end
        end
    end

    [methodGrid, sampleGrid, repeatGrid] = ndgrid(1:2, 1:numel(idxTime), 1:nRepeats);
    schedule = [methodGrid(:), sampleGrid(:), repeatGrid(:)];
    active = (schedule(:,1) == 1 & ~isempty(shapExp)) | ...
        (schedule(:,1) == 2 & ~isempty(limeExp));
    schedule = schedule(active, :);
    rng(9000 + m, 'twister');
    schedule = schedule(randperm(size(schedule, 1)), :);

    fprintf('    Timing %d randomized explanation calls...\n', size(schedule, 1));
    for q = 1:size(schedule, 1)
        methodCode = schedule(q, 1);
        samplePos = schedule(q, 2);
        repeatNum = schedule(q, 3);
        methodName = methodNames(methodCode);
        sampleIdx = idxTime(samplePos);
        success = false;
        elapsed = NaN;
        errorMessage = "";

        try
            tStart = tic;
            if methodName == "SHAP"
                fit(shapExp, XtestM(sampleIdx, :));
            else
                fit(limeExp, XtestM(sampleIdx, :), topKFit);
            end
            elapsed = toc(tStart);
            success = true;
        catch ME
            errorMessage = string(ME.message);
            failureRows = [failureRows; failureRow(pipelineName, mName, sampleIdx, ...
                "timing_repeat_" + string(repeatNum), repeatNum, ...
                "timing_fit_" + lower(methodName) + "_attack_score", ME)]; %#ok<AGROW>
        end

        timingRaw = [timingRaw; table(mName, methodName, sampleIdx, repeatNum, q, ...
            elapsed, success, errorMessage, ...
            'VariableNames', timingRaw.Properties.VariableNames)]; %#ok<AGROW>

        if mod(q, 100) == 0 || q == size(schedule, 1)
            fprintf('      ... %d / %d timed calls completed\n', q, size(schedule, 1));
        end
    end

    for methodName = methodNames
        for s = 1:numel(idxTime)
            sampleIdx = idxTime(s);
            mask = timingRaw.Model == mName & timingRaw.Method == methodName & ...
                timingRaw.SampleIdx == sampleIdx;
            vals = timingRaw.Seconds(mask & timingRaw.Success);
            nAttempted = sum(mask);
            nSuccess = numel(vals);
            completeCase = nAttempted == nRepeats && nSuccess == nRepeats;

            timingCaseTbl = [timingCaseTbl; table(mName, methodName, sampleIdx, ...
                nAttempted, nSuccess, completeCase, safeMean(vals), safeStd(vals), ...
                safeMedian(vals), safeIqr(vals), ...
                'VariableNames', timingCaseTbl.Properties.VariableNames)]; %#ok<AGROW>
        end

        methodCases = timingCaseTbl(timingCaseTbl.Model == mName & ...
            timingCaseTbl.Method == methodName, :);
        completeCases = methodCases(methodCases.CompleteCase, :);
        caseMeans = completeCases.MeanSeconds;
        nComplete = height(completeCases);
        meanSeconds = safeMean(caseMeans);
        sdSeconds = safeStd(caseMeans);
        medianSeconds = safeMedian(caseMeans);
        iqrSeconds = safeIqr(caseMeans);
        meanWithinCaseSD = safeMean(completeCases.SDSeconds);
        [ciLow, ciHigh] = meanConfidenceInterval(caseMeans);

        methodRaw = timingRaw(timingRaw.Model == mName & timingRaw.Method == methodName, :);
        nAttempts = height(methodRaw);
        nSuccess = sum(methodRaw.Success);
        timingTbl = [timingTbl; table(mName, methodName, meanSeconds, sdSeconds, ...
            medianSeconds, iqrSeconds, ciLow, ciHigh, meanWithinCaseSD, ...
            numel(idxTime), nComplete, nRepeats, nAttempts, nSuccess, nWarmup, ...
            'VariableNames', timingTbl.Properties.VariableNames)]; %#ok<AGROW>

        fprintf(['    %-4s mean=%.3f s, SD=%.3f, median=%.3f, IQR=%.3f, ' ...
            '95%% CI=[%.3f, %.3f], complete cases=%d/%d, calls=%d/%d\n'], ...
            methodName, meanSeconds, sdSeconds, medianSeconds, iqrSeconds, ...
            ciLow, ciHigh, nComplete, numel(idxTime), nSuccess, nAttempts);
    end
end
end

function T = runtimeEnvironment(nCases, nRepeats, nWarmup)
osName = string(computer);
processor = "not available";
memoryBytes = "not available";
logicalCores = "not available";

if ismac
    osProduct = systemValue('sw_vers -productName');
    osVersion = systemValue('sw_vers -productVersion');
    if osProduct ~= "not available"
        osName = strtrim(osProduct + " " + osVersion);
    end
    processor = systemValue('sysctl -n machdep.cpu.brand_string');
    memoryBytes = systemValue('sysctl -n hw.memsize');
    logicalCores = systemValue('sysctl -n hw.logicalcpu');
elseif isunix
    osName = systemValue('uname -sr');
    processor = systemValue('lscpu | sed -n ''s/^Model name:[[:space:]]*//p''');
    memoryBytes = systemValue('awk ''/MemTotal/ {print $2 * 1024}'' /proc/meminfo');
    logicalCores = systemValue('nproc');
elseif ispc
    osName = string(getenv('OS'));
    processor = string(getenv('PROCESSOR_IDENTIFIER'));
    logicalCores = string(getenv('NUMBER_OF_PROCESSORS'));
end

parallelPool = "none";
try
    pool = gcp('nocreate');
    if ~isempty(pool)
        parallelPool = string(pool.NumWorkers) + " workers";
    end
catch
    parallelPool = "not available";
end

items = ["RunTimestamp"; "MATLABVersion"; "MATLABRelease"; ...
    "OperatingSystem"; "Architecture"; "Processor"; "LogicalCores"; ...
    "PhysicalMemoryBytes"; "ParallelPool"; "TimingCasesPerModel"; ...
    "TimedRepeatsPerCase"; "WarmupCallsPerMethod"; "TimedScope"; ...
    "Clock"; "MethodOrder"; "FailureExclusionRule"];
values = [string(datetime('now', 'Format','yyyy-MM-dd HH:mm:ss')); string(version); ...
    string(version('-release')); osName; string(computer('arch')); processor; ...
    logicalCores; memoryBytes; parallelPool; string(nCases); string(nRepeats); ...
    string(nWarmup); ...
    "Individual fit call only; explainer construction excluded"; ...
    "MATLAB tic/toc wall-clock seconds"; ...
    "Randomized within each model across method, case, and repeat"; ...
    "Exclude a model-method-case unless all planned timed repeats succeed"];
T = table(items, values, 'VariableNames', {'Item','Value'});
end

function value = systemValue(command)
[status, output] = system(command);
if status == 0 && strlength(strtrim(string(output))) > 0
    value = strtrim(string(output));
else
    value = "not available";
end
end

function value = safeMean(x)
x = x(isfinite(x));
if isempty(x), value = NaN; else, value = mean(x); end
end

function value = safeStd(x)
x = x(isfinite(x));
if isempty(x), value = NaN; elseif isscalar(x), value = 0; else, value = std(x, 0); end
end

function value = safeMedian(x)
x = x(isfinite(x));
if isempty(x), value = NaN; else, value = median(x); end
end

function value = safeIqr(x)
x = x(isfinite(x));
if isempty(x)
    value = NaN;
elseif isscalar(x)
    value = 0;
else
    value = prctile(x, 75) - prctile(x, 25);
end
end

function [ciLow, ciHigh] = meanConfidenceInterval(x)
x = x(isfinite(x));
n = numel(x);
if n < 2
    ciLow = NaN;
    ciHigh = NaN;
    return;
end
meanX = mean(x);
halfWidth = tinv(0.975, n - 1) * std(x, 0) / sqrt(n);
ciLow = meanX - halfWidth;
ciHigh = meanX + halfWidth;
end

function [ZTbl, muX, sdX] = standardizeTable(XTbl, muX, sdX)
X = table2array(XTbl);
if nargin < 2 || isempty(muX)
    muX = mean(X, 1);
    sdX = std(X, 0, 1);
    sdX(sdX == 0) = 1;
end
Z = (X - muX) ./ sdX;
ZTbl = array2table(Z, 'VariableNames', XTbl.Properties.VariableNames);
end

function model = trainModel(modelKind, X, Y)
switch char(modelKind)
    case 'RandomForest'
        model = fitcensemble(X, Y, 'Method','Bag', ...
            'NumLearningCycles',100, 'Learners','tree', ...
            'ClassNames',{'normal','attack'});
    case 'BoostedTrees'
        model = fitcensemble(X, Y, 'Method','LogitBoost', ...
            'NumLearningCycles',100, 'Learners','tree', ...
            'ClassNames',{'normal','attack'});
    case 'SVM'
        model = fitcsvm(X, Y, 'KernelFunction','rbf', ...
            'KernelScale','auto', 'Standardize',false, ...
            'ClassNames',{'normal','attack'});
        model = fitPosterior(model, X, Y);
    otherwise
        error('Unknown model kind: %s', modelKind);
end
end

function [acc, prec, rec, f1, auc] = evaluateBinary(model, X, Ytrue, posClass)
[ypred, yscore] = predict(model, X);
ypred = categorical(ypred, categories(Ytrue));
cm = confusionmat(Ytrue, ypred, 'Order', categorical({'normal','attack'}, {'normal','attack'}));
posIdx = find(strcmp({'normal','attack'}, char(posClass)));
tp = cm(posIdx, posIdx);
fp = sum(cm(:, posIdx)) - tp;
fn = sum(cm(posIdx, :)) - tp;
tn = sum(cm(:)) - tp - fp - fn; %#ok<NASGU>
acc  = sum(diag(cm)) / sum(cm(:));
prec = tp / max(tp + fp, 1);
rec  = tp / max(tp + fn, 1);
f1   = 2 * prec * rec / max(prec + rec, eps);

attackCol = attackColumn(model);
try
    [~,~,~,auc] = perfcurve(Ytrue, yscore(:, attackCol), 'attack');
catch
    auc = NaN;
end
end

function [meanAcc, stdAcc, meanF1, stdF1, foldTbl, failureTbl] = ...
    crossValBinary(modelKind, XRaw, Y, posClass, k, catPredictors)

cv = cvpartition(Y, 'KFold', k, 'Stratify', true);
accs = nan(k,1);
f1s = nan(k,1);
foldTbl = table();
failureTbl = emptyFailureLog();
classOrder = categorical({'normal','attack'}, {'normal','attack'});

for f = 1:k
    trIdx = training(cv, f);
    teIdx = test(cv, f);
    XtrRaw = XRaw(trIdx, :);
    XteRaw = XRaw(teIdx, :);
    Ytr = Y(trIdx);
    Yte = Y(teIdx);

    try
        preprocFold = fitPreprocessor(XtrRaw, catPredictors);
        Xtr = transformWithPreprocessor(XtrRaw, preprocFold);
        Xte = transformWithPreprocessor(XteRaw, preprocFold);
        if strcmp(modelKind, 'SVM')
            [Xtr, muFold, sdFold] = standardizeTable(Xtr);
            Xte = standardizeTable(Xte, muFold, sdFold);
        end

        fm = trainModel(modelKind, Xtr, Ytr);
        ypred = predict(fm, Xte);
        ypred = categorical(ypred, categories(Yte));
        cm = confusionmat(Yte, ypred, 'Order', classOrder);
        posIdx = find(strcmp({'normal','attack'}, char(posClass)));
        tp = cm(posIdx, posIdx);
        fp = sum(cm(:, posIdx)) - tp;
        fn = sum(cm(posIdx, :)) - tp;
        accs(f) = sum(diag(cm)) / sum(cm(:));
        prec = tp / max(tp + fp, 1);
        rec  = tp / max(tp + fn, 1);
        f1s(f) = 2 * prec * rec / max(prec + rec, eps);
    catch ME
        failureTbl = [failureTbl; failureRow("cv", modelKind, NaN, "fold_"+string(f), NaN, ...
            "crossValBinary", ME)]; %#ok<AGROW>
    end

    foldTbl = [foldTbl; table(string(modelKind), f, accs(f), f1s(f), ...
        'VariableNames', {'Model','Fold','Accuracy','F1'})]; %#ok<AGROW>
end

meanAcc = mean(accs, 'omitnan');
stdAcc = std(accs, 'omitnan');
meanF1 = mean(f1s, 'omitnan');
stdF1 = std(f1s, 'omitnan');
end

function family = mapAttackFamilies(rawAttackTest)
dosAttacks = ["neptune","smurf","back","teardrop","pod","land","apache2", ...
    "processtable","mailbomb","udpstorm","worm"];
probeAttacks = ["satan","ipsweep","portsweep","nmap","mscan","saint"];
r2lAttacks = ["guess_passwd","ftp_write","imap","phf","multihop", ...
    "warezmaster","warezclient","spy","xlock","xsnoop", ...
    "snmpguess","snmpgetattack","httptunnel","sendmail","named"];
u2rAttacks = ["buffer_overflow","loadmodule","perl","rootkit","ps", ...
    "sqlattack","xterm"];

rawAttackTest = lower(strtrim(string(rawAttackTest)));
family = strings(numel(rawAttackTest), 1);
for i = 1:numel(rawAttackTest)
    a = rawAttackTest(i);
    if a == "normal"
        family(i) = "normal";
    elseif any(a == dosAttacks)
        family(i) = "DoS";
    elseif any(a == probeAttacks)
        family(i) = "Probe";
    elseif any(a == r2lAttacks)
        family(i) = "R2L";
    elseif any(a == u2rAttacks)
        family(i) = "U2R";
    else
        family(i) = "other";
    end
end
end

function [sampleRows, attributionRows, failureRows, validityRows, freqTables, rhoByModel] = ...
    runExplanationSet(pipelineName, models, modelNames, trainFeatTbl, testFeatTbl, ...
    selectedIdx, selectedGroup, allFeatures, countGroups, topK, maxRef, nSynthetic, ...
    preproc, scaleMetas)

sampleRows = emptyExplanationRows();
attributionRows = emptyAttributionRows();
failureRows = emptyFailureLog();
validityRows = emptyValidityRows();
freqTables = cell(numel(models), 1);
rhoByModel = struct();

F = numel(allFeatures);
selectedGroup = string(selectedGroup(:));
allFeatures = string(allFeatures(:));
countGroups = string(countGroups(:));

for m = 1:numel(models)
    mName = string(modelNames{m});
    fprintf('\n--- Model %d/%d: %s ---\n', m, numel(models), mName);
    XtrainM = trainFeatTbl{m};
    XtestM = testFeatTbl{m};
    refRows = min(maxRef, height(XtrainM));
    XRef = XtrainM(1:refRows, :);
    attackScoreFcn = @(X) attackScore(models{m}, X);

    fprintf('  Building attack-class SHAP explainer... ');
    tBuild = tic;
    try
        shapExp = shapley(models{m}, XRef, ...
            'NumObservationsToSample', min(100, height(XRef)));
        fprintf('done [%.1fs]\n', toc(tBuild));
    catch ME
        fprintf('failed\n');
        shapExp = [];
        failureRows = [failureRows; failureRow(pipelineName, mName, NaN, "", NaN, ...
            "build_shapley_attack_class", ME)]; %#ok<AGROW>
    end

    fprintf('  Building LIME explainer with valid custom one-hot synthetic data... ');
    tBuild = tic;
    try
        rng(1000 + m);
        limeSynthetic = makeValidLimeSyntheticData(XRef, nSynthetic);
        validity = oneHotValidity(limeSynthetic, preproc.OneHotGroups, scaleMetas{m});
        validityRows = [validityRows; table(string(pipelineName), mName, ...
            nSynthetic, validity.InvalidRows, validity.InvalidRate, ...
            'VariableNames', validityRows.Properties.VariableNames)]; %#ok<AGROW>
        limeExp = lime(attackScoreFcn, 'CustomSyntheticData', limeSynthetic, ...
            'Type','regression', 'SimpleModelType','linear');
        fprintf('done [%.1fs, invalid synthetic rate %.4f]\n', toc(tBuild), validity.InvalidRate);
    catch ME
        fprintf('failed\n');
        limeExp = [];
        failureRows = [failureRows; failureRow(pipelineName, mName, NaN, "", NaN, ...
            "build_lime_attack_score", ME)]; %#ok<AGROW>
    end

    shapCounts = zeros(F, numel(countGroups));
    limeCounts = zeros(F, numel(countGroups));
    rhoVals = nan(numel(selectedIdx), 1);

    fprintf('  Explaining %d samples...\n', numel(selectedIdx));
    for s = 1:numel(selectedIdx)
        rowIdx = selectedIdx(s);
        group = selectedGroup(s);
        xq = XtestM(rowIdx, :);

        fullShap = nan(F, 1);
        fullLime = nan(F, 1);
        shapOK = false;
        limeOK = false;
        limeBlackbox = NaN;
        limeSimple = NaN;
        limeNonzeroN = NaN;
        limeNonzeroFraction = NaN;

        if ~isempty(shapExp)
            try
                shapFit = fit(shapExp, xq);
                sv = shapleyTable(shapFit);
                fullShap = alignShapleyVector(sv, allFeatures, "attack");
                shapOK = true;
            catch ME
                failureRows = [failureRows; failureRow(pipelineName, mName, rowIdx, group, NaN, ...
                    "fit_shapley_attack_class", ME)]; %#ok<AGROW>
            end
        end

        if ~isempty(limeExp)
            try
                limeFit = fit(limeExp, xq, F);
                [fullLime, limeBlackbox, limeSimple] = alignLimeVector(limeFit, F);
                limeNonzeroN = sum(isfinite(fullLime) & abs(fullLime) > 1e-12);
                limeNonzeroFraction = limeNonzeroN / F;
                limeOK = true;
            catch ME
                failureRows = [failureRows; failureRow(pipelineName, mName, rowIdx, group, NaN, ...
                    "fit_lime_attack_score", ME)]; %#ok<AGROW>
            end
        end

        rho = NaN;
        overlapN = NaN;
        unionN = NaN;
        jaccard = NaN;
        shapTopSize = NaN;
        limeTopSize = NaN;
        shapTopNames = strings(0,1);
        limeTopNames = strings(0,1);
        limeAbsError = NaN;

        if shapOK && limeOK
            shapAbs = abs(fullShap);
            limeAbs = abs(fullLime);
            rho = spearmanAligned(shapAbs, limeAbs);
            rhoVals(s) = rho;

            shapTopIdx = topKIndices(shapAbs, topK);
            limeTopIdx = topKIndices(limeAbs, topK);
            shapTopSize = numel(shapTopIdx);
            limeTopSize = numel(limeTopIdx);
            shapTopNames = allFeatures(shapTopIdx);
            limeTopNames = allFeatures(limeTopIdx);
            overlapN = numel(intersect(shapTopIdx, limeTopIdx));
            unionN = numel(union(shapTopIdx, limeTopIdx));
            if unionN > 0
                jaccard = overlapN / unionN;
            else
                jaccard = NaN;
            end
            limeAbsError = abs(limeBlackbox - limeSimple);

            gIdx = find(countGroups == group, 1);
            if ~isempty(gIdx)
                shapCounts(:, gIdx) = shapCounts(:, gIdx) + ismember((1:F)', shapTopIdx);
                limeCounts(:, gIdx) = limeCounts(:, gIdx) + ismember((1:F)', limeTopIdx);
            end

            attributionRows = [attributionRows; table( ...
                repmat(string(pipelineName), F, 1), repmat(mName, F, 1), ...
                repmat(rowIdx, F, 1), repmat(group, F, 1), allFeatures, ...
                fullShap, shapAbs, fullLime, limeAbs, ...
                'VariableNames', attributionRows.Properties.VariableNames)]; %#ok<AGROW>
        end

        sampleRows = [sampleRows; table(string(pipelineName), mName, rowIdx, group, ...
            rho, overlapN, unionN, jaccard, shapTopSize, limeTopSize, ...
            limeNonzeroN, limeNonzeroFraction, shapOK, limeOK, ...
            limeBlackbox, limeSimple, limeAbsError, ...
            strjoin(shapTopNames, ';'), strjoin(limeTopNames, ';'), ...
            'VariableNames', sampleRows.Properties.VariableNames)]; %#ok<AGROW>

        if mod(s, 5) == 0
            fprintf('    ... %d / %d explained\n', s, numel(selectedIdx));
        end
    end

    rhoByModel.(char(matlab.lang.makeValidName(mName))) = rhoVals;

    if ~isempty(countGroups)
        freqTbl = table(allFeatures, 'VariableNames', {'Feature'});
        for g = 1:numel(countGroups)
            suffix = matlab.lang.makeValidName(countGroups(g));
            freqTbl.(sprintf('SHAP_top10_%s', char(suffix))) = shapCounts(:, g);
            freqTbl.(sprintf('LIME_top10_%s', char(suffix))) = limeCounts(:, g);
        end
        freqTables{m} = freqTbl;
    end
end
end

function score = attackScore(model, X)
[~, yscore] = predict(model, X);
score = yscore(:, attackColumn(model));
end

function idx = attackColumn(model)
classNames = string(model.ClassNames);
idx = find(classNames == "attack", 1);
if isempty(idx)
    error('Could not find attack class in model.ClassNames.');
end
end

function T = makeValidLimeSyntheticData(XRef, nRows)
nRef = height(XRef);
idx = randi(nRef, nRows, 1);
T = XRef(idx, :);
end

function validity = oneHotValidity(T, oneHotGroups, scaleMeta)
X = table2array(T);
if isfield(scaleMeta, 'IsStandardized') && scaleMeta.IsStandardized
    X = X .* scaleMeta.Sd + scaleMeta.Mu;
end

invalid = false(size(X, 1), 1);
tol = 1e-6;
for g = 1:height(oneHotGroups)
    cols = oneHotGroups.StartIndex(g):oneHotGroups.EndIndex(g);
    block = X(:, cols);
    entriesAreBinary = all(abs(block - round(block)) <= tol & block >= -tol & block <= 1 + tol, 2);
    rowSums = sum(block, 2);
    groupIsValid = entriesAreBinary & (abs(rowSums - 1) <= tol | abs(rowSums) <= tol);
    invalid = invalid | ~groupIsValid;
end

validity = struct();
validity.InvalidRows = sum(invalid);
validity.InvalidRate = sum(invalid) / max(size(X, 1), 1);
end

function sv = shapleyTable(shapFit)
if isprop(shapFit, 'Shapley')
    sv = shapFit.Shapley;
elseif isprop(shapFit, 'ShapleyValues')
    sv = shapFit.ShapleyValues;
else
    error('No Shapley or ShapleyValues property found on shapley fit object.');
end
end

function full = alignShapleyVector(sv, allFeatures, targetClass)
if nargin < 3
    targetClass = "";
end
full = nan(numel(allFeatures), 1);
predictors = string(sv{:, 1});
varNames = string(sv.Properties.VariableNames);

if nargin >= 3 && strlength(targetClass) > 0
    targetVar = string(matlab.lang.makeValidName(targetClass));
    valueCol = find(varNames == string(targetClass) | varNames == targetVar, 1);
else
    valueCol = [];
end

if isempty(valueCol)
    if width(sv) == 2
        valueCol = 2;
    else
        error('Could not find requested SHAP target class column: %s', string(targetClass));
    end
end

values = sv{:, valueCol};
for j = 1:numel(predictors)
    idx = find(allFeatures == predictors(j), 1);
    if ~isempty(idx)
        full(idx) = values(j);
    end
end
end

function [full, blackboxFitted, simpleFitted] = alignLimeVector(limeFit, F)
full = zeros(F, 1);
idxs = limeFit.ImportantPredictors(:);
beta = limeFit.SimpleModel.Beta(:);
n = min(numel(idxs), numel(beta));
full(idxs(1:n)) = beta(1:n);
blackboxFitted = scalarOrNaN(limeFit.BlackboxFitted);
simpleFitted = scalarOrNaN(limeFit.SimpleModelFitted);
end

function x = scalarOrNaN(value)
if isempty(value)
    x = NaN;
elseif isnumeric(value)
    x = value(1);
elseif iscell(value) && isnumeric(value{1})
    x = value{1};
else
    x = NaN;
end
end

function rho = spearmanAligned(shapAbs, limeAbs)
valid = isfinite(shapAbs) & isfinite(limeAbs);
if sum(valid) < 3 || std(shapAbs(valid)) == 0 || std(limeAbs(valid)) == 0
    rho = NaN;
else
    rho = corr(shapAbs(valid), limeAbs(valid), 'Type','Spearman', 'Rows','complete');
end
end

function idx = topKIndices(values, k)
values = values(:);
finiteValues = values(isfinite(values));
if isempty(finiteValues)
    idx = zeros(0,1);
    return;
end

scale = max(abs(finiteValues));
tol = max(1e-12, 10 * eps(max(scale, 1)));
eligibleIdx = find(isfinite(values) & values > tol);
[~, ord] = sort(values(eligibleIdx), 'descend');
idx = eligibleIdx(ord(1:min(k, numel(ord))));
end

function T = emptyExplanationRows()
T = table(strings(0,1), strings(0,1), zeros(0,1), strings(0,1), ...
    zeros(0,1), zeros(0,1), zeros(0,1), zeros(0,1), ...
    zeros(0,1), zeros(0,1), zeros(0,1), zeros(0,1), ...
    false(0,1), false(0,1), zeros(0,1), zeros(0,1), zeros(0,1), ...
    strings(0,1), strings(0,1), ...
    'VariableNames', {'Pipeline','Model','SampleIdx','Group','Spearman_rho', ...
    'TopK_Overlap','TopK_Union','TopK_Jaccard', ...
    'SHAP_TopK_Size','LIME_TopK_Size','LIME_NonzeroCount','LIME_NonzeroFraction', ...
    'SHAP_OK','LIME_OK', ...
    'LIME_BlackboxAttackScore','LIME_SimpleAttackScore','LIME_LocalAbsError', ...
    'SHAP_TopK','LIME_TopK'});
end

function T = emptyAttributionRows()
T = table(strings(0,1), strings(0,1), zeros(0,1), strings(0,1), strings(0,1), ...
    zeros(0,1), zeros(0,1), zeros(0,1), zeros(0,1), ...
    'VariableNames', {'Pipeline','Model','SampleIdx','Group','Feature', ...
    'SHAP_Value','SHAP_Abs','LIME_Value','LIME_Abs'});
end

function T = emptyValidityRows()
T = table(strings(0,1), strings(0,1), zeros(0,1), zeros(0,1), zeros(0,1), ...
    'VariableNames', {'Pipeline','Model','NumSyntheticRows','InvalidRows','InvalidRate'});
end

function T = emptyTimingRaw()
T = table(strings(0,1), strings(0,1), zeros(0,1), zeros(0,1), zeros(0,1), ...
    zeros(0,1), false(0,1), strings(0,1), ...
    'VariableNames', {'Model','Method','SampleIdx','Repeat','OrderWithinModel', ...
    'Seconds','Success','ErrorMessage'});
end

function T = emptyTimingCaseSummary()
T = table(strings(0,1), strings(0,1), zeros(0,1), zeros(0,1), zeros(0,1), ...
    false(0,1), zeros(0,1), zeros(0,1), zeros(0,1), zeros(0,1), ...
    'VariableNames', {'Model','Method','SampleIdx','N_Attempted','N_Success', ...
    'CompleteCase','MeanSeconds','SDSeconds','MedianSeconds','IQRSeconds'});
end

function T = emptyTimingSummary()
T = table(strings(0,1), strings(0,1), zeros(0,1), zeros(0,1), zeros(0,1), ...
    zeros(0,1), zeros(0,1), zeros(0,1), zeros(0,1), zeros(0,1), zeros(0,1), ...
    zeros(0,1), zeros(0,1), zeros(0,1), zeros(0,1), ...
    'VariableNames', {'Model','Method','MeanSeconds','SDSeconds','MedianSeconds', ...
    'IQRSeconds','CI95Lower','CI95Upper','MeanWithinCaseSDSeconds', ...
    'N_CasesAttempted','N_CompleteCases','N_RepeatsPerCase','N_Attempts', ...
    'N_Success','N_Warmup'});
end

function T = emptyFailureLog()
T = table(strings(0,1), strings(0,1), zeros(0,1), strings(0,1), zeros(0,1), ...
    strings(0,1), strings(0,1), ...
    'VariableNames', {'Pipeline','Model','SampleIdx','Group','Seed','Stage','ErrorMessage'});
end

function T = failureRow(pipelineName, modelName, sampleIdx, group, seed, stage, ME)
if isa(ME, 'MException')
    msg = string(ME.message);
else
    msg = string(ME);
end
template = emptyFailureLog();
T = table(string(pipelineName), string(modelName), sampleIdx, string(group), seed, ...
    string(stage), msg, 'VariableNames', template.Properties.VariableNames);
end
