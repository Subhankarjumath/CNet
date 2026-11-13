clc;
% Define the parameter ranges for LHS sampling
paramRanges = [
   0.5, 0.6;  % b1
   0.43, 0.51;  % l1
   0.12, 0.13;   %lambda1
   0.09, 0.11;  % ro0
   0.3 , 0.4;  % b2
   0.55, 0.65;  %l2
   0.34, 0.346; %lambda2
   1.05, 1.15; %lambdaM
   13.5, 16.5; %CT
   0.00015, 0.00025;  % delta1 
   0.0009, 0.0015;   %gamma
   0.0065, 0.0075;  % xi
   0.065, 0.075;  % lambdaT
   18, 22;  % CD
   0.00095, 0.0015;  % delta2
   0.13, 0.17;  % lambda_D
   28, 32;  % CK
   0.095, 0.15;  % eta
   0.18, 0.22;  % lambdaK
];

%T_max = 300;
% Define the number of samples for LHS
numSamples = 1000;

% Latin Hypercube Sampling to generate parameter sets
lhsSamples = lhsdesign(numSamples, size(paramRanges, 1));

% Scale the samples to the parameter ranges
paramSamples = lhsSamples .* (paramRanges(:,2) - paramRanges(:,1))' + paramRanges(:,1)';

% Initialize array to store peak viral DNA values for each sample
peak_keratinocytes = zeros(numSamples, 1);

% Time span for solving the ODE
tSpan = [0 100];

% Initial conditions for my model
S10=24;
S20=15;
SM0=11;
T0=15;
D0=10;
K0=20;
initialConditions = [S10, S20, SM0, T0, D0, K0];

% Solve the model for each sampled parameter set
for i = 1:numSamples
    b1 = paramSamples(i, 1);
    l1 = paramSamples(i, 2);
    lambda1 = paramSamples(i, 3);
    ro0 = paramSamples(i, 4);
    b2 = paramSamples(i, 5);
    l2 = paramSamples(i, 6);
    lambda2 = paramSamples(i, 7);
    lambdaM = paramSamples(i, 8);
    CT = paramSamples(i, 9);
    delta1 = paramSamples(i, 10);
    gamma = paramSamples(i, 11);
    xi = paramSamples(i, 12);
    lambdaT = paramSamples(i, 13);
    CD = paramSamples(i, 14);
    delta2 = paramSamples(i, 15);
    lambdaD = paramSamples(i, 16);
    CK = paramSamples(i, 17);
    eta = paramSamples(i, 18);
    lambdaK = paramSamples(i, 19);
    
    % Define the ODE function for SIR model
    Model = @(t, x) [(2*b1-1)*l1*x(1)-lambda1*x(1)+ro0;
            (2*b2-1)*l2*x(2)+2*(1-b1)*l1*x(1)-lambda2*x(2);
            2*(1-b2)*l2*x(2)-lambdaM*x(2);
            CT-gamma*x(4)*x(5)-delta1*x(4)*x(6)-xi*x(3)*x(2)-lambdaT*x(4);
            CD-delta2*x(5)*x(6)-lambdaD*x(5);
            CK+((delta1*x(4)*x(6)+delta2*x(5)*x(6))/(1+eta*x(2)))-lambdaK*x(6)];
    
    % Solve the ODE using ode45
    [t, x] = ode45(Model, tSpan, initialConditions);
    
    % Extract the infected population (I) and find the peak value
    keratinocytes = x(:, 6);
    peak_keratinocytes(i) = max(keratinocytes);
    
    % Plot the SIR dynamics for the first parameter set
%     if i == 1
%         figure;
%         plot(t, y(:, 1), 'b', 'LineWidth', 2); hold on;
%         plot(t, y(:, 2), 'r', 'LineWidth', 2);
%         plot(t, y(:, 3), 'g', 'LineWidth', 2);
%         xlabel('Time');
%         ylabel('Population Fraction');
%         legend('Susceptible', 'Infected', 'Recovered');
%         title('SIR Dynamics for Sample Parameter Set');
%         grid on;
%     end
end

% PRCC analysis
% Compute rank correlations between parameters and peak infected values
paramNames = {'b_1', 'l_1', '\lambda_1', 'ro_0', 'b_2', 'l_2', '\lambda_2',...
    '\lambda_M', 'CT', '\delta_1', '\gamma', 'xi', '\lambda_T', 'CD', '\delta_2',...
    '\lambda_D', 'CK', '\eta', '\lambda_K'};
rankedParams = tiedrank(paramSamples);
rankedPeak_keratinocytes = tiedrank(peak_keratinocytes);

% Calculate PRCC values
numParams = size(paramSamples, 2);
prccValues = zeros(1, numParams);
pValues = zeros(1, numParams);

for i = 1:numParams
    % Exclude the current parameter
    otherParams = rankedParams;
    otherParams(:, i) = [];
    % Calculate residuals for the current parameter and the output
   %paramResiduals = rankedParams(:, i) - regress(rankedParams(:, i), [ones(numSamples, 1), otherParams]);
   % outputResiduals = rankedPeakInfected - regress(rankedPeakInfected, [ones(numSamples, 1), otherParams]);
      % Fit linear model for the current parameter
    lmParam = fitlm(otherParams, rankedParams(:, i));
    paramResiduals = lmParam.Residuals.Raw;
    
    % Fit linear model for the output (peak infected values)
    lmOutput = fitlm(otherParams, rankedPeak_keratinocytes);
    outputResiduals = lmOutput.Residuals.Raw;
    % Calculate PRCC as the correlation between residuals
    [prccValues(i), pValues(i)] = corr(paramResiduals, outputResiduals, 'Type', 'Spearman');
    
     % Scatter plot of parameter residuals vs. output residuals
    subplot(4,5, i);
    scatter(paramResiduals, outputResiduals, '.');
    xlabel(['Residuals of ', paramNames{i}]);
   % ylabel('Residuals of Peak Infected');
    %title(['Scatter Plot of Residuals for ', paramNames{i}]);
    grid on;
end

% Display PRCC results
for i = 1:numParams
     fprintf('PRCC for %s: %.4f, p-value: %.4f\n', paramNames{i}, prccValues(i), pValues(i));
end
