%% MA(1): MLE and indirect inference.


% Suppose that the true data generating process (DGP) is:

% x_t = u_t - theta*u_t-1
% where u_t ~iid N(0,1)


% Estimate:
% (1) Directly --> true model's log-lieklihood function 
% or 
% (2) Indirectly --> AR(1) model's log-lieklihood function, 
% 
% and compare the two estimators. For agrid of values of theta ~(-0.9,0.9)

%% Putting SEED:

% Uncheck for Monte Carlo simulation (?):
rng(1234); % Establece la semilla en 1234

%% a) DIRECT ESTIMATION:
% 
% Generate S = 10,000 simulations from the DGP, 
% where T = 100 and estimate theta_hat using TRUE MLE.

T = 100;


% God give me this number:
theta = 0.7;

% true DGP: ( MA(1) with mu = 0 ):
x_t = zeros(T,1);
u_t = zeros(T,1);

    % Set some initial values:
u_t(1) = randn(1);
x_t(1) = randn(1);

for t = 2:T
    
    u_t(t) = randn(1);
    x_t(t) = u_t(t) - theta*u_t(t-1);

end

% PLot result to see if it makes any sense:
plot(x_t, 'b-', 'LineWidth', 1.5)
xlabel('Tiempo')
ylabel('Valor')
title('Serie Temporal')
grid on


%% Minimización

theta_init = 0.5;

options2 = optimoptions('fminunc', 'Algorithm', 'quasi-newton','Display', 'off');
[theta_hat1] = fminunc(@(theta) criterior_function_MA(theta, T, x_t, u_t), theta_init, options2);

% Estimated Value:
% (Due to negative in DGP)
theta_ML = - theta_hat1;




%% b) INDIRECT ESTIMATION:

% Calculate the implied estimate of theta from the AR(1) model 
% (see Digression: 'The other way around' in Computer_lab_I_UA2025.pdf)

% Basically Same as before but for a AR(1):
% This end up being wrong--> we use the indirect estimation extracted from
% the Digression stated before.

%% Minimización
% Prubing that using AR(1) doesnt' work pretty well for estimating theta.
param_init = [0.5,0.5];

options2 = optimoptions('fminunc', 'Algorithm', 'quasi-newton','Display', 'off');
[optimal_hat, fval] = fminunc(@(p) criterior_function_AR(p, T, x_t), param_init, options2);

% Due to negative in DGP:
theta_ML_AR = - optimal_hat(1);
sigma2_ML_AR = optimal_hat(2);




%% c) MONTE CARLO SIMULATION:

% Compute the Monte Carlo mean and standard deviation of the direct
% and indirect estimates.
% plot them as three diFFerent time series in the same figure
% (with the true values of theta on the x-axis). 
% Comment on the results. 
% What happens when you increase the sample length to T = 1000 and 10000?

% Grid of Time periods and Monte Carlo size:

T = [100, 1000, 10000];
S = 10000;

% Grid of thetas:
thetas = -0.9:0.1:0.9;
theta_init = 0.5;

% Direct (MA):
MC_theta_MA = zeros(S,length(thetas),length(T));

% Indirect (AR):
MC_theta_AR = zeros(S,length(thetas),length(T));

% Indirect (theta_tilda):
MC_theta_indirect = zeros(S,length(thetas),length(T));



%% (i) Direct Monte Carlo Simulation:

counter = zeros(1,3);

for i = 1:length(T)

    for s = 1:S
    
        for j = 1:length(thetas)
        
            theta = thetas(j);
        
            x_t = zeros(T(i),1); u_t = zeros(T(i),1);    
            u_t(1) = randn(1); x_t(1) = randn(1);
        
                for t = 2:T(i)    
                    u_t(t) = randn(1);
                    x_t(t) = u_t(t) - theta*u_t(t-1);
                end
        
            options2 = optimoptions('fminunc', 'Algorithm', 'quasi-newton','Display', 'off');
            [theta_hat1, fval] = fminunc(@(theta) criterior_function_MA(theta, T(i), x_t, u_t), theta_init, options2);
        
            theta_ML =  - theta_hat1;
            MC_theta_MA(s,j,i) = theta_ML;
    
        end
        counter(i) = counter(i) + 1;
        disp(counter);
    end

end

%% (ii) Indirect Monte Carlo Simulation (AR estimation):

counter = zeros(1,3);
for i = 1:length(T)

    for s = 1:S
    
        for j = 1:length(thetas)
        
            theta = thetas(j);
        
            x_t = zeros(T(i),1); u_t = zeros(T(i),1);    
            u_t(1) = randn(1); x_t(1) = randn(1);
        
                for t = 2:T(i)    
                    u_t(t) = randn(1);
                    x_t(t) = u_t(t) - theta*u_t(t-1);
                end
        
            options2 = optimoptions('fminunc', 'Algorithm', 'quasi-newton','Display', 'off');
            [theta_hat2, fval] = fminunc(@(theta) criterior_function_AR1(theta, T(i), x_t), theta_init, options2);
        
            theta_ML2 =  - theta_hat2;
            MC_theta_AR(s,j,i) = theta_ML2;        
    
        end
        counter(i) = counter(i) + 1;
        disp(counter);
    end
end

%% (iii) Direct Monte Carlo Simulation (MA indirect estimation):

counter = zeros(1,3);
for i = 1:length(T)

    for s = 1:S
    
        for j = 1:length(thetas)
        
            theta = thetas(j);
        
            x_t = zeros(T(i),1); u_t = zeros(T(i),1);    
            u_t(1) = randn(1); x_t(1) = randn(1);
        
                for t = 2:T(i)    
                    u_t(t) = randn(1);
                    x_t(t) = u_t(t) - theta*u_t(t-1);
                end
        
            theta_indirect = indirect_MA(x_t);
        
            
            MC_theta_indirect(s,j,i) = - theta_indirect;        
    
        end
        counter(i) = counter(i) + 1;
        disp(counter);
    end
end


%% Computing mean and SD of MC estimators and plotting it:

% MA:
MA_est = mean(MC_theta_MA);
MA_SD_est = std(MC_theta_MA);
% AR:
AR_est = mean(MC_theta_AR);
AR_SD_est = std(MC_theta_AR);
% inidrect MA:
MA_indi = mean(MC_theta_indirect);
MA_SD_indi = std(MC_theta_indirect);

%% Plot results1:

x = thetas;

figure(1)
% ----------------------------------------------
subplot(3,2,1); % T = 100;  MA1 vs AR1 (MEAN)
plot(x, MA_est(:,:,1) , 'r', 'LineWidth', 2);
hold on;
plot(x, AR_est(:,:,1) , 'b', 'LineWidth', 2);
plot(x, x , 'g--','LineWidth', 2);
hold off;
title('T = 100'); xlabel('true \theta'); ylabel('estimated mean(\theta)');
legend('MA \theta', 'AR \theta', 'true \theta');
grid on;

subplot(3,2,2); % T = 100; MA1 vs AR1 (SD)
plot(x, MA_SD_est(:,:,1), 'r', 'LineWidth', 2);
hold on;
plot(x, AR_SD_est(:,:,1), 'b', 'LineWidth', 2);
hold off;
title('T = 100'); xlabel('true \theta'); ylabel('estimated SD(\theta)');
legend('MA \theta', 'AR \theta');
grid on;

% ----------------------------------------------
subplot(3,2,3); % T = 1000; MA1 vs AR1 (MEAN)
plot(x, MA_est(:,:,2), 'r', 'LineWidth', 2);
hold on;
plot(x, AR_est(:,:,2), 'b', 'LineWidth', 2);
plot(x, x , 'g--','LineWidth', 2);
hold off;
title('T = 1000'); xlabel('true \theta'); ylabel('estimated mean(\theta)');
grid on;

subplot(3,2,4);  % T = 1000; MA1 vs AR1 (SD)
plot(x, MA_SD_est(:,:,2), 'r', 'LineWidth', 2);
hold on;
plot(x, AR_SD_est(:,:,2), 'b', 'LineWidth', 2);
hold off;
title('T = 1000'); xlabel('true \theta'); ylabel('estimated SD(\theta)');
grid on;

% ----------------------------------------------
subplot(3,2,5);  % T = 10000; MA1 vs AR1 (MEAN)
plot(x, MA_est(:,:,3), 'r', 'LineWidth', 2);
hold on;
plot(x, AR_est(:,:,3), 'b', 'LineWidth', 2);
plot(x, x , 'g--','LineWidth', 2);
hold off;
title('T = 10000'); xlabel('true \theta'); ylabel('estimated mean(\theta)');
grid on;

subplot(3,2,6);  % T = 10000; MA1 vs AR1 (SD)
plot(x, MA_SD_est(:,:,3), 'r', 'LineWidth', 2);
hold on;
plot(x, AR_SD_est(:,:,3), 'b', 'LineWidth', 2);
hold off;
title('T = 10000'); xlabel('true \theta'); ylabel('estimated SD(\theta)');
grid on;

% Adjust general title
sgtitle('MA1 vs AR1: Misspecifications Consequences');


%% Plot results2:

x = thetas;

figure(2)
% ----------------------------------------------
subplot(3,2,1); % T = 100;  MA1 vs MA1 indi (MEAN)
plot(x, MA_est(:,:,1) , 'r', 'LineWidth', 2);
hold on;
plot(x, MA_indi(:,:,1) , 'b', 'LineWidth', 2);
plot(x, x , 'g--','LineWidth', 2);
hold off;
title('T = 100'); xlabel('true \theta'); ylabel('estimated mean(\theta)');
legend('MA \theta', 'MA indi.\theta', 'true \theta');
grid on;

subplot(3,2,2); % T = 100; MA1 vs MA1 indi (SD)
plot(x, MA_SD_est(:,:,1), 'r', 'LineWidth', 2);
hold on;
plot(x, MA_SD_indi(:,:,1), 'b', 'LineWidth', 2);
hold off;
title('T = 100'); xlabel('true \theta'); ylabel('estimated SD(\theta)');
legend('MA \theta', 'MA indi. \theta');
grid on;

% ----------------------------------------------
subplot(3,2,3); % T = 1000; MA1 vs AR1 (MEAN)
plot(x, MA_est(:,:,2), 'r', 'LineWidth', 2);
hold on;
plot(x, MA_indi(:,:,2), 'b', 'LineWidth', 2);
plot(x, x , 'g--','LineWidth', 2);
hold off;
title('T = 1000'); xlabel('true \theta'); ylabel('estimated mean(\theta)');
grid on;

subplot(3,2,4);  % T = 1000; MA1 vs AR1 (SD)
plot(x, MA_SD_est(:,:,2), 'r', 'LineWidth', 2);
hold on;
plot(x, MA_SD_indi(:,:,2), 'b', 'LineWidth', 2);
hold off;
title('T = 1000'); xlabel('true \theta'); ylabel('estimated SD(\theta)');
grid on;

% ----------------------------------------------
subplot(3,2,5);  % T = 10000; MA1 vs AR1 (MEAN)
plot(x, MA_indi(:,:,3), 'r', 'LineWidth', 2);
hold on;
plot(x, MA_indi(:,:,3), 'b', 'LineWidth', 2);
plot(x, x , 'g--','LineWidth', 2);
hold off;
title('T = 10000'); xlabel('true \theta'); ylabel('estimated mean(\theta)');
grid on;

subplot(3,2,6);  % T = 10000; MA1 vs AR1 (SD)
plot(x, MA_SD_est(:,:,3), 'r', 'LineWidth', 2);
hold on;
plot(x, MA_SD_indi(:,:,3), 'b', 'LineWidth', 2);
hold off;
title('T = 10000'); xlabel('true \theta'); ylabel('estimated SD(\theta)');
grid on;

% Adjust general title
sgtitle('MA1: Direct vs. Implied estimation');

%% Functions:


% Vamos a hacer el estimador que utilizas en el Implied Computer LAb

function [indirect_est] = indirect_MA(x_t)
    T = length(x_t);
    
    % Calcular sumatorias
    sum1 = zeros(T,2);
    for t = 2:T
        sum1(t,1) = x_t(t) * x_t(t-1);
        sum1(t,2) = x_t(t-1)^2;
    end

    % Calcular rho_til sin signo negativo
    rho_til = sum(sum1(:,1)) / sum(sum1(:,2));

    % Aplicar la función de estimación según la ecuación en la imagen
    if rho_til < -1/2
        indirect_est = -1;
    elseif rho_til >= -1/2 && rho_til <= 0
        indirect_est = (1/(2*rho_til)) + sqrt((1/(2*rho_til)^2) - 1);
    elseif rho_til > 0 && rho_til <= 1/2
        indirect_est = (1/(2*rho_til)) - sqrt((1/(2*rho_til)^2) - 1);
    else
        indirect_est = 1;
    end
end


%--------------------------------------------------------------------------
% Function to maximice: (We are using MA Conditional log Likelihood)
%--------------------------------------------------------------------------
function [criterior_fun] = criterior_function_MA(theta_ML,T,x_t,u_t)

    sigma2 = 1; % This we know.    
    
        % This generates de sumation part of the LF:
    eslavon = zeros(T,1);
    for t = 2:T
        eslavon(t) = ((x_t(t) - theta_ML*u_t(t-1))^2)/(2*sigma2);
    end
    sumation = sum(eslavon);
        % Actual LF:
    Likelihood = -((T)/2)*log(2*pi) - ((T)/2)*log(sigma2) - sumation;

    criterior_fun = - Likelihood;

end

%--------------------------------------------------------------------------

% Function to maximice (AR log likelihood): b)
%--------------------------------------------------------------------------

function [criterior_fun] = criterior_function_AR(param_values,T,x_t)

phi = param_values(1);
sigma2 = param_values(2);

c = 0; % we assume no constant value.


    % Formula Portions:
    term1 = -(T/2) * log(2*pi);
    term2 = -(T/2) * log(sigma2);
    term3 = (1/2) * log(1 - phi^2);
    term4 = - ((x_t(1) - (c / (1 - phi)))^2) / ((2 * sigma2) / (1 - phi^2));
    
            % This generates de sumation part of the LF:
    eslavon = zeros(T,1);
        for t = 2:T
            eslavon(t) = ((x_t(t) - c - phi*x_t(t-1))^2)/(2*sigma2);
        end
    sumation = sum(eslavon);
    
    
    % Final Value
    Likelihood = term1 + term2 + term3 + term4 - sumation;

    criterior_fun = -Likelihood;
end

%--------------------------------------------------------------------------

% Function to maximice (AR log likelihood): c)
%--------------------------------------------------------------------------

% For part c) we slightly modify the Likelihood to set sigma2 = 1
% as we perfectly estimated in part b)


function [criterior_fun] = criterior_function_AR1(theta_ML,T,x_t)

phi = theta_ML;
sigma2 = 1;

c = 0; % we assume no constant value.


    % Formula Portions:
    term1 = -(T/2) * log(2*pi);
    term2 = -(T/2) * log(sigma2);
    term3 = (1/2) * log(1 - phi^2);
    term4 = - ((x_t(1) - (c / (1 - phi)))^2) / ((2 * sigma2) / (1 - phi^2));
    
            % This generates de sumation part of the LF:
    eslavon = zeros(T,1);
        for t = 2:T
            eslavon(t) = ((x_t(t) - c - phi*x_t(t-1))^2)/(2*sigma2);
        end
    sumation = sum(eslavon);
    
    
    % Final Value
    Likelihood = term1 + term2 + term3 + term4 - sumation;

    criterior_fun = -Likelihood;
end

%--------------------------------------------------------------------------




