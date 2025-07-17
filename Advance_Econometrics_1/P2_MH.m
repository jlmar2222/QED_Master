%%  Metropolis-Hastings & a fancy AR(2).


% Consider you have the following AR(2) process:

% y_t = alpha * y_(t-1) + alpha * y_(t-2) + epsilon_t;
% where epsilon_t ~iid N(0, 1 + alpha^2)

%% Putting SEED:

% Uncheck for Monte Carlo simulation(?):
rng(1234); % Establece la semilla en 1234

%% a) Simulate and Plot:


% Write a computer code to simulate data from the model above
% Simulate three samples of size T = 125 of {y_t} with alpha = 0.45
% each one with different init. value:
% 1) y_0 = y_-1 = 0
% 2) y_0 = y_-1 = 25
% 3) y_0 = y_-1 = 100
% Plot them (3 different time seies in same graph)

%T = 5; % Probar prior esta bien calculado
%T = 500; % Probar el prior pierde fuerza
T = 125;

y_t = zeros(T,3);
init_values = [0,25,100;0,25,100];

alpha = 0.45;

% Feed with initial values
for j = 1:length(init_values)
    y_t(1:2,j) = init_values(1:2,j);
end

for t = 3:T    

    for j = 1:length(init_values)    

    % Recall epsilon follows a Noraml with particualr variance (not standar)
    % so we need to transform it:
    % X = mu + sigma*Z; where Z ~N(0,1):
    mu = 0;
    sigma = sqrt(1 + alpha^2);
    y_t(t,j) = alpha*y_t(t-1,j) + alpha*y_t(t-2,j) + (mu + randn(1)*sigma);
    
    end 
end


%% Plot it:

% Dictionary 1 :
x = 1:1:T;
y1 = y_t(:,1);
y2 = y_t(:,2);
y3 = y_t(:,3);

figure(3)
% Graficar las tres series en la misma figura
plot(x, y1, 'r', 'LineWidth', 2); % Serie 1 en rojo
hold on;
plot(x, y2, 'b', 'LineWidth', 2); % Serie 2 en azul
plot(x, y3, 'g', 'LineWidth', 2); % Serie 3 en verde
hold off;

% Personalización de la gráfica
xlabel('Time');
ylabel('Value');
title('AR2 with different init values');
legend('y_0 = y_-1 = 0', 'y_0 = y_-1 = 25', 'y_0 = y_-1 = 100');
grid on;


%% b)c) Metropolis Hasting for alpha:

% Explain how to design a Metropolis-Hastings posterior simulator for
% alpha 
% Include restrictions/PRIORS that would impose stationarity for the AR(2) 

%% Set the different chain sizes:

Rondas = [100,1000,10000];

%% NO PRIORS
% First, to make sure we know what we are doing 
% (also for comparing pourpose)
% We will start making the MH with out priors.
% So our traget is simply the Likelihood function.(Conditional in this Case)


% Also for the whole problem we'll be using a random walk
% as candidates generator so we are on:
% Random Walk Metropolis Hasting.


% (0) Set ini. value for alpha (main coeff.)and Variance of Random Walk.
%
% L00P:
% +---(1) Generate candidate
% |
% +---(2) Compute likelihoods: (Conditional) (Target function)
% |
% +---(3) Compute Acceptance probability (then decide to keep or to drop)



% (0):
alpha_init = 0.2;
Sigma = 0.005;
%R = 10000; Pruebas.

% DB for draws:
draws1 = zeros(max(Rondas),length(Rondas));
Accepted_Proportion = zeros(1,length(Rondas));

for c  = 1:length(Rondas)      
    
    % Set size of chain:
    R = Rondas(c);

    accepted = 0;
    
    % L00P:

    alpha_series = zeros(R,1);
    alpha_series(1,:) = alpha_init;    
    error = randn(R,1)*sqrt(Sigma); 
   
    for r = 2:R

        % (1):  
        alpha_new = alpha_series(r-1) + error(r);
        % (2):
        L_old = Likelihooh_AR2(T,y_t,alpha_series(r-1));
        L_new = Likelihooh_AR2(T,y_t,alpha_new);
    
        % We transform it to exp.
        Ratio = exp(L_new - L_old);
        p_keep = min(Ratio,1);
    
        % (3):    
            if rand(1) > p_keep
            % We drop it (the new one):
            alpha_series(r) = alpha_series(r-1);
            
            else
            % We keep it (the new one):
            alpha_series(r) = alpha_new;
            accepted = accepted + 1;
            end
    
    end
    
    % Acceptance Ratio:    
    Accepted_Proportion(c) = accepted/R;

    % Sample of draws:
    draws1(1:R,c) = alpha_series;


end

%% Plot result:
figure(4)

% This eliminates ceros from series. (clean DB)
a100 = draws1(all(draws1 ~= 0,2),1); 
a100 = a100(50:100,1);
column2 = draws1(:,2); 
a1000 = column2(all(column2 ~= 0,2),1);
a1000 = a1000(100:1000,1);
a10k = draws1(:,3);
a10k = a10k(1000:10000,1);

% Crear la figura con subgráficos
% Define sample sizes
t_100 = 1:100;
t_1000 = 1:1000;
t_10k = 1:10000;

% Crear la figura con subgráficos
% Define sample sizes
t_100 = 50:100;
t_1000 = 100:1000;
t_10k = 1000:10000;

% ----------------------- SERIES 1 -----------------------
subplot(3,2,1); % First row, first column (time series)
plot(t_100, a100, 'r', 'LineWidth', 2);
title('MS chain'); xlabel('Period'); ylabel('Value');
grid on;

subplot(3,2,2); % First row, second column (density)
[f, xi] = ksdensity(a100); % Estimate density of a100
plot(xi, f, 'r', 'LineWidth', 2);
title('Density of MS chain'); xlabel('Value'); ylabel('Density');
grid on;

% ----------------------- SERIES 2 -----------------------
subplot(3,2,3); % Second row, first column (time series)
plot(t_1000, a1000, 'g', 'LineWidth', 2);
grid on;

subplot(3,2,4); % Second row, second column (density)
[f, xi] = ksdensity(a1000);
plot(xi, f, 'g', 'LineWidth', 2);
grid on;

% ----------------------- SERIES 3 -----------------------
subplot(3,2,5); % Third row, first column (time series)
plot(t_10k, a10k, 'b', 'LineWidth', 2);
grid on;

subplot(3,2,6); % Third row, second column (density)
[f, xi] = ksdensity(a10k);
plot(xi, f, 'b', 'LineWidth', 2);
grid on;

% Adjust general title
sgtitle('No priors MH');

%% INFORMATIVE PRIOR (Accounting for Stationarity)

% To adress the Invertibility Concern, we enter Priors.
% The only thong that is going to change (in code) is that we are going to
% use POSTERIOR Distributions (proprtional to: Priors x Likelihood)

% This Prior simply discard Non_Stationary Candidates.
% BUT if Stationary Simply do the normal Likelihood Ratio (as in the example)

% (0):
alpha_init = 0.2;
Sigma = 0.005;
%R = 10000; Pruebas.

% DB for draws:
draws2 = zeros(max(Rondas),length(Rondas));
Accepted2_Proportion = zeros(1,length(Rondas));

for c  = 1:length(Rondas)      
    
    % Set size of chain:
    R = Rondas(c);

    accepted = 0;
    
    % L00P:

    Bs_alpha_series = zeros(R,1);
    Bs_alpha_series(1,:) = alpha_init;    
    error = randn(R,1)*sqrt(Sigma); 
   
    for r = 2:R

        % (1):  
        alpha_new = Bs_alpha_series(r-1) + error(r);

        % check for Sationarity:
        
        if Stationary_Only(alpha_new) == 0 % No stationary.
            % We drop it:
            Bs_alpha_series(r) = Bs_alpha_series(r-1);

        else % Stationary:

            % (2):
            L_old = Likelihooh_AR2(T,y_t,Bs_alpha_series(r-1));
            L_new = Likelihooh_AR2(T,y_t,alpha_new);
        
            % We transform it to exp.
            Ratio = exp(L_new - L_old);
            p_keep = min(Ratio,1);
        
            % (3):    
                if rand(1) > p_keep
                % We drop it (the new one):
                Bs_alpha_series(r) = Bs_alpha_series(r-1);
                
                else
                % We keep it (the new one):
                Bs_alpha_series(r) = alpha_new;
                accepted = accepted + 1;
                end    

        end        

    end
    
    % Acceptance Ratio:    
    Accepted2_Proportion(c) = accepted/R;

    % Sample of draws:
    draws2(1:R,c) = Bs_alpha_series;


end

%% Plot result:
figure(5)

% This eliminates ceros from series. (clean DB)
b100 = draws2(all(draws2 ~= 0,2),1); 
b100 = b100(50:100,1);
column2 = draws2(:,2); 
b1000 = column2(all(column2 ~= 0,2),1);
b1000 = b1000(100:1000,1);
b10k = draws2(:,3);
b10k = b10k(1000:10000,1);



% Crear la figura con subgráficos
% Define sample sizes
t_100 = 1:100;
t_1000 = 1:1000;
t_10k = 1:10000;

% Crear la figura con subgráficos
% Define sample sizes
t_100 = 50:100;
t_1000 = 100:1000;
t_10k = 1000:10000;



% ----------------------- SERIES 1 -----------------------
subplot(3,2,1); % First row, first column (time series)
plot(t_100, b100, 'r', 'LineWidth', 2);
title('MS chain'); xlabel('Period'); ylabel('Value');
grid on;

subplot(3,2,2); % First row, second column (density)
[f, xi] = ksdensity(b100); % Estimate density of a100
plot(xi, f, 'r', 'LineWidth', 2);
title('Density of MS chain'); xlabel('Value'); ylabel('Density');
grid on;

% ----------------------- SERIES 2 -----------------------
subplot(3,2,3); % Second row, first column (time series)
plot(t_1000, b1000, 'g', 'LineWidth', 2);
grid on;

subplot(3,2,4); % Second row, second column (density)
[f, xi] = ksdensity(b1000);
plot(xi, f, 'g', 'LineWidth', 2);
grid on;

% ----------------------- SERIES 3 -----------------------
subplot(3,2,5); % Third row, first column (time series)
plot(t_10k, b10k, 'b', 'LineWidth', 2);
grid on;

subplot(3,2,6); % Third row, second column (density)
[f, xi] = ksdensity(b10k);
plot(xi, f, 'b', 'LineWidth', 2);
grid on;

% Adjust general title
sgtitle('Non-Stationary Prior MH');


%% INFORMATIVE PRIOR (Normal Distributed)

% To adress the Invertibility Concern, we enter Priors.
% The only thong that is going to change (in code) is that we are going to
% use POSTERIOR Distributions (proprtional to: Priors x Likelihood)

% (0):
alpha_init = 0.2;
Sigma = 0.0005;
%R = 10000; Pruebas.

% DB for draws:
draws3 = zeros(max(Rondas),length(Rondas));
Accepted3_Proportion = zeros(1,length(Rondas));

for c  = 1:length(Rondas)      
    
    % Set size of chain:
    R = Rondas(c);

    accepted = 0;
    
    % L00P:

    BN_alpha_series = zeros(R,1);
    BN_alpha_series(1,:) = alpha_init;    
    error = randn(R,1)*sqrt(Sigma); 
   
    for r = 2:R
    
        % (1):  
        alpha_new = BN_alpha_series(r-1) + error(r);
        % (2): 
        L_old = log(Normal_prior(BN_alpha_series(r-1))) + Likelihooh_AR2(T,y_t,BN_alpha_series(r-1));
        L_new = log(Normal_prior(alpha_new)) + Likelihooh_AR2(T,y_t,alpha_new);
    
        % We transform it to exp.
        Ratio = exp(L_new - L_old);
        p_keep = min(Ratio,1);
    
        % (3):    
            if rand(1) > p_keep
            % We drop it (the new one):
            BN_alpha_series(r) = BN_alpha_series(r-1);
            
            else
            % We keep it (the new one):
            BN_alpha_series(r) = alpha_new;  
            accepted = accepted + 1;
            end
    
    end
    
    % Acceptance Ratio:    
    Accepted3_Proportion(c) = accepted/R;

    % Sample of draws:
    draws3(1:R,c) = BN_alpha_series;


end

%% Plot result:
figure(6)

% This eliminates ceros from series. (clean DB)
c100 = draws3(all(draws3 ~= 0,2),1); 
c100 = c100(50:100,1);
column2 = draws3(:,2); 
c1000 = column2(all(column2 ~= 0,2),1);
c1000 = c1000(100:1000,1);
c10k = draws3(:,3);
c10k = c10k(1000:10000,1);



% Crear la figura con subgráficos
% Define sample sizes
t_100 = 1:100;
t_1000 = 1:1000;
t_10k = 1:10000;

% Crear la figura con subgráficos
% Define sample sizes
t_100 = 50:100;
t_1000 = 100:1000;
t_10k = 1000:10000;



% ----------------------- SERIES 1 -----------------------
subplot(3,2,1); % First row, first column (time series)
plot(t_100, c100, 'r', 'LineWidth', 2);
title('MS chain'); xlabel('Period'); ylabel('Value');
grid on;

subplot(3,2,2); % First row, second column (density)
[f, xi] = ksdensity(c100); % Estimate density of a100
plot(xi, f, 'r', 'LineWidth', 2);
title('Density of MS chain'); xlabel('Value'); ylabel('Density');
grid on;

% ----------------------- SERIES 2 -----------------------
subplot(3,2,3); % Second row, first column (time series)
plot(t_1000, c1000, 'g', 'LineWidth', 2);
grid on;

subplot(3,2,4); % Second row, second column (density)
[f, xi] = ksdensity(c1000);
plot(xi, f, 'g', 'LineWidth', 2);
grid on;

% ----------------------- SERIES 3 -----------------------
subplot(3,2,5); % Third row, first column (time series)
plot(t_10k, c10k, 'b', 'LineWidth', 2);
grid on;

subplot(3,2,6); % Third row, second column (density)
[f, xi] = ksdensity(c10k);
plot(xi, f, 'b', 'LineWidth', 2);
grid on;

% Adjust general title
sgtitle('Normal Distributed Prior (~N(0,0.01^2)) MH');



%% d) INFORMATIVE PRIOR (Beta(10,10) Distributed == alpha in (0,0.5))

% Suppose we know that from a previous study alpha is in (0,0.5) and we want
% to impose that constraint. 
% Run the code in (c) but now using as prior p(alpha) ~ Beta(10,10)
% Report the posterior distribution using a nonparametric estimate, 
% showing the results for chains of 1,000  10,000 and 100,000 draws. 
% Discuss the results in (c) and (d)


% (0):
alpha_init = 0.2;
Sigma = 0.005;
%R = 10000; Pruebas.

% DB for draws:
draws4 = zeros(max(Rondas),length(Rondas));
Accepted4_Proportion = zeros(1,length(Rondas));

for c  = 1:length(Rondas)      
    
    % Set size of chain:
    R = Rondas(c);

    accepted = 0;
    
    % L00P:

    BB_alpha_series = zeros(R,1);
    BB_alpha_series(1,:) = alpha_init;    
    error = randn(R,1)*sqrt(Sigma); 
   
    for r = 2:R
    
        % (1):  
        alpha_new = BB_alpha_series(r-1) + error(r);
        % (2): 
        L_old = log(Beta_prior(10,10,BB_alpha_series(r-1))) + Likelihooh_AR2(T,y_t,BB_alpha_series(r-1));
        L_new = log(Beta_prior(10,10,alpha_new)) + Likelihooh_AR2(T,y_t,alpha_new);
    
        % We transform it to exp.
        Ratio = exp(L_new - L_old);
        p_keep = min(Ratio,1);
    
        % (3):    
            if rand(1) > p_keep
            % We drop it (the new one):
            BB_alpha_series(r) = BB_alpha_series(r-1);
            
            else
            % We keep it (the new one):
            BB_alpha_series(r) = alpha_new;  
            accepted = accepted + 1;
            end
    
    end
    
    % Acceptance Ratio:    
    Accepted4_Proportion(c) = accepted/R;

    % Sample of draws:
    draws4(1:R,c) = BB_alpha_series;


end

%% Plot result:
figure(7)

% This eliminates ceros from series. (clean DB)
d100 = draws4(all(draws4 ~= 0,2),1); 
d100 = d100(50:100,1);
column2 = draws4(:,2); 
d1000 = column2(all(column2 ~= 0,2),1);
d1000 = d1000(100:1000,1);
d10k = draws4(:,3);
d10k = d10k(1000:10000,1);

% Crear la figura con subgráficos
% Define sample sizes
t_100 = 1:100;
t_1000 = 1:1000;
t_10k = 1:10000;

% Crear la figura con subgráficos
% Define sample sizes
t_100 = 50:100;
t_1000 = 100:1000;
t_10k = 1000:10000;


% ----------------------- SERIES 1 -----------------------
subplot(3,2,1); % First row, first column (time series)
plot(t_100, d100, 'r', 'LineWidth', 2);
title('MS chain'); xlabel('Period'); ylabel('Value');
grid on;

subplot(3,2,2); % First row, second column (density)
[f, xi] = ksdensity(d100); % Estimate density of a100
plot(xi, f, 'r', 'LineWidth', 2);
title('Density of MS chain'); xlabel('Value'); ylabel('Density');
grid on;

% ----------------------- SERIES 2 -----------------------
subplot(3,2,3); % Second row, first column (time series)
plot(t_1000, d1000, 'g', 'LineWidth', 2);
grid on;

subplot(3,2,4); % Second row, second column (density)
[f, xi] = ksdensity(d1000);
plot(xi, f, 'g', 'LineWidth', 2);
grid on;

% ----------------------- SERIES 3 -----------------------
subplot(3,2,5); % Third row, first column (time series)
plot(t_10k, d10k, 'b', 'LineWidth', 2);
grid on;

subplot(3,2,6); % Third row, second column (density)
[f, xi] = ksdensity(d10k);
plot(xi, f, 'b', 'LineWidth', 2);
grid on;

% Adjust general title
sgtitle('Beta Distributed Prior (~\beta(10,10))');


%% Functions:

% Computing Beta Prior distributed: 

function [result] = B(a,b)
    
    incremento = 0.01;

    x = 0:incremento:1;
    x = x';
    
    results = zeros(length(x),1);
    for i = 1:length(x)
        results(i) = (x(i)^(a-1))*((1-x(i))^(b-1))*incremento;            
    end
    result = sum(results);
end


function [prior_distribution] = Beta_prior(a,b,alpha)
    
    % Set a=b=10 for the exercise --> Beta(10,10)

    term1 = (alpha^(a-1))*((1-alpha)^(b-1));
    term2 = B(a,b); % B is an Intedration(0,1) defined above

    prior_distribution = term1/term2;

end

% Computing Stationary Only restircction/Prior:

function [prior_distribution] = Stationary_Only(alpha)

% A way of checking Stationarity is cheking that:
% 1 - alpha*z - alpha*z^2 must have all roots with |z| > 1

% We are generating the following Prior:
%
% p(alpha) = 1{|z| > 1 for all z in: 1 - alpha*z - alpha*z^2}

coeffs = [-alpha -alpha 1]; % roots order: x^2,x,const
% Compute z (roots)
roots_z = roots(coeffs);

    if all(abs(roots_z) > 1)
        prior_distribution = 1;
    else
        prior_distribution = 0;
    end

% Recall this prior is not usual('proper') in the sence that it only makes 
% imposible to select non-stationary draws
% if we get stationarity then the acceptance probability
% will be simply the usual Likelihood Ratio.

end




% Computing Normal Prior Distribution for Alfa:

function [prior_distribution] = Normal_prior(alpha)

% We assume Alfa follows a Normal distribution with:
% N(0, 0.01^2) (Strong Prior)
alpha_0 = 0; 
Sigma_0 = 0.01^2;

term1 = (1/(sqrt(2*pi*(Sigma_0))));
term2 = exp(-(((alpha-alpha_0)^2)/(2*Sigma_0)));

prior_distribution = term1*term2;


end


% Conditional Likelihood funciton for AR2:

function [Likelihood] = Likelihooh_AR2(T,y_t,alpha)

y_t0 = y_t(:,1); % Condition on y_0 = y_-1 = 0 % Clever is to use the DGP where this is the cas:

sigma2 = 1+alpha^2;

term1 = -((T-2)/2)*log(2*pi);
term2 = -((T-2)/2)*log(sigma2);

chain = zeros(T-3,1);
for t = 3:T
    eslavon = (y_t0(t) - alpha*y_t0(t-1)-alpha*y_t0(t-2))^2/(2*sigma2);
    chain(t) = eslavon;
end

Likelihood = term1 + term2 - sum(chain);

end





