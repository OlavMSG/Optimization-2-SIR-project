function SIRsolver()

    clc; 
        
    % -------- Temporal parameters (measured in days) ---------------
    t = 0; % Simulation Start time 
    T = 800; % Simulation End time
    h = 1; % timestep size; h = 1 means a step-size of 1 day
    M = (T-t)/h; % Number of timesteps 
    tlist = t:h:T;
    
    % -------- system parameters ------------------------------------
    K = 4; % Basic reproduction number
    gamma = 0.07; % Disease recovery rate 
    v = 151/5328000; % Birth/Death rate (they're equal) 
    beta = K*(gamma+v); % Disease reproduction rate
    
    % -------- Initialize system ------------------------------------
    S0 = 5328000; % Initial number of susceptible individuals 
    I0 = 1; % Initial number of infected individuals 
    R0 = 0; % Initial number of recovered / immune individuals 
    N = S0+I0+R0; % Total population size 
    Y = zeros(3,M+1); % Preallocate state vector
    Y(:,1) = [S0;I0;R0]; % Initialize state vector 
    
    % Initialize vaccination protocol
    kmax = 1*1000; % Set constant/max number of people being vaccinated per timestep 
    U = zeros(1,M+1); % Preallocate vector to hold u(t) 
    protocol = "ramping"; % Assign vaccination protocol ("constant" or "ramping")
    vaxStart = 365; % Number of timesteps into the simulation when the vaccine protocol starts
    vaxEnd = vaxStart + 180; % Number of timesteps into the simulation when the vaccine protocol is at full capacity
    um = vaccinationProtocol(S0,kmax,tlist,1,vaxStart,vaxEnd,protocol); % get first value of u(t)
    U(1) = um; % Assign the first value of u(t) to global list 
    
    % Run RK4 solver 
    for m = 1:M
        % Get solution at current timestep
        Ym = Y(:,m);
        
        % Compute the four-stage approximations to the derivative 
        K1 = F(Ym,um,N,beta,gamma,v);
        K2 = F(Ym+0.5*K1,um,N,beta,gamma,v);
        K3 = F(Ym+0.5*K2,um,N,beta,gamma,v);
        K4 = F(Ym+K3,um,N,beta,gamma,v);
        
        % Update solution
        Y(:,m+1) = Ym + h*(K1+2*K2+2*K3+K4)/6;
        
        % Get updated value for u(t) and assign it to list
        um = vaccinationProtocol(Y(1,m+1),kmax,tlist,m+1,vaxStart,vaxEnd,protocol);
        U(m+1) = um;
    end
    
    % Plot results 
    fig = figure;
    set(fig,'defaulttextinterpreter','latex');
    plot(tlist,Y(1,:)/N,'-ob','markerindices',1:int32(M/20):M,'linewidth',1.5); hold on;
    plot(tlist,Y(2,:)/N,'-*r','markerindices',1:int32(M/20):M,'linewidth',1.5); hold on;
    plot(tlist,Y(3,:)/N,'-+','color',[27,146,51]./255,'markerindices',1:int32(M/20):M,'linewidth',1.5); hold on;
    plot(tlist,U,'-k','linewidth',1.5); hold on;
    if strcmpi(protocol,"ramping") && kmax ~= 0
        plot(365*ones(M,1),linspace(0,1,M),'--k'); hold on;
        plot(545*ones(M,1),linspace(0,1,M),'--k'); hold on;
    end
    legend({'$S(t)$','$I(t)$','$R(t)$','$u(t)$'},'fontsize',20,'interpreter','latex');
    xlim([0,tlist(M)+1]);
    ylim([0,1]);
    xlabel('$t$','fontsize',30)
    if strcmpi(protocol,"ramping")
        title(['Ramping Vaccination protocol, $k_{max}$ = ',num2str(kmax)],'fontsize',25);
    else
        title(['Vaccinating ',num2str(kmax),' people per day'],'fontsize',30);
    end
    
end

% Function to evaluate RHS of system of differential equations 
function sysVec = F(Ym,um,N,beta,gamma,v)
    sysVec = zeros(3,1);
    sysVec(1) = v*N - (v+um)*Ym(1) - (beta/N)*Ym(1)*Ym(2);
    sysVec(2) = (beta/N)*Ym(1)*Ym(2) - (gamma+v)*Ym(2);
    sysVec(3) = gamma*Ym(2) - v*Ym(3) + um*Ym(1);
end

% Vaccination protocol 
function u = vaccinationProtocol(Ym,kmax,tlist,m,vaxStart,vaxEnd,protocol)

    if strcmpi(protocol,"ramping")
        % Ramping vaccination protocol
        
        % Set number of people being vaccinated to zero by default
        k = 0;
        
        % Check if vaccination protocol has started 
        if tlist(m) >= vaxStart
            
            % Check if we've reached max capacity
            if tlist(m) <= vaxEnd
                
                % Not at max capacity; computing current ramping capacity  
                a = kmax/(vaxEnd-vaxStart);
                b = kmax-((kmax*vaxEnd)/(vaxEnd-vaxStart));
                k = a*tlist(m) + b;
                
            else
                
                % Vaccinating at full capacity
                k = kmax;
                
            end
        end
        
        % Compute u(t) given S(t) and number of people being vaccinated k
        u = k/Ym(1);
        
    else
        % Constant vaccination protocol
        u = kmax/Ym(1);
    end
    
    % Enforce 0 <= u(t) <= 0.9
    if u > 0.9
        u = 0.9;
    end
    
end
