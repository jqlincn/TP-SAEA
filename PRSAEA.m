function [A2,V] = PRSAEA(Problem,G2,A1,wD,N,V0,V,k)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2018-2019 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Jianqing Lin

	%% reference of HV
	Reference    = max(A1.objs,[],1);
    
    %% Choose wD solutions as the reference solutions
    RefPop  = A1(RefSelection(A1.objs,V,wD,2));

    %% Calculate the directions
	lower     = Problem.lower;
    upper     = Problem.upper;
    if size(RefPop,2) ~= wD
        wD = size(RefPop,2);
    end
	Direction = [sum((RefPop.decs-repmat(lower,wD,1)).^2,2).^(0.5);sum((repmat(upper,wD,1)-RefPop.decs).^2,2).^(0.5)];
    
    % Prevent the base from being 0
    if all(Direction) == 0
        ind = find(Direction == 0);
        Direction(ind) = Direction(ind)+0.0001;
    end
    
	Direct    = [(RefPop.decs-repmat(lower,wD,1));(repmat(upper,wD,1)-RefPop.decs)]./repmat(Direction,1,Problem.D);
	wmax      = sum((upper-lower).^2)^(0.5)*0.5;
    
    %% Create RBF models
    [Populationdesc,indx,~] = unique(A1.decs,'rows');
    Populationobjs = A1(indx).objs;
    
    RBF_para = cell(1,Problem.M);
    for i = 1:Problem.M
        RBF_para{i} = RBF_Create(Populationdesc, Populationobjs(:,i), 'cubic');
    end
    
    %% Optimize the weight variables by DE
	w0 = rand(N,2*wD).*wmax;                                            % Initialize the population
    [fitness,~] = fitfunc(w0,Direct,Problem,Reference,RBF_para);	        % Calculate the fitness and store the solutions
    
	pCR = 0.2;
    beta_min=0.2;   % Lower Bound of Scaling Factor
    beta_max=0.8;   % Upper Bound of Scaling Factor
    empty_individual.Position=[];
    empty_individual.Cost =[];
    pop=repmat(empty_individual,N,1);
    
    %% Create kriging models
    [Kriging_para, ~] = dacefit(w0, fitness,'regpoly0','corrgauss',1*ones(1,2*wD), 0.001*ones(1,2*wD), wmax*ones(1,2*wD));

    %% Define the optimal solution evaluated by RBF
    Opt_pop.Position = [];   
    Opt_pop.Cost     = inf;
    for i = 1 : N
        pop(i).Position = w0(i,:);
        pop(i).Cost = fitness(i);
        if pop(i).Cost < Opt_pop.Cost
            Opt_pop = pop(i);
        end
    end
    
    %% Evolution
    for it = 1 : G2
        for i = 1 : N
            x = pop(i).Position;
            A = randperm(N);
            A(A==i) = [];
            a = A(1); b = A(2); c = A(3);
            % Mutation
            beta = unifrnd(beta_min,beta_max,[1 2*wD]);
            y = pop(a).Position + beta.*(pop(b).Position - pop(c).Position);
            y = min(max(y,0),wmax);
            % Crossover
            z = zeros(size(x));
            j0=randi([1 numel(x)]);
            for j=1:numel(x)
                if j==j0 || rand<=pCR
                    z(j) = y(j);
                else
                    z(j) = x(j);
                end
            end
            NewSol.Position = z;
            
            [Obj,~] = predictor(z,Kriging_para);
            NewSol.Cost  = Obj;
                        
            if NewSol.Cost < pop(i).Cost
                pop(i)=NewSol;
                if NewSol.Cost < Opt_pop.Cost
                    Opt_pop = NewSol;
                end
            end
        end
    end
    
    [~,OffSpring] = fitfunc(Opt_pop.Position,Direct,Problem,Reference,RBF_para);

    % Update and store the non-dominated solutions
    if size(OffSpring,2) < k
        k = size(OffSpring,2);
    end
    [PopNew,~,~] = EnvironmentalSelection(OffSpring,k);
    A2  = Problem.Evaluation(PopNew.decs);    
    % Update V
    V(1:size(V0,1),:) = V0.*repmat(max(A2.objs,[],1)-min(A2.objs,[],1),size(V0,1),1);
    
end

function [Obj,OffSpring] = fitfunc(w0,direct,Global,Reference,RBF_para)
    [SubN,WD] = size(w0); 
    WD      = WD/2;
    Obj   	= zeros(SubN,1);
    OffSpring  = [];
    for i = 1 : SubN 
        PopDec    = [repmat(w0(i,1:WD)',1,Global.D).*direct(1:WD,:)+repmat(Global.lower,WD,1);
                      repmat(Global.upper,WD,1) - repmat(w0(i,WD+1:end)',1,Global.D).*direct(WD+1:end,:)];  % Problem transform back
                  
        PopObj    = RBF_Predictor(PopDec, RBF_para, Global.M);
        OffWPop   = surrogate_individual(PopDec,PopObj);
        
        OffSpring = [OffSpring,OffWPop];
        Obj(i)    = -HV(OffWPop.objs,Reference);
    end
end