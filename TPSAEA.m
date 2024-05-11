classdef TPSAEA < ALGORITHM
% <multi/many> <real/integer> <expensive>
% Surrogate-assisted Reformulation and Decomposition
% wD       --- 5 --- The number of reference solutions
% k        --- 5 --- The number of re-evaluated solutions
% SubN     --- 50 --- The population size of the transferred problem
% G2       --- 50 --- The number of iterations
% alpha    --- 0.7 --- The control parameter

%------------------------------- Reference --------------------------------
% L. Pan, J. Lin, H. Wang, C. He, K.C. Tan, and Y. Jin, Computationally 
% Expensive High-dimensional Multiobjective Optimization via Surrogate-
% assisted Reformulation and Decomposition, IEEE Transactions on 
% Evolutionary Computation, 2024.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2023 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Jianqing Lin

    methods
        function main(Algorithm,Problem)

            warning('off')
            %% Parameter setting
            [wD,k,SubN,G2,alpha]   = Algorithm.ParameterSet(5,5,50,50,0.7);

            %% Generate the reference points and population
            [V0,SubN] = UniformPoint(SubN,Problem.M);
	        V         = V0;

            NI    = 11*(2*wD)-1;
            P     = UniformPoint(NI,Problem.D,'Latin');
            A1    = Problem.Evaluation(repmat(Problem.upper-Problem.lower,NI,1).*P+repmat(Problem.lower,NI,1));

            %% Optimization
            while Algorithm.NotTerminated(A1)
                if Problem.FE <=  alpha * Problem.maxFE
                    [A2,V] =  PRSAEA(Problem,G2,A1,wD,SubN,V0,V,k);
                    A1  = [A1,A2];
                else
                    A2  = DESAEA(Problem,A1,SubN,k,0.5,min(300,size(A1,2)));
                    A1  = [A1,A2];
                end
            end
        end
    end
end