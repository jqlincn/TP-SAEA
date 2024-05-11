function index = RefSelection(PopObj,V,mu,theta)
% Reference selection

% This function is written by Jianqing Lin

    [NVa,va]  = NoActive(PopObj,V);
    NCluster  = min(mu,size(V,1)-NVa);
    Va        = V(va,:);
    [IDX,~]   = kmeans(Va,NCluster);    

    PopObj = PopObj - repmat(min(PopObj,[],1),size(PopObj,1),1);
    cosine = 1 - pdist2(Va,Va,'cosine');
    cosine(logical(eye(length(cosine)))) = 0;
    gamma  = min(acos(cosine),[],2);
    Angle  = acos(1-pdist2(PopObj,Va,'cosine'));
    [~,associate] = min(Angle,[],2);
    Cindex = IDX(associate); 

    Next = zeros(NCluster,1);
    for i = unique(Cindex)'
        current = find(Cindex==i);
        % Calculate the APD value of each solution
        APD = (1+size(PopObj,2)*theta*Angle(current,i)/gamma(i)).*sqrt(sum(PopObj(current,:).^2,2));
        % Select the one with the minimum APD value
        [~,best] = min(APD);
        Next(i)  = current(best);
    end
    index  = Next(Next~=0);
end

function [num,active] = NoActive(PopObj,V)

    [N,~] = size(PopObj);
    NV    = size(V,1);
    
    %% Translate the population
    PopObj = PopObj - repmat(min(PopObj,[],1),N,1);

    %% Associate each solution to a reference vector
    Angle   = acos(1-pdist2(PopObj,V,'cosine'));
    [~,associate] = min(Angle,[],2);
    active  = unique(associate);
	num     = NV-length(active);
end