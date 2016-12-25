function [candidatedes,idx ] = rmdoubledes( candidatedes, removedes )
%removes descriptors that already have 3d-point
for i=1:length(candidatedes(1,:))
    no_duplicate = 1;
    for ii=1:length(removedes(1,:))
        if candidatedes(3:end,i)==removedes(3:end,ii)
            no_duplicate = 0;
        end
    end
    are_unique(i)=no_duplicate;
    
end
candidatedes = candidatedes(:,are_unique>0);
idx = are_unique(:,are_unique>0);
end

