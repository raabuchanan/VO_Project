function [matching,validx] = triagulatedStates(triState,tri_threshold,match_lambda)
% Returns a 1xQ matrix where the i-th coefficient is the index of the
% State which matches to the i-th State and are vaild for triangulation.
%

%% Calculate distances between poses
num_candidates = length(triState(1,:));

% Creating an array with number of instances of each pose
values = unique(triState(end,:));
inst = histc(triState(end,:),values);
validindex = 0;
p = triState(end-2:end,end);
validx=0;
for i=length(inst):-1:1
    validx = validx + inst(i);
    pi = triState(end-2:end,end-validx+1);
    pos2pos = sqrt((p(1)-pi(1))^2+(p(2)-pi(2))^2+(p(3)-pi(3))^2);
    if pos2pos > tri_threshold
        validindex = num_candidates-sum(inst(end+1-i));
        break
    end
end

%extracts first descriptors from same pose
if validindex>0
    matching = matchDescriptors(...
        uint8(triState(1:end-14,validindex+1:end)),...
        uint8(triState(1:end-14,1:validindex)),match_lambda)';
else
    matching=0;
end

end

