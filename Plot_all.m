
figure(1)
set(gcf, 'Position', get(0,'Screensize')); % Maximize figure. 

%    %%plot image with keypoints and landmarks
pos = -R_C_W'*t_C_W;
Cursate_pos =  currState;

subplot('Position',[0.1 0.55 0.4 0.4])
if exist('imHandle')
    delete(imHandle);
end
imHandle = imshow(currImage);    
hold on;
plot(currState(2, :),currState(1, :), 'gx', 'MarkerSize',3);
if ~isempty(dataBase{1,end})
    if exist('keypointHandle')
        delete(keypointHandle);
    end
    keypointHandle = plot(dataBase{1,end}(2,:),dataBase{1,end}(1,:),'rx', 'MarkerSize',2);
end
title('Keypoints (RED), Landmarks (GREEN)')

%%% plot trajectory of last 20 frames and landmarks
alltraj(:,ii) = pos; 
num_trck_lnd(1,ii) = size(Cursate_pos(1,:),2);%vector of number of tracked landmarks at each frame

if ii<22
    last20 =  [alltraj(1,:);alltraj(3,:)];
    num_trck = num_trck_lnd; %for plotting number of tracked landmarks
    framevec = 1:ii;
else
    last20 = [alltraj(1,end-20:end);alltraj(3,end-20:end)];% get last 20 positions

    num_trck = num_trck_lnd(1,(end-20):end);%for plotting number of tracked landmarks
    framevec = (ii - 20):ii;

end



subplot('Position',[0.1 0.1 0.4 0.4])
plot(smooth(last20(1,:),10),smooth(last20(2,:),10), '-x','MarkerSize', 2) %plot last 20 positions
hold on
scatter(Cursate_pos(3, :), Cursate_pos(5, :), 4, 'k');% plot currently tracked landmarks 
set(gcf, 'GraphicsSmoothing', 'on');
view(0,90);

if min(last20(2,:))<min(Cursate_pos(5, :)) %set axes of plot based on current trajectory of camera
   zmin = min(last20(2,:)) - 10;
   zmax = zmin + 80; %may need to change or we may need to set scale as part of initialization
   xmin =  median(Cursate_pos(3,:))-30;
   xmax = median(Cursate_pos(3,:))+30;

else
   zmin = median(Cursate_pos(5, :))-40;
   zmax = median(Cursate_pos(5, :))+40;
   xmin = min(last20(1,:));
   xmax = xmin + 60;
end   
   axis([xmin xmax  zmin zmax]);
   title('Landmarks and Trajectory Last 20 Frames')

   hold off
%%%%%%%plot total trajectory
subplot('Position',[0.55 0.55 0.4 0.4])

plot(smooth(alltraj(1,:),10),smooth(alltraj(3,:),10), '-')
axis equal;
title('Full Trajectory')

%%%% number of landmarks tracked over past 20 frames

subplot('Position',[0.55 0.1 0.4 0.4])
hold on
plot(framevec, num_trck, '-');
title('Landmarks Tracked over Past Frames')
        
        
        
       
        