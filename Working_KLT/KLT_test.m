clear all % clear all variables in the workspace
close all % close all figures
clc       % clear the command window

%loading parking dataset and stores images in array
addpath('../parking/images/')
for i = 0:598
    Vmovie(:,:,i+1) = rgb2gray(imread(sprintf('../parking/images/img_%05d.png', i)));
end
% Get the first movie frame
I = double(Vmovie(:,:,1))*(1/255);

% Show the movie frame
figure, imshow(I,[])
title(['Frame ' num2str(1)]) % Display the frame number on top of the image
set(gcf,'Color','w'); axis on,

% Select the coordinates of several templates
% rect=getrect; TemplateRect{1}=round([rect(2) rect(2)+rect(4);rect(1) rect(1)+rect(3);]);
TemplateRect{1} = [400,440; 500,580];
TemplateRect{2} = [030,091; 370,500];
TemplateRect{3} = [093,103; 600,630];
TemplateRect{4} = [154,171; 550,610];

% Plot templates
color = rand(4,3);
for k=1:numel(TemplateRect),
    v = flipud(TemplateRect{k});
    v = [v(1,1) v(1,1) v(1,2) v(1,2);
         v(2,1) v(2,2) v(2,2) v(2,1)];
    hold on,
    plot_quadrilateral(v, color(k,:));
end

% Save image and to disk
set(gcf,'PaperPositionMode','auto')
print('-dpng','-r0',['picture' num2str(1)])

% Pad templates with extra boundary pixels (not used for the actual template
% tracking, but for smooth image derivatives).
b = 5; % pixels
padding = [-b,b;-b,b];
for k=1:numel(TemplateRect),
    TemplateRect{k} = TemplateRect{k} + padding;
end


% -Set initial parameters of the templates
% -Set padded template image.
% 
% Affine Transformation Matrix is used in Lucas Kanade Tracking
% with 6 parameters
% M    = [ 1+p(1) p(3)   p(5); 
%          p(2)   1+p(4) p(6); 
%          0      0      1];
%

% Make a struct to store templates
TemplateData = struct;
for k=1:numel(TemplateRect)
    w = TemplateRect{k};
    center = [w(1,1)+w(1,2)-1 w(2,1)+w(2,2)-1]/2; % center of the template, in image coordinates.
    TemplateData(k).p = [0 0 0 0 center(1) center(2)]; % parameters for the affine warp of the template
    TemplateData(k).image = I(w(1,1):w(1,2), w(2,1):w(2,2)); % pixels of the template
end

%%
% LK Tracking Options (default values for other options)
Options.TranslationIterations = 15;
Options.AffineIterations = 5;

% Make a colormap
cmap = hot(256);

% Matrix to store squared pixel error between template and ROI in
% movieframe after template tracking.
T_error = zeros(size(Vmovie,3), length(TemplateData));


% Loop through the movie frames
% L = size(Vmovie,3);
L = 50;
for i=1:L
    % Get the a movie frame
    I = double(Vmovie(:,:,i))*(1/255);
    
    % Do the tracking for all templates, using the Lucas-Kanade method
    for k=1:length(TemplateData)
        [TemplateData(k).p,ROIimage,T_error(i,k)] = ...
            LucasKanadeAffine(I,TemplateData(k).p,TemplateData(k).image,Options);
    end
    
    
    % Show the location of the templates in the movie frame
    figure(1), imshow(I); % Show the movie frame
    title(['Frame ' num2str(i)]) % Display the frame number on top of the image
    set(gcf,'Color','w'); axis on,
    hold on
    
    % Display the tracked templates
    for k=1:length(TemplateData)
        plot(TemplateData(k).p(6),TemplateData(k).p(5),'go',...
            'MarkerFaceColor',cmap(round(255 * k/length(TemplateData))+1,:));

        % Get vertices of the template
        [template_height, template_width] = size(TemplateData(k).image);
        halfw = template_width/2 - b;
        halfh = template_height/2 - b;
        v = [halfw, -halfw, -halfw,  halfw;
             halfh,  halfh, -halfh, -halfh];
        % Compute the position of the vertives of the (warped) template...
        v = v + [TemplateData(k).p(6); TemplateData(k).p(5)] * ones(1,4);
        
        % Plot vertices of (warped) template
        plot_quadrilateral(v, color(k,:));
    end
    drawnow
    
    if(mod(i,10)==0)
	% Save image and to disk
        set(gcf,'PaperPositionMode','auto')
        print('-dpng','-r0',['picture' num2str(i)])
    end
end
