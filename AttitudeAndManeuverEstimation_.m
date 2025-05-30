% Specify the image folder and output folder for saving the figures
imageFolder = 'C:\';
outputFolder = 'C:\';

% Create output folder if it doesn't exist
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Create the image datastore for the images in the folder
imds = imageDatastore(imageFolder);

% Loop through each image in the datastore and perform functions
% After performing functions, save the plots
% This allows animations to be made
for i = 1:length(imds.Files)
    % Read the current image
    I = readimage(imds, i);
    
    % Test the semantic segmentation network
    C = semanticseg(I, trainedNet, Classes=classes);
    
    % Make a colormap for the overlay
    cmap = [
        0 0 255;    % not satellite (blue)
        0 255 0;    % body (green)
        255 0 0     % panel (red)
    ] ./ 255;
    
    % Overlay the colormap on the segmented image
    B = labeloverlay(I, C, Colormap=cmap, Transparency=0.4);
    
    % Create a figure with four subplots
    figure;
    
    % Subplot 1: Original Image
    subplot(3, 2, 1);
    imshow(I);
    title('Original Image');
    
    % Subplot 2: Segmented Image
    subplot(3, 2, 2);
    imshow(B);
    title('Segmented Image');
    hold on;
    
    % Display a legend
    colormap(gca, cmap);
    c = colorbar('peer', gca);
    c.TickLabels = classes;
    numClasses = size(cmap, 1);
    c.Ticks = 1 / (numClasses * 2):1 / numClasses:1;
    c.TickLength = 0;
    
    % Begin attitude estimation
    categories = ["body", "panel"];
    numCategories = numel(categories);
    angles = zeros(numCategories, 1);
    centroids = zeros(numCategories, 2);
    
    % For each category, determine the angle of the calculated centroid
    for j = 1:numCategories
        [rows, cols] = find(C == categories(j)); 
        if isempty(rows)
            angles(j) = NaN;
            continue;
        end
        
        centroid = [mean(cols), mean(rows)];
        centroids(j, :) = centroid;
        X = [cols, rows] - centroid;
        [~, ~, V] = svd(X, 0);
        principalDirection = V(:, 1);
        angle = atan2d(principalDirection(2), principalDirection(1));
        
        % Force angle to be positive
        if angle < 0, angle = angle + 180; end
        angles(j) = angle;
    end
    
    hold off;
    
    % Compute rotation angles
    Yaw = abs(-angles(2)); 
    Pitch = abs(asind(norm(centroids(2,:) - centroids(1,:)) / 45)); 
    Roll = -abs(angles(1) - angles(2)); 
    
    % Read the STL file
    stlFile = 'hinode.stl';
    model = stlread(stlFile);
    vertices = model.Points;
    faces = model.ConnectivityList;
    
    % Convert degrees to radians
    rx = deg2rad(Pitch);
    ry = deg2rad(Yaw);
    rz = deg2rad(Roll);
    
    % Compute rotation matrices
    Rx = [1 0 0; 0 cos(rx) -sin(rx); 0 sin(rx) cos(rx)];
    Ry = [cos(ry) 0 sin(ry); 0 1 0; -sin(ry) 0 cos(ry)];
    Rz = [cos(rz) -sin(rz) 0; sin(rz) cos(rz) 0; 0 0 1];
    
    % Apply rotation
    R = Rz * Ry * Rx;
    rotatedVertices = (R * vertices')';
    rotatedModel = triangulation(faces, rotatedVertices);
    
    % Subplot 3: 3D Model
    subplot(3, 2, 3);
    trisurf(rotatedModel, 'FaceColor', 'magenta', 'EdgeColor', 'none');
    axis equal; camlight; lighting gouraud;
    title('Estimated Attitude');
    camproj('perspective');
    campos([-200, 0, 0]);
    axis off;
    
    % Subplot 4: Maneuver Representation
    subplot(3, 2, 4);
    hold on;
    
    % Coordinate frame correction to match STL visualization
    % Flip Z-axis (mirror in XY plane)
    frameCorrection = diag([1, -1, -1]);
    R_corrected = frameCorrection * R;
    
    % Compute angles the same way with new coordinate frame
    categories = ["body", "panel"];
    numCategories = numel(categories);
    angles = zeros(numCategories, 1);
    centroids = zeros(numCategories, 2);

    for j = 1:numCategories
        [rows, cols] = find(C == categories(j)); 
        if isempty(rows)
            angles(j) = NaN;
            continue;
        end
        centroid = [mean(cols), mean(rows)];
        centroids(j, :) = centroid;
        X = [cols, rows] - centroid;
        [~, ~, V] = svd(X, 0);
        principalDirection = V(:, 1);
        angle = atan2d(principalDirection(2), principalDirection(1));
        if angle < -90, angle = angle + 180;
        elseif angle > 90, angle = angle - 180;
        end
        angles(j) = angle;
    end

    % Compute rotation angles
    Yaw = -angles(2);  % Yaw from panel orientation
    Pitch = -angles(1);  % Pitch from body orientation
    Roll = angles(2) - angles(1); % Relative roll
    
    % Define known lengths
    panelLength = 195;
    bodyLength = 45;
    
    % Compute unit vectors based on detected angles
    panelVector = [cosd(Yaw), sind(Yaw)] * panelLength / 2;
    bodyVector = [cosd(Pitch), sind(Pitch)] * bodyLength / 2;
    
    hold on;
    
    % Observer (Camera) Location
    observer = [0; 0; 1];  % Arbitrary camera position in 3D space
    satellite = [0; 0; 0];  % Satellite at the origin
    
    % Compute radius of the sphere (distance from observer to satellite)
    r = norm(observer - satellite);
    
    % Generate a sphere centered at the satellite with radius r
    [x, y, z] = sphere(50); 
    surf(r*x, r*y, r*z, ...
        'FaceAlpha', 0.2, ...
        'EdgeColor', 'none', ...
        'FaceColor', 'blue');
    
    % Define the observers point
    obsv_pt = (observer / norm(observer)) * r;
    
    % Extract the longitudinal direction from rotation matrix
    body_direction = R_corrected(:, 1);
    body_direction = body_direction / norm(body_direction);
    
    % Project this direction onto the sphere of radius r
    target_pt = body_direction * r;
    
    % Plot points
    scatter3(satellite(1), satellite(2), satellite(3), 100, ...
        'k', 'filled', 'DisplayName', 'Satellite');
    scatter3(observer(1), observer(2), observer(3), 100, ...
        'r', 'filled', 'DisplayName', 'Observer (Camera)');
    scatter3(obsv_pt(1), obsv_pt(2), obsv_pt(3), 100, ...
        'g', 'filled', 'DisplayName', 'Radial Point (Observer Aligned)');
    scatter3(target_pt(1), target_pt(2), target_pt(3), 100, ...
        'b', 'filled', 'DisplayName', 'Radial Point (Body Aligned)');
    
    % Connect points with lines
    plot3([satellite(1), observer(1)], [satellite(2), observer(2)], ...
        [satellite(3), observer(3)], 'r--');
    plot3([satellite(1), obsv_pt(1)], [satellite(2), obsv_pt(2)], ...
        [satellite(3), obsv_pt(3)], 'g--');
    plot3([satellite(1), target_pt(1)], [satellite(2), target_pt(2)], ...
        [satellite(3), target_pt(3)], 'b--');
    
    % Set up the calculated maneuver arc
    num_arc_points = 50;
    arc_points = zeros(num_arc_points, 3);
    for k = 1:num_arc_points
        t = (k - 1) / (num_arc_points - 1);
        interp_vec = (1 - t) * obsv_pt + t * target_pt;
        arc_points(k, :) = (interp_vec / norm(interp_vec))' * r;
    end

    % Plot the arc with a dotted line
    plot3(arc_points(:, 1), arc_points(:, 2), arc_points(:, 3), ...
        'k:', 'LineWidth', 1.5, 'DisplayName', 'Arc on Sphere');
    
    % Formatting
    axis equal;
    grid on;
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    title('Relative Positions and Path');

    % Set orthographic projection and viewing angle to see full structure
    view([1, 1, 1]); % Diagonal view to visualize all axes
    camproj('orthographic');
    
    hold off;
    
    % Save the figure
    saveas(gcf, fullfile(outputFolder,...
        sprintf('segmentation_output_%03d.png', i)));
    close;
end
