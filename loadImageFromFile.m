function img = loadImageFromFile(filename)
    data = load(filename);      % Load the .mat file
    img  = data.img;            % Extract the 'img' variable
end