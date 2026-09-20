function label = loadLabelFromFile(filename)
    data  = load(filename);     % Load the .mat file
    label = data.label;         % Extract the 'label' variable
end