from Training import train


"""
main script

"""

# data_dir: the directory in which datasets are stored
data_dir = '../Files/'
epochs = 200 # number of epochs
units = 16 # number of model's units
b_size = 600 # batch size

lr = 3e-4 # initial learning rate

model = 'S4D'
datasets = ['SphereC', '8D'] # name of the dataset


conditioning = True # if conditioning included
film = True # if use Film layer
gaf = False # if use GAF layer
order = 3 # order of transformation in Film
glu = True  # if use GLU
gcu = False # if use GCU


name_model = ''

for dataset in datasets:
      train(data_dir=data_dir,
            save_folder=model+dataset+name_model,
            dataset=dataset,
            b_size=b_size,
            order=order,
            glu=glu,
            gcu=gcu,
            gaf=gaf,
            conditioning=conditioning,
            act=None,
            film=film,
            learning_rate=lr,
            units=units,
            epochs=epochs,
            model=model,
            inference=False)
