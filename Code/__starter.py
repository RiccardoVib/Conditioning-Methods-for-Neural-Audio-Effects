# Copyright (C) 2024 Riccardo Simionato, University of Oslo
# Inquiries: riccardo.simionato.vib@gmail.com.com
#
# This code is free software: you can redistribute it and/or modify it under the terms
# of the GNU Lesser General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
#
# This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Less General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along with this code.
# If not, see <http://www.gnu.org/licenses/>.
#
# If you use this code or any part of it in any program or publication, please acknowledge
# its authors by adding a reference to this publication:
#
# R. Simionato, 2024, "Conditioning Methods for Neural Audio Effects" in proceedings of Sound and Music Computing, Porto, Portugal.

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
