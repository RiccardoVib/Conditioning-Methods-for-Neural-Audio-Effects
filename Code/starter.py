from Training import train


data_dir = '../../Files/CondPaper'
epochs = [30, 60]
units = 16
b_size = 600
lr = 3e-4
model = 'S4D'
conditioning = True
film = True
gaf = False

dataset = 'SphereC'

name_model = '_order3linGLU_post'
order = 3
glu = True
gcu = False

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

epochs = [1, 60]
dataset = '8D'

name_model = '_order3linGLU_post'
order = 3

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


dataset = 'SphereC'

name_model = '_order3linGLU_post'
order = 5


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

dataset = '8D'

name_model = '_order5linGLU_post'
order = 5

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

dataset = 'SphereC'

name_model = '_order3linGCU_post'
order = 3
glu = False
gcu = True

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

dataset = '8D'

name_model = '_order3linGCU_post'
order = 3

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


dataset = 'SphereC'

name_model = '_order5linGCU_post'
order = 5

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

dataset = '8D'

name_model = '_order5linGCU_post'
order = 5

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