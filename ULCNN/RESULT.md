Epoch 00190: val_loss did not improve                                                                          
Epoch 191/200                                                                                                  
 - 317s - loss: 1.0046 - acc: 0.6384 - val_loss: 1.0303 - val_acc: 0.6299                                      
                                                                                                               
Epoch 00191: val_loss improved from 1.03169 to 1.03035, saving model to model/ULCNN_MN=6_N=16_KS=5.hdf5        
Epoch 192/200                                                                                                  
 - 320s - loss: 1.0044 - acc: 0.6385 - val_loss: 1.0325 - val_acc: 0.6292                                      
                                                                                                               
Epoch 00192: val_loss did not improve                                                                          
Epoch 193/200                                          
 - 319s - loss: 1.0044 - acc: 0.6383 - val_loss: 1.0339 - val_acc: 0.6290

Epoch 00193: val_loss did not improve                                                                          
Epoch 194/200                                          
 - 318s - loss: 1.0046 - acc: 0.6383 - val_loss: 1.0331 - val_acc: 0.6285

Epoch 00194: val_loss did not improve                                                                          
Epoch 195/200                                          
 - 315s - loss: 1.0047 - acc: 0.6385 - val_loss: 1.0304 - val_acc: 0.6302

Epoch 00195: val_loss did not improve
Epoch 196/200
 - 317s - loss: 1.0039 - acc: 0.6383 - val_loss: 1.0354 - val_acc: 0.6276

Epoch 00196: val_loss did not improve
Epoch 197/200
 - 315s - loss: 1.0045 - acc: 0.6384 - val_loss: 1.0322 - val_acc: 0.6297

Epoch 00197: val_loss did not improve
Epoch 198/200
 - 313s - loss: 1.0041 - acc: 0.6385 - val_loss: 1.0335 - val_acc: 0.6294

Epoch 00198: val_loss did not improve
Epoch 199/200
 - 317s - loss: 1.0045 - acc: 0.6386 - val_loss: 1.0324 - val_acc: 0.6295

Epoch 00199: val_loss did not improve
Epoch 200/200
 - 318s - loss: 1.0046 - acc: 0.6384 - val_loss: 1.0321 - val_acc: 0.6295

Epoch 00200: val_loss did not improve
Traceback (most recent call last):
  File "5ULCNN.py", line 176, in <module>
    model.load_weights(f"model/{filename}.hdf5")
  File "/home/lijunkai/anaconda3/envs/ulcnn/lib/python3.7/site-packages/keras/engine/topology.py", line 2652, in load_weights
    f, self.layers, reshape=reshape)
  File "/home/lijunkai/anaconda3/envs/ulcnn/lib/python3.7/site-packages/keras/engine/topology.py", line 3135, in load_weights_from_hdf5_group
    original_keras_version = f.attrs['keras_version'].decode('utf8')
AttributeError: 'str' object has no attribute 'decode'