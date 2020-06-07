import model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import load_model

GoogLeNet = model.model_v2()
#GoogLeNet = load_model('./model/chinese_character_classification_inception_v13_nofc.hdf5')
#squeezenet = model.SqueezeNet(input_shape=(64, 64, 1))
#squeezenet.summary()

GoogLeNet.summary()
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
adam = Adam(lr=0.01)
GoogLeNet.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#squeezenet.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# 데이터 불러오기
train_generator = ImageDataGenerator(rescale=1./255)

train_data_flow = train_generator.flow_from_directory('./database/train_56',
                                                     target_size=(56, 56),
                                                     batch_size=48,
                                                     color_mode='grayscale',
                                                     class_mode='categorical')

test_generator = ImageDataGenerator(rescale=1./255)

test_data_flow = test_generator.flow_from_directory('./database/test_56',
                                                     target_size=(56, 56),
                                                     batch_size=48,
                                                     color_mode='grayscale',
                                                     class_mode='categorical')

model_path = './model/chinese_character_classification_inception_v15_nofc.hdf5'

checkpoint = ModelCheckpoint(filepath=model_path,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True)

cb_early_stopping = EarlyStopping(monitor='val_loss',
                                  patience=50)

tensorboard = TensorBoard(log_dir='C:\DeepLearning\한자 손글씨 학습\logs15_nofc',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)

# 모델 학습
history = GoogLeNet.fit_generator(train_data_flow,
                                  epochs=500,
                                  verbose=1,
                                  validation_data=test_data_flow,
                                  callbacks=[checkpoint, cb_early_stopping, tensorboard])