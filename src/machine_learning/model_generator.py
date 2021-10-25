import tensorflow as tf


def get_model_gru(embedding_len, drop_out_rate, learning_rate, l1, l2, nd):
    opt = tf.optimizers.Adam(learning_rate)
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(embedding_len, 1,)),
        tf.keras.layers.ActivityRegularization(l1=l1, l2=l2),
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(drop_out_rate),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(nd, activation='softmax')
    ])
    # model.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
    model.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy,
                  metrics=[tf.keras.metrics.mae, tf.keras.metrics.categorical_crossentropy, ['accuracy']])
    print(model.summary())
    return model


def get_model_lstm(embedding_len, drop_out_rate, learning_rate, l1, l2, nd):
    opt = tf.optimizers.Adam(learning_rate)
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(embedding_len,)),
        tf.keras.layers.ActivityRegularization(l1=l1, l2=l2),
        tf.keras.layers.Embedding(10000, 32),
        tf.keras.layers.LSTM(32, ),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(drop_out_rate),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(nd, activation='softmax')
    ])
    model.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
    # model.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy,
    #               metrics=[tf.keras.metrics.mae, tf.keras.metrics.categorical_crossentropy, ['accuracy']])
    print(model.summary())
    return model


def get_model_rnn(embedding_len, drop_out_rate, learning_rate, l1, l2, nd):
    opt = tf.optimizers.Adam(learning_rate)
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(embedding_len, 1,)),
        tf.keras.layers.ActivityRegularization(l1=l1, l2=l2),
        tf.keras.layers.SimpleRNN(128, ),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(drop_out_rate),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(nd, activation='softmax')
    ])
    # model.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
    model.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy,
                  metrics=[tf.keras.metrics.mae, tf.keras.metrics.categorical_crossentropy, ['accuracy']])
    print(model.summary())
    return model


def get_model_cnn(embedding_len, drop_out_rate, learning_rate, l1, l2, nd):
    opt = tf.optimizers.Adam(learning_rate)
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=[60, 37, 1]),
        tf.keras.layers.Conv2D(16, (3, 3), strides=(1, 1), padding="SAME", activation=tf.nn.relu),
        tf.keras.layers.MaxPool2D((1, 1), strides=(1, 1), padding='valid'),
        tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding="SAME", activation=tf.nn.relu),
        tf.keras.layers.MaxPool2D((1, 1), strides=(1, 1), padding='valid'),
        tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding="SAME", activation=tf.nn.relu),
        tf.keras.layers.MaxPool2D((2, 2), strides=(1, 1), padding='valid'),
        tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding="SAME", activation=tf.nn.relu),
        tf.keras.layers.MaxPool2D((2, 2), strides=(1, 1), padding='valid'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(drop_out_rate),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dropout(drop_out_rate),
        tf.keras.layers.Dense(nd, activation=tf.nn.softmax)
    ])
    model.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
    print(model.summary())
    return model


def get_model_dense(embedding_len, drop_out_rate, learning_rate, l1, l2, nd):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(embedding_len,)),
        tf.keras.layers.ActivityRegularization(l1=l1, l2=l2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(drop_out_rate),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(drop_out_rate),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(drop_out_rate),
        tf.keras.layers.Dense(nd, activation='softmax')
    ])
    print(model.summary())

    optimizer = tf.optimizers.Adadelta(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    # opt = tf.optimizers.Adam(learning_rate)
    # # model.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
    # # model.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy,
    # #               metrics=[tf.keras.metrics.mae, tf.keras.metrics.categorical_crossentropy, ['accuracy']])
    # model.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy,
    #               metrics=[tf.keras.metrics.categorical_crossentropy, ['accuracy']])
    return model
