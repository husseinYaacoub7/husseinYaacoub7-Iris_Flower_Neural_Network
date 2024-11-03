import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2

# Load the dataset
data = pd.read_csv('/content/IRIS.csv')

# Encode the target variable
encoder = LabelEncoder()
data['species'] = encoder.fit_transform(data['species'])

# Feature scaling
scaler = StandardScaler()
features = data.drop('species', axis=1)
scaled_features = scaler.fit_transform(features)

# Prepare the data
X = scaled_features
y = to_categorical(data['species'])  # One-hot encode target

# Split the dataset into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model using an Input layer
model = Sequential([
    Input(shape=(X_train.shape[1],)),  # Make sure X_train is already defined and loaded correctly before this step
    Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    BatchNormalization(),
    Dense(8, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    BatchNormalization(),
    Dense(y_train.shape[1], activation='softmax')  # Ensure y_train is defined and one-hot encoded
])

# Compile the model with updated learning rate argument
model.compile(optimizer=RMSprop(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[reduce_lr, early_stopping])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}, Test accuracy: {accuracy}')
