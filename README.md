How to use the model
1-Capture Hand Gesture Data:

Run the first part of the provided code to capture hand gesture data. This part will save images from your webcam into the ./data directory. Ensure your webcam is connected and functional.

2-Preprocess the Data:

Run the second part of the code to preprocess the captured hand gesture data. This step will extract hand landmarks from the captured images using MediaPipe and save the processed data in a pickle file named data.pickle.

3-Train the Model:

Execute the third part of the code to train the neural network model using the preprocessed hand gesture data. The model will learn to recognize hand gestures based on the extracted hand landmarks.

4-Test the Model:

After training, the model will be evaluated on the test data to assess its performance. The test accuracy will be displayed.

5-Save the Model:

The trained model will be saved to a file named model.h5 in the specified directory.

6-Use the Trained Model:

Finally, run the last part of the code to use the trained model for real-time hand gesture recognition. The webcam will capture frames, detect hand landmarks, and predict the hand gestures in real-time
