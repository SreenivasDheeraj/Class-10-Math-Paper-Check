1. Feature Extraction - Encoder Component
- We will try and use ResNet for this and upgrade as need arises
- The aim is to capture the intricate details of Latin, Devanagari, and other script shapes, which are visually more complex than standard printed fonts.

2. Sequence Recognition - Decoder Component
- We will a Convolutional-Bidirectional LSTM-Connectionist Temporal Classification (CNN-BiLSTM-CTC)

The CNN converts the 2D image data into a 1D sequence of features suitable for the LSTM. Using ResNet (Residual Network) as the backbone is preferred due to its superior performance in extracting robust features from complex images.


Also note that the documents could be rotated and you'd need to make a choice based on the orientation of the document, each docuemnt could be in a different direction.
