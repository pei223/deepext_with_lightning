# Basic interfaces of models.
## Classification
### Training input
- image: Tensor (batch size, channel, height, width)
- annotation: Tensor (batch size, classes)

### Training output
- Tensor (batch size, classes)

### Predict
- Numpy array (batch size, classes)


<br/><br/>


## Segmentation
### Training input
- image: Tensor (batch size, channel, height, width)
- annotation: Tensor (batch size, classes, height, width)
    - <b>include background class</b>

### Training output
- Tensor (batch size, classes, height, width)
    - <b>include background class</b>

### Predict
- Numpy array(batch size, classes, height, width)
    - <b>include background class</b>


<br/><br/>


## Object detection
### Training input
- image: Tensor (batch size, channel, height, width)
- annotation: Tensor (batch size, bounding box count, 5)
    - bounding box contains (x_min, y_min, x_max, y_max, class label)
    - if no object, bounding box and label is -1
    - <b>NOT include background class</b>
    

### Training output
- Unspecified
    - Example: Tuple (bounding box count, ), (bounding box count, ), (bounding box count, 4)
        - scores, classes, coordinates
        
### Predict
- Numpy array (batch size, bounding box count by batch(variable length), 6)
    - bounding box contains (x_min, y_min, x_max, y_max, label, score)


<br/><br/>




