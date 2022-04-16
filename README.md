<div align="center">

# **Python library with Deep Neural Networks for Bi-Temporal Image Change Detection based on [PyTorch](https://pytorch.org/).**

</div>

**This is a Python library, named 'changeDetector', with Deep Neural Networks for Bi-Temporal Image Change Detection based on [PyTorch](https://pytorch.org/). There are some functionalities in this library for change detection tasks.** 

<br>

## ***1. Chunk Change Detection Dataset***

<div style="margin-left: 25px;">
Create a chunker object for 'chunk' an image to sub-images.

```python
import changeDetector as cd

#define a chunker object
chunker = cd.chunk_data(
    number_tiles    #an integer that shows the number of tiles requested for each image
)

#chunk an image to sub-images
save_directory = chunker.chunk(
    data_root,      #a string path where image data are stored
    save_directory  #a string path where chunked data should be stored, default is None
)
```

</div>

<br>

## ***2. Prepare Change Detection Dataset***

<div style="margin-left: 25px;">
Prepare a custom dataset for change detection according the following data structure:

```python
├── data_root
│   ├── train
│   │   ├── base_dir
│   │   ├── target_dir
│   │   ├── label_dir
│   ├── val
│   │   ├── base_dir
│   │   ├── target_dir
│   │   ├── label_dir
│   ├── optional [test]
│   │   ├── base_dir
│   │   ├── target_dir
│   │   ├── optional [label_dir]
```

```python
import changeDetector as cd

ds = cd.prepare_data(
    data_root,          #a string path of change detection data structure
    base_dir,           #a string name of the base images directory, default is 'A'
    base_img_suffix,    #a string indicator of the base images suffix, default is '*.png'
    target_dir,         #a string name of the target images directory, default is 'B'
    target_img_suffix,  #a string indicator of the target images suffix, default is '*.png'
    label_dir,          #a string name of the ground truth masks directory, default is 'label'
    label_mask_suffix,  #a string indicator of the masks suffix, default is '*.png'
    size,               #an integer of the model input size, default is 256
    transform           #a combination of albumentations library transforms, default is None
)
```

</div>

<br>

## ***3. Prepare Change Detection DataLoader***

<div style="margin-left: 25px;">
Prepare a custom dataloader for change detection.

```python
import changeDetector as cd

dl = cd.prepare_dataloader(
    dataset,     #a custom dataset prepared by 'prepare_data' functionality
    batch_size,  #the number of samples used for forward/backward pass, default is 1
    **kwargs,    #any compatible arguments of torch.utils.data.DataLoader
)
```

</div>

<br>

## ***4. Prepare Change Detection Model***

<div style="margin-left: 25px;">
Create a UNet/UNet++ fully convolutional neural network for change detection.

```python
import changeDetector as cd

model = cd.UNet(
    in_channels,            #an integer number of input channels for the model, default is 3
    encoder_name,           #an encoder name (of the timm library), default is 'resnet34'
    pretrained,             #a boolean indicator of using pretrained weight, default is True
    decoder_channels,       #in_channels in decoder part, default is (256, 128, 64, 32, 16)
    encoder_fusion_type,    #fusion type ('sum', 'diff', and 'concat'), default is 'concat'
    decoder_attention_type, #attentiona module (of the timm library), default is None
    classes                 #number of output classes/channels
)
```

```python
import changeDetector as cd

model = cd.UNetPlusPlus(
    in_channels,            #an integer number of input channels for the model, default is 3
    encoder_name,           #an encoder name (of the timm library), default is 'resnet34'
    pretrained,             #a boolean indicator of using pretrained weight, default is True
    decoder_channels,       #in_channels in decoder part, default is (256, 128, 64, 32, 16)
    encoder_fusion_type,    #fusion type ('sum', 'diff', and 'concat'), default is 'concat'
    decoder_attention_type, #attentiona module (of the timm library), default is None
    classes                 #number of output classes/channels
)
```
</div>

<br>

## ***5. Prepare Change Detection Learner***

<div style="margin-left: 25px;">
create a learning object for change detection.

```python
import changeDetector as cd

learner = cd.prepare_learning(
    model,      #a change detection deep net that inherits properties of 'torch.nn.Module'
    loss,       #a loss function that inherits properties of 'torch.nn.modules.loss._Loss'
    optim,      #an optimizer that inherits properties of 'torch.optim.Optimizer'
    scheduler,  #a decay rate that inherits properties of 'torch.optim', default is None
    num_epoch,  #number of training epochs over dataloaders, default is 25
    device,     #use 'cpu' or 'cuda' for training change detection model, default is 'cpu'
    metrics,    #detection accuracy, default is ('accuracy', 'kappa', 'fscore', 'similarity')
    verbose,    #a boolean indicator for showing training/validation results, default is True,
    save_dir    #save directory for the best models, default is None (main directory)
    )
```

```python
#train change detection net
train_logs, valid_logs = learner.train(data_loader=train_dl)
```

```python
#test change detection net
test_logs = learner.predict(data_loader=test_dl)
```
</div>

<br>

## ***6. Dechunk Change Detection Dataset***

<div style="margin-left: 25px;">

 Create a chunker object for 'dechunk' sub-images to an image.

```python
import changeDetector as cd

#define a chunker object
chunker = cd.chunk_data(
    number_tiles    #an integer that shows the number of tiles requested for each image
)

#dechunk sub-images to an image
save_directory = chunker.dechunk(
    data_root,      #a string path where image data are stored
    pattern,        #a regular expression operations, default is re.compile('_\d{2}_\d{2}\.')
    save_directory, #a string path where chunked data should be stored, default is None
    del_chunk       #a boolean indicator of deleting chunked data 
)
```

</div>

<br>

## ***7. Prepare Change Detection Reporter***

<div style="margin-left: 25px;">
Report and export a clean version of change detection architecture.

```python
import changeDetector as cd

#print change detection net summary
cd.summary(model=model, input_size=((1, 3, 256, 256), (1, 3, 256, 256)))
```

```python
import changeDetector as cd

#save change detection net as onnx
cd.export_onnx(model=model, input_size=((1, 3, 256, 256), (1, 3, 256, 256)), filename='change detection.onnx', input_names=['base image', 'target image'], output_names=['change map'], opset_version=11)
```
<br>

<div align="center">