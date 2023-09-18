# Rodent MRI Bias Field Correction (rodbfc.py)

**rodbfc.py** is a Python module for performing bias field correction on rodent MRI images. This tool utilizes a pre-trained model to correct intensity inhomogeneities caused by the bias field present in MRI scans of rodents.

## Features
- Corrects bias field artifacts in rodent MRI images.
- Easy-to-use command-line interface.
- Uses a pre-trained model for accurate correction.

## Installation
To use **rodbfc.py**, you'll need to have Python 3 installed. Also please install `monai`, `pytorch`, and `nilearn` python libraries. 
You can install the module by cloning this repository.


## Usage
```bash
rodbfc.py [-h] -i INPUT -m MODEL -o OUTPUT
```

### Arguments:
- `-h, --help`: Show the help message and exit.
- `-i INPUT, --input INPUT`: Path to the input MRI image file (in NIfTI format).
- `-m MODEL, --model MODEL`: Path to the pre-trained bias field correction model.
- `-o OUTPUT, --output OUTPUT`: Path to save the corrected output MRI image (in NIfTI format).

### Example:
```bash
rodbfc.py -i input_image.nii.gz -m model_weights.pth -o output_image_corrected.nii.gz
```

## Training the model
A pretrained model is included in models directory. If you want to train a new model, you can use the ```main_training.ipynb``` notebook. 


## License
This project is licensed under the GPL (V3) License - see the [LICENSE](LICENSE_GNU_v3.md) file for details.

## Contributing
Contributions are welcome! Please contact (ajoshi@usc.edu)[mailto:ajoshi@usc.edu] for further discussion.

## Issues and Support
If you encounter any issues or need assistance, please create a [Jira Issue](https://bitbucket.org/brainsuite/rodbfc/jira) in our repository.

## Acknowledgments
- This tool was developed by [Anand A Joshi] and [Ronald Salloum] in [2023].
- The bias field correction model used in this tool was trained on [Dataset Name], which is available at [Dataset Link].

---

