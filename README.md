# Rodent MRI Bias Field Correction (rodbfc.py)

![GitHub](https://img.shields.io/github/license/your-username/rodbfc)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/your-username/rodbfc)
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/your-username/rodbfc/CI)

**rodbfc.py** is a Python module for performing bias field correction on rodent MRI images. This tool utilizes a pre-trained model to correct intensity inhomogeneities caused by the bias field present in MRI scans of rodents.

## Features
- Corrects bias field artifacts in rodent MRI images.
- Easy-to-use command-line interface.
- Uses a pre-trained model for accurate correction.

## Installation
To use **rodbfc.py**, you'll need to have Python installed. You can install the module and its dependencies using pip:

```bash
pip install rodbfc
```

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

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing
Contributions are welcome! Please read our [Contribution Guidelines](CONTRIBUTING.md) for more information on how to get started.

## Issues and Support
If you encounter any issues or need assistance, please create a [GitHub Issue](https://github.com/your-username/rodbfc/issues) in our repository.

## Acknowledgments
- This tool was developed by [Your Name] and [Contributor Name] in [Year].
- The bias field correction model used in this tool was trained on [Dataset Name], which is available at [Dataset Link].

---

**Note**: Replace `[Your Name]`, `[Contributor Name]`, `[Year]`, `[Dataset Name]`, `[Dataset Link]`, and `[your-username]` with appropriate information for your repository. Make sure to include a license file (e.g., LICENSE) and a contribution guidelines file (e.g., CONTRIBUTING.md) in your repository.