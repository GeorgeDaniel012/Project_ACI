# Setup instructions
First of all, install all dependencies from `requirements.txt` (for example, by running `pip install -r requirements.txt`).

Clone the following repositories:
```
git clone https://github.com/vita-group/deblurganv2
git clone https://github.com/megvii-research/nafnet
git clone https://github.com/swz30/Restormer
```

Once the repositories have been cloned, we need to download the corresponding pretrained models.

### DeblurGANv2 Setup
Download `fpn_inception.h5` from [this link](https://drive.google.com/uc?id=1UXcsRVW-6KF23_TNzxw-xC0SzaMfXOaR), and paste it into `deblurganv2/`.


# Run instructions (Milestone 1)
Run the `demo.py` script in the CLI, like so:
```
python .\demo.py [image_path]
```

Note that the `image_path` argument is optional. If it's not provided, the application will use a default image instead.