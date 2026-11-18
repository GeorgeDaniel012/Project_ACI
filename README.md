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

### NAFNet Setup
Download `NAFNet-GoPro-width64.pth` from [here](https://drive.google.com/file/d/1S0PVRbyTakYY9a82kujgZLbMihfNBLfC), and place it inside `nafnet/experiments/pretrained_models/`.

### Restormer Setup
Get `motion_deblurring.pth` from [this link](https://drive.google.com/file/d/1pwcOhDS5Erzk8yfAbu7pXTud606SB4-L), and put it in `Restormer/Motion_Deblurring/pretrained_models/`.

# Run instructions (Milestone 1)
Run the `demo.py` script in the CLI, like so:
```
python .\demo.py [image_path]
```

Note that the `image_path` argument is optional. If it's not provided, the application will use a default image instead.

# Disclaimers
The models recommended for use have been pretrained on the GoPro dataset. However, they can be replaced by other models.

### GoPro dataset
You can get the dataset from [here](https://drive.google.com/file/d/1KStHiZn5TNm2mo3OLZLjnRvd0vVFCI0W).